#include "LayerNorm.h"
#include <reduce.cuh>

__device__ unsigned dynamic_smem_size() {
    unsigned ret;
    asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    return ret;
}
__global__ void __kernel_LayerNorm(const half *__restrict__ data,
                                   half *__restrict__ out, const int size,
                                   const half *__restrict__ weight,
                                   const half *__restrict__ bias,
                                   const half eps) {
    extern __shared__ half smem[];
    const auto smem_size = dynamic_smem_size() / sizeof(half);
    const auto offset = size * blockIdx.x;
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        smem[i] = data[i + offset];
    for (int i = size + threadIdx.x; i < smem_size; i += blockDim.x)
        smem[i] = 0;
    __syncthreads();
    const auto avg = block_sum(smem, smem_size) / __int2half_rn(size);
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        smem[i] = data[i + offset] * data[i + offset];
    for (int i = size + threadIdx.x; i < smem_size; i += blockDim.x)
        smem[i] = 0;
    __syncthreads();
    const auto avg_square = block_sum(smem, smem_size) / __int2half_rn(size);
    const auto var = avg_square - avg * avg;
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        out[i + offset] =
            ((data[i + offset] - avg) / hsqrt(var + __float2half(eps))) *
                weight[i] +
            bias[i];
}

LayerNorm::LayerNorm(const half *w, const half *b, int _size)
    : size(_size), eps(__float2half(1e-5)), weight(w, _size), bias(b, _size) {}

unsigned int nextPowerOf2(unsigned int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

void LayerNorm::forward(half *out, const half *src, int num) const {
    const auto smem_size = nextPowerOf2(size) * sizeof(half);
    const auto num_thd = culib::util::cuda_num_thd(size / 2);
    __kernel_LayerNorm<<<num, num_thd, smem_size>>>(
        src, out, size, weight.get(), bias.get(), eps);
    cudaChk(cudaDeviceSynchronize());
}

std::unique_ptr<LayerNorm> gen_LN(int size) {
    std::vector<half> w(size, __float2half(0.2));
    std::vector<half> b(size, __float2half(0.2));
    auto res = std::make_unique<LayerNorm>(w.data(), b.data(), size);
    return std::move(res);
}