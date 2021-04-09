#pragma once

template <typename T, typename F>
__global__ void __kernel__cuda_map(T *data, const int len, F op) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_thd = blockDim.x * gridDim.x;
    for (auto i = tid; i < len; i += num_thd)
        data[i] = op(data, i);
}
namespace culib {
template <typename T, typename F>
void cuda_map(T *data, const int len, F op, cudaStream_t stream=0) {
    int num_blk, num_thd;
    cudaOccupancyMaxPotentialBlockSize(&num_blk, &num_thd,
                                       __kernel__cuda_map<T, F>);
    __kernel__cuda_map<<<num_blk, num_thd, 0, stream>>>(data, len, op);
}
/*
 * A = A + B
 */
template <typename T>
void cuda_vecadd(T *A, const T *B, const int size, cudaStream_t stream=0) {
    const auto add_op = [B] __device__(T * data, int i) -> T {
        return data[i] + B[i];
    };
    cuda_map(A, size, add_op, stream);
}
} // namespace culib