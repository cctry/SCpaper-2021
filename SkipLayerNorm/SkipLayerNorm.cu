#include "SkipLayerNorm.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void
__kernel_skLayerNorm_smem(const half *__restrict__ data,
                          const half *__restrict__ skip, half *__restrict__ out,
                          const int size, const half *__restrict__ weight,
                          const half *__restrict__ bias, const half eps) {
    extern __shared__ half smem[]; // size + num_warp * 2
    auto cta = cg::this_thread_block();
    auto warp = cg::coalesced_threads();
    const auto num_warp = cta.size() / 32;
    const auto warp_id = cta.thread_rank() / 32;
    auto src = &data[blockIdx.x * size];
    // load one row
    cg::memcpy_async(cta, smem, src, sizeof(half) * size);
    cg::wait(cta);
    // add the skip
    auto smem2 = reinterpret_cast<half2 *>(smem);
    auto skip2 = reinterpret_cast<const half2 *>(&skip[blockIdx.x * size]);
    for (int i = cta.thread_rank(); i < size / 2; i += cta.size())
        smem2[i] += skip2[i];
    cta.sync();
    // loop over and sum the array in a warp
    half2 sum{half_zero, half_zero};
    for (int i = cta.thread_rank(); i < size; i += cta.size()) {
        const auto val = smem[i];
        const auto square = __hmul(val, val);
        sum = __hadd2(sum, half2{square, val});
        smem[i] = val;
    }
    auto res = reinterpret_cast<half2 *>(&smem[size]) + warp_id;
    sum = cg::reduce(warp, sum, cg::plus<half2>());
    // write down warp sum
    if (warp.thread_rank() == 0)
        *res = sum;
    cta.sync();
    // get the overall sum
    sum = half2{half_zero, half_zero};
    for (int i = warp.thread_rank(); i < num_warp; i += 32) {
        sum += reinterpret_cast<half2 *>(&smem[size])[i];
    }
    warp.sync();
    sum = cg::reduce(warp, sum, cg::plus<half2>());
    // get avg and var
    const auto size_h = __int2half_rn(size);
    sum = __h2div(sum, half2{size_h, size_h});
    const half2 avg2{sum.y, sum.y};
    const half2 var2{sum.x - sum.y, sum.x - sum.y};
    // compute LN
    const auto n = h2sqrt(var2 + half2{eps, eps});
    auto weight2 = reinterpret_cast<const half2 *>(weight);
    auto bias2 = reinterpret_cast<const half2 *>(bias);
    for (int i = cta.thread_rank(); i < size / 2; i += cta.size()) {
        const auto w2 = weight2[i];
        const auto b2 = bias2[i];
        const auto x2 = smem2[i];
        const auto y2 = __h2div(x2 - avg2, n);
        smem2[i] = y2 * w2 + b2;
    }
    auto dst = &out[blockIdx.x * size];
    cta.sync();
    cg::memcpy_async(cta, dst, smem, sizeof(half) * size);
    cg::wait(cta);
}

SkipLayerNorm::SkipLayerNorm(const half *w, const half *b, int _size)
    : size(_size), eps(__float2half(1e-5)), weight(w, _size), bias(b, _size) {}

void SkipLayerNorm::forward(half *out, const half *src, const half *skip,
                            int num) const {
    auto smem = [&](int n) { return sizeof(half) * (size + (n / 16)); };
    int num_thd, num_blk;
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &num_blk, &num_thd, __kernel_skLayerNorm_smem, smem, size);
    __kernel_skLayerNorm_smem<<<num, num_thd, smem(num_thd)>>>(
        src, skip, out, size, weight.get(), bias.get(), eps);
    // cudaChk(cudaDeviceSynchronize());
}

std::unique_ptr<SkipLayerNorm> gen_skLN(int size) {
    std::vector<half> w(size, __float2half(0.2));
    std::vector<half> b(size, __float2half(0.2));
    auto res = std::make_unique<SkipLayerNorm>(w.data(), b.data(), size);
    return std::move(res);
}