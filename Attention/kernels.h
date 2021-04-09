#pragma once
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.hpp>
#include <utils.h>

namespace cg = cooperative_groups;

__global__ void __kernel_batch_softmax(half *__restrict__ data,
                                       const int num_batch, const int size,
                                       const int num);
__global__ void
__kernel_multi_head(const half *__restrict__ Q, const half *__restrict__ K,
                    const half *__restrict__ V, half *__restrict__ Z,
                    const int kdim, const int vdim, const int seq_len,
                    const int num_head, const half *__restrict__ mask);
// __global__ void
// __kernel_multi_head_full(const half *__restrict__ Q, const half *__restrict__
// K,
//                          const half *__restrict__ V, half *__restrict__ Z,
//                          const int kdim, const int vdim, const int seq_len,
//                          const int num_head, const half *__restrict__ mask);
__global__ void
__kernel_multi_head_tile(const half *__restrict__ Q, const half *__restrict__ K,
                         const half *__restrict__ V, half *__restrict__ Z,
                         const int kdim, const int vdim, const int seq_len,
                         const int num_head, const half *__restrict__ mask,
                         const int tile_size);

__global__ void __kernel_multi_head_sharedQK(const half *__restrict__ QK,
                                             const half *__restrict__ V,
                                             half *__restrict__ Z,
                                             const int vdim, const int seq_len,
                                             const int num_head,
                                             const half *__restrict__ mask);

/* utils */

template <typename T>
__device__ void warp_load_row(T *__restrict__ dst, const T *__restrict__ src,
                              const int row_id, const int row_len,
                              const int stride) {
    const auto ld = row_id * stride;
    for (int i = threadIdx.x % 32; i < row_len; i += 32)
        dst[i] = src[(i / stride) * 256 + ld + i % stride];
}

template <typename T>
__device__ void warp_store_row(T *__restrict__ dst, const T *__restrict__ src,
                               const int row_id, const int row_len,
                               const int stride) {
    const auto ld = row_id * stride;
    for (int i = threadIdx.x % 32; i < row_len; i += 32)
        dst[(i / stride) * 256 + ld + i % stride] = src[i];
}

/*
 * It returns the pointer of the top-left corner of give block in a matrix.
 * Assume the matrix is stored in a row-major array.
 * It needs the number of columns of the matrix (leading dimension).
 */
template <typename T, int SIZE = 16>
__device__ T *get_blk_start(T *data, const int row_blk, const int col_blk,
                            const int stride) {
    return &data[row_blk * SIZE * stride + SIZE * col_blk];
}

template <int num_thd>
__global__ void __launch_bounds__(num_thd)
    __kernel_multi_head_sharedQK(const half *__restrict__ QK,
                                 const half *__restrict__ V,
                                 half *__restrict__ Z, const int vdim,
                                 const int seq_len, const int num_head,
                                 const half *__restrict__ mask) {
    constexpr int FP16_skew = 16;
    using frag_t = culib::mma::mma_t<16, 16, 16>;
    extern __shared__ half smem[]; // 16 * seq_len
    auto cta = cg::this_thread_block();
    const auto head_id = cta.group_index().y;
    const auto z_row = cta.group_index().x;
    auto warp = cg::tiled_partition<32>(cta);
    const auto warp_id = warp.meta_group_rank();
    constexpr auto num_warp = num_thd / 32;
    const auto ldm = seq_len + FP16_skew;
    const auto seq_len2 = seq_len >> 1;
    for (auto r = warp_id; r < 16; r += num_warp) {
        const auto row = 16 * z_row + r;
        auto dst = &smem[r * ldm];
        auto src = &QK[head_id * seq_len * seq_len + seq_len * row];
        cg::memcpy_async(warp, dst, src, sizeof(half) * seq_len);
        cg::wait(warp);
        // mask and find the max
        half val_max = half_zero;
        for (auto i = warp.thread_rank(); i < seq_len2; i += 32) {
            const auto temp = dst[i] + mask[seq_len * row + i];
            dst[i] = temp;
            val_max = val_max > temp ? val_max : temp;
        }
        warp.sync();
        const auto max = cg::reduce(warp, val_max, cg::greater<half>());
        half2 max2{max, max};
        // compute the sum of exp-ed and shifted array
        half2 val_sum2{half_zero, half_zero};
        auto dst2 = reinterpret_cast<half2 *>(dst);
        for (auto i = warp.thread_rank(); i < seq_len2; i += 32) {
            const auto temp2 = h2exp2(dst2[i] - max2);
            val_sum2 += temp2;
            dst2[i] = temp2;
        }
        warp.sync();
        const auto sum2 = cg::reduce(warp, val_sum2, cg::plus<half2>());
        // update with softmax scaling
        const auto sum = sum2.x + sum2.y;
        half2 one_over_sum =
            __h2div(half2{half_one, half_one}, half2{sum, sum});
        for (auto i = warp.thread_rank(); i < seq_len2; i += 32) {
            dst2[i] = dst2[i] * one_over_sum;
        }
    }
    cta.sync();
    const auto vhead_dim = vdim / num_head;
    for (auto VC = warp_id; VC < vhead_dim / 16; VC += num_warp) {
        frag_t::a_t<wmma::row_major> a_frag;
        frag_t::b_t<wmma::row_major> b_frag;
        frag_t::c_t<half> c_frag;
        wmma::fill_fragment(c_frag, half_zero);
        for (int i = 0; i < seq_len; i += 16) {
            auto V_ptr = &V[vdim * i + head_id * vhead_dim + VC * 16];
            wmma::load_matrix_sync(a_frag, smem + i, ldm);
            wmma::load_matrix_sync(b_frag, V_ptr, vdim);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        auto res = &Z[blockIdx.x * 16 * vdim + head_id * vhead_dim + VC * 16];
        wmma::store_matrix_sync(res, c_frag, vdim, wmma::mem_row_major);
    }
}

template <typename _TyGroup>
__device__ void softmax_blk(const _TyGroup &cta, half *data, const int size,
                            const int num, const int ldm) {
    auto warp = cg::tiled_partition<32>(cta);
    const auto warp_id = warp.meta_group_rank();
    const auto lane_id = warp.thread_rank();
    const auto num_warp = warp.meta_group_size();
    const auto size2 = size >> 1;
    for (int row = warp_id; row < num; row += num_warp) {
        auto row_ptr = data + row * ldm;
        // find the max
        half val_max = half_zero;
        for (auto i = lane_id; i < size; i += 32) {
            const auto temp = row_ptr[i];
            val_max = val_max > temp ? val_max : temp;
        }
        warp.sync();
        const auto max = cg::reduce(warp, val_max, cg::greater<half>());
        half2 max2{max, max};
        // compute the sum of exp-ed and shifted array
        auto row_ptr2 = reinterpret_cast<half2 *>(row_ptr);
        half2 val_sum2{half_zero, half_zero};
        for (auto i = lane_id; i < size2; i += 32) {
            const auto temp = h2exp2(__hsub2(row_ptr2[i], max2));
            val_sum2 += temp;
            row_ptr2[i] = temp;
        }
        warp.sync();
        const auto sum2 = cg::reduce(warp, val_sum2, cg::plus<half2>());
        // update with softmax scaling
        const auto sum = sum2.x + sum2.y;
        half2 one_over_sum =
            __h2div(half2{half_one, half_one}, half2{sum, sum});
        for (auto i = lane_id; i < size2; i += 32) {
            row_ptr2[i] = row_ptr2[i] * one_over_sum;
        }
    }
    cta.sync();
}