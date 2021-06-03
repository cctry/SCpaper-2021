#pragma once
#include "kernels.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.hpp>
#include <utils.h>

namespace cg = cooperative_groups;

constexpr int FP16_skew = 16;

template <int num_thd>
__global__ void __launch_bounds__(num_thd)
    __kernel_multi_head_full_skew_warpSFM(const half *__restrict__ Q,
                                          const half *__restrict__ K,
                                          const half *__restrict__ V,
                                          half *__restrict__ Z, const int kdim,
                                          const int vdim, const int seq_len,
                                          const int num_head,
                                          const half *__restrict__ mask) {
    // blockIdx.x: block row id
    // blockIdx.y: head_id
    using frag_t = culib::mma::mma_t<16, 16, 16>;
    extern __shared__ half smem[];
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    const auto warp_id = warp.meta_group_rank();
    const auto lane_id = warp.thread_rank();
    constexpr int num_warp = num_thd / 32;
    const auto head_dim = kdim / num_head;
    auto Q_ptr = &Q[16 * blockIdx.x * kdim + head_dim * blockIdx.y];
    auto temp_Q = smem;
    const auto ldQ = head_dim + FP16_skew;
    const auto ldRow = seq_len + FP16_skew;
    for (int r = warp_id; r < 16; r += num_warp) {
        auto dst = &temp_Q[r * ldQ];
        auto src = &Q_ptr[r * kdim];
        cg::memcpy_async(warp, dst, src, sizeof(half) * head_dim);
    }
    cta.sync();
    frag_t::a_t<wmma::row_major> a_frag;
    frag_t::b_t<wmma::col_major> b_frag;
    frag_t::c_t<half> c_frag;
    auto temp_row = &smem[16 * ldQ];
    for (int KR = warp_id * 16; KR < seq_len; KR += num_warp * 16) {
        auto K_ptr = &K[KR * kdim + head_dim * blockIdx.y];
        wmma::fill_fragment(c_frag, half_zero);
        for (int i = 0; i < head_dim; i += 16) {
            wmma::load_matrix_sync(a_frag, smem + i, ldQ);
            wmma::load_matrix_sync(b_frag, K_ptr + i, kdim);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(temp_row + KR, c_frag, ldRow,
                                wmma::mem_row_major);
    }
    cta.sync();
    // mask

    // const auto mask_base = &mask[blockIdx.x * 16 * seq_len];
    // for (int i = threadIdx.x; i < seq_len * 16; i += blockDim.x) {
    //     const auto r = i / seq_len;
    //     const auto c = i % seq_len;
    //     temp_row[r * ldRow + c] += mask_base[i];
    // }

    auto mask_base =
        reinterpret_cast<const half2 *>(&mask[blockIdx.x * 16 * seq_len]);
    auto temp_row2 = reinterpret_cast<half2 *>(temp_row);
    const auto ldRow2 = ldRow >> 1;
    const auto seq_len2 = seq_len >> 1;
    for (int i = threadIdx.x; i < seq_len2 * 16; i += num_thd) {
        const auto r = i / seq_len2;
        const auto c = i % seq_len2;
        temp_row2[r * ldRow2 + c] += mask_base[i];
    }
    cta.sync();
    softmax_blk(cta, temp_row, seq_len, 16, ldRow);
    // for (int row = warp_id; row < 16; row += num_warp) {
    //     auto row_ptr = temp_row + row * ldRow;
    //     // find the max
    //     half val_max = half_zero;
    //     for (auto i = warp.thread_rank(); i < seq_len; i += 32) {
    //         const auto temp = row_ptr[i];
    //         val_max = val_max > temp ? val_max : temp;
    //     }
    //     warp.sync();
    //     const auto max = cg::reduce(warp, val_max, cg::greater<half>());
    //     half2 max2{max, max};
    //     // compute the sum of exp-ed and shifted array
    //     auto row_ptr2 = reinterpret_cast<half2 *>(row_ptr);
    //     half2 val_sum2{half_zero, half_zero};
    //     for (auto i = warp.thread_rank(); i < seq_len2; i += 32) {
    //         const auto temp = h2exp2(__hsub2(row_ptr2[i], max2));
    //         val_sum2 += temp;
    //         row_ptr2[i] = temp;
    //     }
    //     warp.sync();
    //     const auto sum2 = cg::reduce(warp, val_sum2, cg::plus<half2>());
    //     // update with softmax scaling
    //     const auto sum = sum2.x + sum2.y;
    //     half2 one_over_sum =
    //         __h2div(half2{half_one, half_one}, half2{sum, sum});
    //     for (auto i = warp.thread_rank(); i < seq_len2; i += 32) {
    //         row_ptr2[i] = row_ptr2[i] * one_over_sum;
    //     }
    // }
    // cta.sync();
    const auto vhead_dim = vdim / num_head;
    for (int VC = warp_id; VC < vhead_dim / 16; VC += num_warp) {
        frag_t::b_t<wmma::row_major> b_frag;
        wmma::fill_fragment(c_frag, half_zero);
        for (int i = 0; i < seq_len; i += 16) {
            auto V_ptr = &V[vdim * i + blockIdx.y * vhead_dim + VC * 16];
            wmma::load_matrix_sync(a_frag, temp_row + i, ldRow);
            wmma::load_matrix_sync(b_frag, V_ptr, vdim);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        auto res =
            &Z[blockIdx.x * 16 * vdim + blockIdx.y * vhead_dim + VC * 16];
        wmma::store_matrix_sync(res, c_frag, vdim, wmma::mem_row_major);
    }
}

template <typename T, typename TyGroup>
__device__ T block_max(const TyGroup &cta, const T *data, const int size,
                       T *temp) {
    T max = (T)-1e20f;
    for (int i = cta.thread_rank(); i < size; i += cta.size()) {
        const auto t = data[i];
        max = (t > max) ? t : max;
    }
    auto warp = cg::tiled_partition<32>(cta);
    const auto warp_id = warp.meta_group_rank();
    const auto num_warp = warp.meta_group_size();
    const auto lane_id = warp.thread_rank();
    max = cg::reduce(warp, max, cg::greater<T>());
    if (lane_id == 0)
        temp[warp_id] = max;
    max = (T)-1e20f;
    cta.sync();
    if (lane_id < num_warp) {
        max = temp[lane_id];
    }
    warp.sync();
    return cg::reduce(warp, max, cg::greater<T>());
}

template <typename T, typename TyGroup>
__device__ T block_sum(const TyGroup &cta, const T *data, const int size,
                       T *temp) {
    T sum = (T)0;
    for (int i = cta.thread_rank(); i < size; i += cta.size()) {
        sum += data[i];
    }
    auto warp = cg::tiled_partition<32>(cta);
    const auto warp_id = warp.meta_group_rank();
    const auto num_warp = warp.meta_group_size();
    const auto lane_id = warp.thread_rank();
    sum = cg::reduce(warp, sum, cg::plus<T>());
    if (lane_id == 0)
        temp[warp_id] = sum;
    sum = (T)0;
    cta.sync();
    if (lane_id < num_warp) {
        sum = temp[lane_id];
    }
    warp.sync();
    return cg::reduce(warp, sum, cg::plus<T>());
}

template <int num_thd>
__global__ void __launch_bounds__(num_thd)
    __kernel_multi_head_full_skew_blkSFM(const half *__restrict__ Q,
                                         const half *__restrict__ K,
                                         const half *__restrict__ V,
                                         half *__restrict__ Z, const int kdim,
                                         const int vdim, const int seq_len,
                                         const int num_head,
                                         const half *__restrict__ mask) {
    // blockIdx.x: block row id
    // blockIdx.y: head_id
    using frag_t = culib::mma::mma_t<16, 16, 16>;
    extern __shared__ half smem[];
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    const auto warp_id = warp.meta_group_rank();
    const auto lane_id = warp.thread_rank();
    constexpr int num_warp = num_thd / 32;
    const auto head_dim = kdim / num_head;
    auto Q_ptr = &Q[16 * blockIdx.x * kdim + head_dim * blockIdx.y];
    auto temp_Q = smem;
    const auto ldQ = head_dim + FP16_skew;
    const auto ldRow = seq_len + FP16_skew;
    for (int r = warp_id; r < 16; r += num_warp) {
        auto dst = &temp_Q[r * ldQ];
        auto src = &Q_ptr[r * kdim];
        cg::memcpy_async(warp, dst, src, sizeof(half) * head_dim);
    }
    cta.sync();
    frag_t::a_t<wmma::row_major> a_frag;
    frag_t::b_t<wmma::col_major> b_frag;
    frag_t::c_t<half> c_frag;
    auto temp_row = &smem[16 * ldQ];
    for (int KR = warp_id; KR < seq_len / 16; KR += num_warp) {
        auto K_ptr = &K[16 * KR * kdim + head_dim * blockIdx.y];
        wmma::fill_fragment(c_frag, half_zero);
        for (int i = 0; i < head_dim; i += 16) {
            wmma::load_matrix_sync(a_frag, smem + i, ldQ);
            wmma::load_matrix_sync(b_frag, K_ptr + i, kdim);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(temp_row + KR * 16, c_frag, ldRow,
                                wmma::mem_row_major);
    }
    cta.sync();
    // mask
    const auto mask_base = &mask[(blockIdx.x * 16) * seq_len];
    for (int i = threadIdx.x; i < seq_len * 16; i += blockDim.x) {
        const auto r = i / seq_len;
        const auto c = i % seq_len;
        temp_row[r * ldRow + c] += mask_base[i];
    }
    cta.sync();
    for (int row = 0; row < 16; row++) {
        auto row_ptr = temp_row + row * ldRow;
        // find the max
        const auto val_max = block_max<half>(cta, row_ptr, seq_len, smem);
        // exp and shift
        for (int i = cta.thread_rank(); i < seq_len; i += num_thd) {
            half temp = row_ptr[i] - val_max;
            row_ptr[i] = hexp(temp);
        }
        cta.sync();
        // get sum
        const auto val_sum = block_sum<half>(cta, row_ptr, seq_len, smem);
        const auto one_val_sum = half_one / val_sum;
        // update with softmax scaling
        for (int i = cta.thread_rank(); i < seq_len; i += num_thd) {
            row_ptr[i] = row_ptr[i] * one_val_sum;
        }
        cta.sync();
    }
    const auto vhead_dim = vdim / num_head;
    for (int VC = warp_id; VC < vhead_dim / 16; VC += num_warp) {
        frag_t::b_t<wmma::row_major> b_frag;
        wmma::fill_fragment(c_frag, half_zero);
        for (int i = 0; i < seq_len; i += 16) {
            auto V_ptr = &V[vdim * i + blockIdx.y * vhead_dim + VC * 16];
            wmma::load_matrix_sync(a_frag, temp_row + i, ldRow);
            wmma::load_matrix_sync(b_frag, V_ptr, vdim);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        auto res =
            &Z[blockIdx.x * 16 * vdim + blockIdx.y * vhead_dim + VC * 16];
        wmma::store_matrix_sync(res, c_frag, vdim, wmma::mem_row_major);
    }
}
