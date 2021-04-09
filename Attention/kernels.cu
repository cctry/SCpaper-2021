#include "Attention.h"
#include "kernels.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.hpp>
#include <reduce.cuh>
namespace cg = cooperative_groups;
__global__ void __kernel_batch_softmax(half *__restrict__ data,
                                       const int num_batch, const int size,
                                       const int num) {
    extern __shared__ half smem[];
    auto smem_1 = &smem[0];
    auto smem_2 = &smem[size];
    const auto batch_id = blockIdx.x;
    const auto arr_id = blockIdx.y;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        auto temp = data[i * num + arr_id + batch_id * num * size];
        smem_1[i] = temp;
        smem_2[i] = temp;
    }
    __syncthreads();
    const auto max = block_max(smem_1, size);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        auto temp = hexp(smem_2[i] - max);
        smem_1[i] = temp;
        smem_2[i] = temp;
    }
    __syncthreads();
    const auto sum = block_sum(smem_2, size);
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        data[i * num + arr_id + batch_id * num * size] = smem_1[i] / sum;
}

__global__ void
__kernel_multi_head(const half *__restrict__ Q, const half *__restrict__ K,
                    const half *__restrict__ V, half *__restrict__ Z,
                    const int kdim, const int vdim, const int seq_len,
                    const int num_head, const half *__restrict__ mask) {
    // 256 * seq_len / 16 + seq_len * num_warp
    // temp of blocks + temp of rows
    extern __shared__ half smem[];
    const auto warp_id = threadIdx.x / 32;
    const auto lane_id = threadIdx.x % 32;
    half *const temp_blk = &smem[0];
    half *const temp_row = &smem[16 * seq_len + warp_id * seq_len];
    // const auto z_col = blockIdx.x;
    // const auto z_row = blockIdx.y;
    // const auto head_id = z_col / ((vdim / 16) / num_head);
    const auto scale = hsqrt(half_one / (kdim / num_head));
    // Compute z_row in Q * K^T
    using frag_t = culib::mma::mma_t<16, 16, 16>;
    frag_t::a_t<wmma::row_major> a;
    frag_t::b_t<wmma::col_major> b;
    frag_t::c_t<half> c;
    // Each warp compute an element
    // It takes the blk_row 'th block row from K^T
    for (int blk_row = warp_id; blk_row < seq_len / 16;
         blk_row += blockDim.x / 32) {
        for (int i = 0; i < (kdim / num_head) / 16; i++) {
            const auto head_blk_offset =
                (blockIdx.x / ((vdim / 16) / num_head)) *
                ((kdim / num_head) / 16);
            // scale Q
            const auto q =
                get_blk_start(Q, blockIdx.y, i + head_blk_offset, kdim);
            wmma::load_matrix_sync(c, q, kdim, wmma::mem_row_major);
#pragma unroll
            for (int t = 0; t < c.num_elements; t++)
                c.x[t] *= scale;
            wmma::store_matrix_sync(&temp_blk[blk_row * 256], c, 16,
                                    wmma::mem_row_major);
            wmma::fill_fragment(c, half_zero);
            wmma::load_matrix_sync(a, &temp_blk[blk_row * 256], 16);
            // load K
            const auto k = get_blk_start(K, blk_row, i + head_blk_offset, kdim);
            wmma::load_matrix_sync(b, k, kdim);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(&temp_blk[blk_row * 256], c, 16,
                                wmma::mem_row_major);
    }
    __syncthreads();
    for (int row = warp_id; row < 16; row += blockDim.x / 32) {
        warp_load_row(temp_row, temp_blk, row, seq_len, 16);
        // mask
        const auto mask_row_base = &mask[(blockIdx.y * 16 + row) * seq_len];
        for (int i = lane_id; i < seq_len; i += 32)
            temp_row[i] += mask_row_base[i];
        // softmax
        half max = warp_max(temp_row, seq_len);
        for (int i = lane_id; i < seq_len; i += 32)
            temp_row[i] = hexp(temp_row[i] - max);
        half sum = warp_sum(temp_row, seq_len);
        for (int i = lane_id; i < seq_len; i += 32)
            temp_row[i] = temp_row[i] / sum;
        warp_store_row(temp_blk, temp_row, row, seq_len, 16);
    }
    __syncthreads();
    if (warp_id == 0) {
        frag_t::b_t<wmma::row_major> b;
        wmma::fill_fragment(c, half_zero);
        for (int i = 0; i < seq_len / 16; i++) {
            wmma::load_matrix_sync(a, &temp_blk[i * 256], 16);
            const auto v = get_blk_start(V, i, blockIdx.x, vdim);
            wmma::load_matrix_sync(b, v, vdim);
            wmma::mma_sync(c, a, b, c);
        }
        const auto dst = get_blk_start(Z, blockIdx.y, blockIdx.x, vdim);
        wmma::store_matrix_sync(dst, c, vdim, wmma::mem_row_major);
    }
}

// __global__ void
// __kernel_multi_head_full(const half *__restrict__ Q, const half *__restrict__
// K,
//                          const half *__restrict__ V, half *__restrict__ Z,
//                          const int kdim, const int vdim, const int seq_len,
//                          const int num_head, const half *__restrict__ mask) {
//     // blockIdx.x: block row id
//     // blockIdx.y: head_id
//     using frag_t = culib::mma::mma_t<16, 16, 16>;
//     extern __shared__ half smem[];
//     auto cta = cg::this_thread_block();
//     auto warp = cg::tiled_partition<32>(cta);
//     const auto warp_id = warp.meta_group_rank();
//     const auto lane_id = warp.thread_rank();
//     const auto num_warp = warp.meta_group_size();
//     const auto head_dim = kdim / num_head;
//     auto Q_ptr = &Q[16 * blockIdx.x * kdim + head_dim * blockIdx.y];
//     auto temp_Q = smem;
//     const half scale = hsqrt(__int2half_rn(head_dim));
//     for (int r = warp_id; r < 16; r += num_warp) {
//         auto dst = &temp_Q[r * head_dim];
//         auto src = &Q_ptr[r * kdim];
//         cg::memcpy_async(warp, dst, src, sizeof(half) * head_dim);
//         // auto dst2 = reinterpret_cast<half2 *>(dst);
//         // for (int i = lane_id; i < head_dim / 2; i += 32) {
//         //     dst2[i] = __h2div(dst2[i], half2{scale, scale});
//         // }
//     }
//     cta.sync();
//     frag_t::a_t<wmma::row_major> a_frag;
//     frag_t::b_t<wmma::col_major> b_frag;
//     frag_t::c_t<half> c_frag;
//     auto temp_row = &smem[16 * head_dim];
//     for (int KR = warp_id; KR < seq_len / 16; KR += num_warp) {
//         auto K_ptr = &K[16 * KR * kdim + head_dim * blockIdx.y];
//         wmma::fill_fragment(c_frag, half_zero);
//         for (int i = 0; i < head_dim; i += 16) {
//             wmma::load_matrix_sync(a_frag, smem + i, head_dim);
//             wmma::load_matrix_sync(b_frag, K_ptr + i, kdim);
//             wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
//         }
//         wmma::store_matrix_sync(temp_row + KR * 16, c_frag, seq_len,
//                                 wmma::mem_row_major);
//     }
//     cta.sync();
//     // mask
//     const auto mask_base =
//         reinterpret_cast<const half2 *>(&mask[(blockIdx.x * 16) * seq_len]);
//     auto temp_row_2 = reinterpret_cast<half2 *>(temp_row);
//     for (int i = threadIdx.x; i < seq_len * 8; i += blockDim.x) {
//         temp_row_2[i] += mask_base[i];
//     }
//     cta.sync();
//     for (int row = warp_id; row < 16; row += num_warp) {
//         auto row_ptr = temp_row + row * seq_len;
//         // find the max
//         half val_max = half_zero, temp;
//         for (auto i = warp.thread_rank(); i < seq_len; i += warp.size()) {
//             temp = row_ptr[i];
//             val_max = val_max > temp ? val_max : temp;
//         }
//         warp.sync();
//         const auto max = cg::reduce(warp, val_max, cg::greater<half>());
//         // compute the sum of exp-ed and shifted array
//         half val_sum = half_zero;
//         for (auto i = warp.thread_rank(); i < seq_len; i += warp.size()) {
//             temp = hexp(row_ptr[i] - max);
//             val_sum += temp;
//             row_ptr[i] = temp;
//         }
//         warp.sync();
//         const auto sum = cg::reduce(warp, val_sum, cg::plus<half>());
//         // update with softmax scaling
//         for (auto i = warp.thread_rank(); i < seq_len; i += warp.size()) {
//             row_ptr[i] = row_ptr[i] / sum;
//         }
//     }
//     cta.sync();
//     const auto vhead_dim = vdim / num_head;
//     for (int VC = warp_id; VC < vhead_dim / 16; VC += num_warp) {
//         frag_t::b_t<wmma::row_major> b_frag;
//         wmma::fill_fragment(c_frag, half_zero);
//         for (int i = 0; i < seq_len; i += 16) {
//             auto V_ptr = &V[vdim * i + blockIdx.y * vhead_dim + VC * 16];
//             wmma::load_matrix_sync(a_frag, temp_row + i, seq_len);
//             wmma::load_matrix_sync(b_frag, V_ptr, vdim);
//             wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
//         }
//         auto res =
//             &Z[blockIdx.x * 16 * vdim + blockIdx.y * vhead_dim + VC * 16];
//         wmma::store_matrix_sync(res, c_frag, vdim, wmma::mem_row_major);
//     }
// }

__global__ void
__kernel_multi_head_tile(const half *__restrict__ Q, const half *__restrict__ K,
                         const half *__restrict__ V, half *__restrict__ Z,
                         const int kdim, const int vdim, const int seq_len,
                         const int num_head, const half *__restrict__ mask,
                         const int tile_size) {
    // blockIdx.x: block row id
    // blockIdx.y: head_id
    // blockIdx.z: tile id
    // tile_size: number of
    using frag_t = culib::mma::mma_t<16, 16, 16>;
    extern __shared__ half smem[];
    const auto warp_id = threadIdx.x / 32;
    const auto lane_id = threadIdx.x % 32;
    const auto num_warp = blockDim.x / 32;
    const auto head_dim = kdim / num_head;
    frag_t::c_t<half> load_frag;
    const auto scale = hsqrt(half_one / (kdim / num_head));
    auto Q_ptr = &Q[16 * blockIdx.x * kdim + head_dim * blockIdx.y];
    for (int i = warp_id * 16; i < head_dim; i += 16 * num_warp) {
        wmma::load_matrix_sync(load_frag, Q_ptr + i, kdim, wmma::mem_row_major);
#pragma unroll
        for (int t = 0; t < load_frag.num_elements; t++)
            load_frag.x[t] *= scale;
        wmma::store_matrix_sync(smem + i, load_frag, head_dim,
                                wmma::mem_row_major);
    }
    __syncthreads();
    frag_t::a_t<wmma::row_major> a_frag;
    frag_t::b_t<wmma::col_major> b_frag;
    frag_t::c_t<half> c_frag;
    auto temp_row = &smem[16 * head_dim];
    for (int KR = warp_id; KR < seq_len / 16; KR += num_warp) {
        auto K_ptr = &K[16 * KR * kdim + head_dim * blockIdx.y];
        wmma::fill_fragment(c_frag, half_zero);
        for (int i = 0; i < head_dim; i += 16) {
            wmma::load_matrix_sync(a_frag, smem + i, head_dim);
            wmma::load_matrix_sync(b_frag, K_ptr + i, kdim);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(temp_row + KR * 16, c_frag, seq_len,
                                wmma::mem_row_major);
    }
    __syncthreads();
    // mask
    const auto mask_base =
        reinterpret_cast<const half2 *>(&mask[(blockIdx.x * 16) * seq_len]);
    auto temp_row_2 = reinterpret_cast<half2 *>(temp_row);
    for (int i = threadIdx.x; i < seq_len * 8; i += blockDim.x) {
        temp_row_2[i] += mask_base[i];
    }
    __syncthreads();
    for (int row = warp_id; row < 16; row += num_warp) {
        auto row_ptr = temp_row + row * seq_len;
        // softmax
        const half max = warp_max(row_ptr, seq_len);
        for (int i = lane_id; i < seq_len; i += 32)
            row_ptr[i] = hexp(row_ptr[i] - max);
        const half sum = warp_sum(row_ptr, seq_len);
        for (int i = lane_id; i < seq_len; i += 32)
            row_ptr[i] = row_ptr[i] / sum;
    }
    __syncthreads();
    const auto vhead_dim = vdim / num_head;
    for (int VC = warp_id; VC < tile_size; VC += num_warp) {
        frag_t::b_t<wmma::row_major> b_frag;
        wmma::fill_fragment(c_frag, half_zero);
        for (int i = 0; i < seq_len; i += 16) {
            auto V_ptr = &V[vdim * i + blockIdx.y * vhead_dim +
                            (blockIdx.z * tile_size + VC) * 16];
            wmma::load_matrix_sync(a_frag, temp_row + i, seq_len);
            wmma::load_matrix_sync(b_frag, V_ptr, vdim);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        auto res = &Z[blockIdx.x * 16 * vdim + blockIdx.y * vhead_dim +
                      (blockIdx.z * tile_size + VC) * 16];
        wmma::store_matrix_sync(res, c_frag, vdim, wmma::mem_row_major);
    }
}

__global__ void __kernel_multi_head_sharedQK(const half *__restrict__ QK,
                                             const half *__restrict__ V,
                                             half *__restrict__ Z,
                                             const int vdim, const int seq_len,
                                             const int num_head,
                                             const half *__restrict__ mask) {
    using frag_t = culib::mma::mma_t<16, 16, 16>;
    extern __shared__ half smem[]; // 16 * seq_len
    auto cta = cg::this_thread_block();
    const auto head_id = cta.group_index().y;
    const auto z_row = cta.group_index().x;
    auto warp = cg::tiled_partition<32>(cta);
    for (auto r = warp.meta_group_rank(); r < 16; r += warp.meta_group_size()) {
        const auto row = 16 * z_row + r;
        auto dst = &smem[r * seq_len];
        auto src = &QK[head_id * seq_len * seq_len + seq_len * row];
        cg::memcpy_async(warp, dst, src, sizeof(half) * seq_len);
        cg::wait(warp);
        // mask and find the max
        half val_max = half_zero, temp;
        for (auto i = warp.thread_rank(); i < seq_len; i += warp.size()) {
            temp = dst[i] + mask[seq_len * row + i];
            dst[i] = temp;
            val_max = val_max > temp ? val_max : temp;
        }
        warp.sync();
        const auto max = cg::reduce(warp, val_max, cg::greater<half>());
        // compute the sum of exp-ed and shifted array
        half val_sum = half_zero;
        for (auto i = warp.thread_rank(); i < seq_len; i += warp.size()) {
            temp = hexp(dst[i] - max);
            val_sum += temp;
            dst[i] = temp;
        }
        warp.sync();
        const auto sum = cg::reduce(warp, val_sum, cg::plus<half>());
        // update with softmax scaling
        for (auto i = warp.thread_rank(); i < seq_len; i += warp.size()) {
            dst[i] = dst[i] / sum;
        }
    }
    cta.sync();
    const auto vhead_dim = vdim / num_head;
    for (auto VC = warp.meta_group_rank(); VC < vhead_dim / 16;
         VC += warp.meta_group_size()) {
        frag_t::a_t<wmma::row_major> a_frag;
        frag_t::b_t<wmma::row_major> b_frag;
        frag_t::c_t<half> c_frag;
        wmma::fill_fragment(c_frag, half_zero);
        for (int i = 0; i < seq_len; i += 16) {
            auto V_ptr = &V[vdim * i + head_id * vhead_dim + VC * 16];
            wmma::load_matrix_sync(a_frag, smem + i, seq_len);
            wmma::load_matrix_sync(b_frag, V_ptr, vdim);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        auto res = &Z[blockIdx.x * 16 * vdim + head_id * vhead_dim + VC * 16];
        wmma::store_matrix_sync(res, c_frag, vdim, wmma::mem_row_major);
    }
}

template <typename _Tg>
__device__ void clear_smem(const _Tg &group, void *smem, const int N) {
    auto ptr = reinterpret_cast<int *>(smem);
#pragma unroll
    for (int i = group.thread_rank(); i < N; i += group.size()) {
        ptr[i] = 0;
    }
    group.sync();
}

/*
 * generate a dense matrix
 */
__global__ void __kernel_multi_head_prune(
    const half *__restrict__ Q, const half *__restrict__ K,
    const half *__restrict__ V, half *__restrict__ Z, const int kdim,
    const int vdim, const int seq_len, const int num_head,
    const int *__restrict__ v_col_id, const int *__restrict__ v_head_ptr,
    const int nnz_col_v, const half *__restrict__ mask) {
    constexpr int FP16_skew = 16;
    // blockIdx.x: block row id
    // blockIdx.y: head_id
    /*
     * smem partition:
     * First part: 16 * seq_len * sizeof(half)
     * temp_row (16 * seq_len * sizeof(half)): one block row of QK
     * Second part: max(temp_Q, temp_blk+temp_id)
     *  temp_Q (16 * head_dim * sizeof(half)): one block row of Q
     *  temp_blk (256 * num_warp * sizeof(half)): one compressed block of Z
     *  temp_id (16 * num_warp * sizeof(int))
     */
    using frag_t = culib::mma::mma_t<16, 16, 16>;
    extern __shared__ half smem[];
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    const auto warp_id = warp.meta_group_rank();
    const auto lane_id = warp.thread_rank();
    const auto num_warp = warp.meta_group_size();
    const auto head_dim = kdim / num_head;
    auto Q_ptr = &Q[16 * blockIdx.x * kdim + head_dim * blockIdx.y];
    half *temp_row = smem;
    const auto ldQ = head_dim + FP16_skew;
    const auto ldRow = seq_len + FP16_skew;
    auto temp_Q = &smem[16 * ldRow];
    for (int r = warp_id; r < 16; r += num_warp) {
        auto dst = &temp_Q[r * ldQ];
        auto src = &Q_ptr[r * kdim];
        cg::memcpy_async(warp, dst, src, sizeof(half) * head_dim);
    }
    cta.sync();
    frag_t::a_t<wmma::row_major> a_frag;
    frag_t::b_t<wmma::col_major> b_frag;
    frag_t::c_t<half> c_frag;
    for (int KR = warp_id * 16; KR < seq_len; KR += num_warp * 16) {
        auto K_ptr = &K[KR * kdim + head_dim * blockIdx.y];
        wmma::fill_fragment(c_frag, half_zero);
        for (int i = 0; i < head_dim; i += 16) {
            wmma::load_matrix_sync(a_frag, temp_Q + i, ldQ);
            wmma::load_matrix_sync(b_frag, K_ptr + i, kdim);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(temp_row + KR, c_frag, ldRow,
                                wmma::mem_row_major);
    }
    cta.sync();
    // mask
    const auto mask_base =
        reinterpret_cast<const half2 *>(&mask[(blockIdx.x * 16) * seq_len]);
    auto temp_row_2 = reinterpret_cast<half2 *>(temp_row);
    for (int i = threadIdx.x; i < seq_len * 8; i += blockDim.x) {
        temp_row_2[i] += mask_base[i];
    }
    cta.sync();
    softmax_blk(cta, temp_row, seq_len, 16, ldRow);
    // compute Z
    const auto ld_tempblk = 16 * num_warp + FP16_skew;
    auto temp_blk = temp_Q + 16 * warp_id;
    auto temp_id =
        reinterpret_cast<int *>(&temp_Q[16 * ld_tempblk]) + (16 * warp_id);
    const auto VC_start = v_head_ptr[blockIdx.y];
    const auto VC_end = v_head_ptr[blockIdx.y + 1];
    for (int VC_blk = VC_start + warp_id * 16; VC_blk < VC_end;
         VC_blk += num_warp * 16) {
        frag_t::b_t<wmma::row_major> b_frag;
        wmma::fill_fragment(c_frag, half_zero);
        const int num_valid_col = (VC_blk + 16) < VC_end ? 16 : VC_end - VC_blk;
        cg::memcpy_async(warp, temp_id, &v_col_id[VC_blk],
                         sizeof(int) * num_valid_col);
        auto two_thd = cg::tiled_partition<2>(warp);
        auto dst = &temp_blk[ld_tempblk * two_thd.meta_group_rank()];
        clear_smem(two_thd, dst, 16 * sizeof(half));
        for (int i = 0; i < seq_len; i += 16) {
            auto src = &V[nnz_col_v * (i + two_thd.meta_group_rank()) + VC_blk];
            for (int k = two_thd.thread_rank(); k < num_valid_col; k += 2)
                dst[k] = src[k];
            warp.sync();
            wmma::load_matrix_sync(b_frag, temp_blk, ld_tempblk);
            wmma::load_matrix_sync(a_frag, temp_row + i, ldRow);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(temp_blk, c_frag, ld_tempblk,
                                wmma::mem_row_major);
        cg::wait(warp);
        auto src = &temp_blk[16 * two_thd.meta_group_rank()];
        for (int k = two_thd.thread_rank(); k < num_valid_col; k += 2) {
            const auto col = temp_id[k];
            auto dst =
                &Z[(blockIdx.x * 16 + two_thd.meta_group_rank()) * vdim + col];
        }
    }
}
