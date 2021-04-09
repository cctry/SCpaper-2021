#include "kernel_fuseO.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <mma.hpp>
#include <utils.h>

namespace cg = cooperative_groups;
/*
 * It returns the pointer of the top-left corner of give block in a matrix.
 * Assume the matrix is stored in a row-major array.
 * It needs the number of columns of the matrix (leading dimension).
 */
template <typename T, int SIZE = 16>
__device__ T *get_blk_start(T *data, const int row_blk, const int col_blk,
                            const int stride) {
    auto res = &data[row_blk * SIZE * stride + SIZE * col_blk];
    return res;
}

template <typename _Tg>
__device__ void clear_smem(const _Tg &group, void *smem, const int N) {
    auto ptr = reinterpret_cast<int *>(smem);
#pragma unroll
    for (int i = group.thread_rank(); i < N / sizeof(int); i += group.size()) {
        ptr[i] = 0;
    }
    group.sync();
}

// A*B = C
__device__ void __kernel_blk_mmul_blk_bias_smem_blk(
    half *__restrict__ smem, const int *__restrict__ A_row_ptr,
    const int *__restrict__ A_row_offset, const half *__restrict__ A_data,
    const int A_blk_row_num, const half *__restrict__ B, half *__restrict__ C,
    const half *__restrict__ bias, const int out_row_blk_num) {
    // num_warp * 256
    // temp_blk
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    const auto warp_id = cta.thread_rank() >> 5;
    const auto num_warp = cta.size() / 32;
    auto tile_temp = &smem[warp.meta_group_rank() * 256];
    for (int id = blockIdx.x; id < out_row_blk_num * A_blk_row_num;
         id += gridDim.x) {
        const auto out_row = id / A_blk_row_num;
        const auto out_col = id % A_blk_row_num;
        clear_smem(cta, smem, num_warp * 256 * sizeof(half));
        // const auto out_col_blk_num = A_blk_row_num;
        // out_row and out_col are untransposed positions
        using frag_t = culib::mma::mma_t<16, 16, 16>;
        frag_t::a_t<wmma::row_major> a;
        frag_t::b_t<wmma::col_major> b;
        frag_t::c_t<half> c;
        wmma::fill_fragment(c, 0);
        const auto ldm = A_blk_row_num * 16;
        for (auto i = A_row_ptr[out_col] + warp_id; i < A_row_ptr[out_col + 1];
             i += num_warp) {
            wmma::load_matrix_sync(a, &A_data[i * 256], 16);
            auto src = &B[out_row * 16 * ldm + A_row_offset[i] * 16];
            wmma::load_matrix_sync(b, src, ldm);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(tile_temp, c, 16, wmma::mem_col_major);
        cta.sync();
        // reduce across warp
        for (int e = cta.thread_rank(); e < 256; e += cta.size()) {
            auto base = &smem[e];
            auto sum = *base;
            for (int i = 0; i < num_warp; i++) {
                sum += base[i * 256];
            }
            *base = sum;
        }
        cta.sync();
        // add bias
        if (warp_id == 0) {
            const auto bias_value =
                bias[out_col * 16 + warp.thread_rank() % 16];
#pragma unroll 8
            for (int i = warp.thread_rank(); i < 256; i += warp.size()) {
                tile_temp[i] += bias_value;
            }
            const auto dst =
                get_blk_start(C, out_row, out_col, A_blk_row_num * 16);
            wmma::load_matrix_sync(c, tile_temp, 16, wmma::mem_col_major);
            wmma::store_matrix_sync(dst, c, A_blk_row_num * 16,
                                    wmma::mem_col_major);
        }
    }
}

// A*B = C
__device__ void __kernel_blk_mmul_blk_bias_smem(
    half *__restrict__ smem, const int *__restrict__ A_row_ptr,
    const int *__restrict__ A_row_offset, const half *__restrict__ A_data,
    const int A_blk_row_num, const half *__restrict__ B, half *__restrict__ C,
    const half *__restrict__ bias, const int out_row_blk_num) {
    // num_warp * 256
    // temp_blk
    auto grid = cg::this_grid();
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    auto tile_temp = &smem[warp.meta_group_rank() * 256];
    const auto gwarp_id = grid.thread_rank() >> 5;
    const auto total_warp = grid.size() / 32;
    const auto total_tile = A_blk_row_num * out_row_blk_num;
    for (int t = gwarp_id; t < total_tile; t += total_warp) {
        // const auto out_col_blk_num = A_blk_row_num;
        // out_row and out_col are untransposed positions
        const auto out_row = gwarp_id / A_blk_row_num;
        const auto out_col = gwarp_id % A_blk_row_num;
        using frag_t = culib::mma::mma_t<16, 16, 16>;
        frag_t::a_t<wmma::row_major> a;
        frag_t::b_t<wmma::col_major> b;
        frag_t::c_t<half> c;
        wmma::fill_fragment(c, 0);
        for (auto i = A_row_ptr[out_col]; i < A_row_ptr[out_col + 1]; i++) {
            wmma::load_matrix_sync(a, &A_data[i * 256], 16);
            const half *src =
                get_blk_start(B, out_row, A_row_offset[i], A_blk_row_num * 16);
            wmma::load_matrix_sync(b, src, A_blk_row_num * 16);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(tile_temp, c, 16, wmma::mem_col_major);
        const auto bias_value = bias[out_col * 16 + warp.thread_rank() % 16];
#pragma unroll 8
        for (int i = warp.thread_rank(); i < 256; i += warp.size()) {
            tile_temp[i] += bias_value;
        }
        const auto dst = get_blk_start(C, out_row, out_col, A_blk_row_num * 16);
        wmma::load_matrix_sync(c, tile_temp, 16, wmma::mem_col_major);
        wmma::store_matrix_sync(dst, c, A_blk_row_num * 16,
                                wmma::mem_col_major);
    }
}

__global__ void __kernel_multi_head_full_fuseO(
    const half *__restrict__ Q, const half *__restrict__ K,
    const half *__restrict__ V, half *__restrict__ Z, const int kdim,
    const int vdim, const int seq_len, const int num_head,
    const half *__restrict__ mask, const int *__restrict__ O_row_ptr,
    const int *__restrict__ O_row_offset, const half *__restrict__ O_data,
    const int O_blk_row_num, half *__restrict__ output,
    const half *__restrict__ Obias) {
    // blockIdx.x: block row id
    // blockIdx.y: head_id
    using frag_t = culib::mma::mma_t<16, 16, 16>;
    extern __shared__ half smem[];
    auto grid = cg::this_grid();
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    const auto warp_id = warp.meta_group_rank();
    const auto lane_id = warp.thread_rank();
    const auto num_warp = warp.meta_group_size();
    // attention
    for (int id = blockIdx.x; id < num_head * seq_len / 16; id += gridDim.x) {
        const int head_id = id / (seq_len / 16);
        const int blk_row_id = id % (seq_len / 16);
        const auto head_dim = kdim / num_head;
        auto Q_ptr = &Q[16 * blk_row_id * kdim + head_dim * head_id];
        auto temp_Q = smem;
        // const half scale = hsqrt(__int2half_rn(head_dim));
        for (int r = warp_id; r < 16; r += num_warp) {
            auto dst = &temp_Q[r * head_dim];
            auto src = &Q_ptr[r * kdim];
            cg::memcpy_async(warp, dst, src, sizeof(half) * head_dim);
        }
        cta.sync();
        frag_t::a_t<wmma::row_major> a_frag;
        frag_t::b_t<wmma::col_major> b_frag;
        frag_t::c_t<half> c_frag;
        auto temp_row = &smem[16 * head_dim];
        for (int KR = warp_id; KR < seq_len / 16; KR += num_warp) {
            auto K_ptr = &K[16 * KR * kdim + head_dim * head_id];
            wmma::fill_fragment(c_frag, half_zero);
            for (int i = 0; i < head_dim; i += 16) {
                wmma::load_matrix_sync(a_frag, smem + i, head_dim);
                wmma::load_matrix_sync(b_frag, K_ptr + i, kdim);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(temp_row + KR * 16, c_frag, seq_len,
                                    wmma::mem_row_major);
        }
        cta.sync();
        // mask
        const auto mask_base =
            reinterpret_cast<const half2 *>(&mask[(blk_row_id * 16) * seq_len]);
        auto temp_row_2 = reinterpret_cast<half2 *>(temp_row);
        for (int i = threadIdx.x; i < seq_len * 8; i += blockDim.x) {
            temp_row_2[i] += mask_base[i];
        }
        cta.sync();
        for (int row = warp_id; row < 16; row += num_warp) {
            auto row_ptr = temp_row + row * seq_len;
            // find the max
            half val_max = half_zero, temp;
            for (auto i = warp.thread_rank(); i < seq_len; i += warp.size()) {
                temp = row_ptr[i];
                val_max = val_max > temp ? val_max : temp;
            }
            warp.sync();
            const auto max = cg::reduce(warp, val_max, cg::greater<half>());
            // compute the sum of exp-ed and shifted array
            half val_sum = half_zero;
            for (auto i = warp.thread_rank(); i < seq_len; i += warp.size()) {
                temp = hexp(row_ptr[i] - max);
                val_sum += temp;
                row_ptr[i] = temp;
            }
            warp.sync();
            const auto sum = cg::reduce(warp, val_sum, cg::plus<half>());
            // update with softmax scaling
            for (auto i = warp.thread_rank(); i < seq_len; i += warp.size()) {
                row_ptr[i] = row_ptr[i] / sum;
            }
        }
        cta.sync();
        const auto vhead_dim = vdim / num_head;
        for (int VC = warp_id; VC < vhead_dim / 16; VC += num_warp) {
            frag_t::b_t<wmma::row_major> b_frag;
            wmma::fill_fragment(c_frag, half_zero);
            for (int i = 0; i < seq_len; i += 16) {
                auto V_ptr = &V[vdim * i + head_id * vhead_dim + VC * 16];
                wmma::load_matrix_sync(a_frag, temp_row + i, seq_len);
                wmma::load_matrix_sync(b_frag, V_ptr, vdim);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            auto res =
                &Z[blk_row_id * 16 * vdim + head_id * vhead_dim + VC * 16];
            wmma::store_matrix_sync(res, c_frag, vdim, wmma::mem_row_major);
        }
    }
    grid.sync();
    // SPMM
    // __kernel_blk_mmul_blk_bias_smem_blk(smem, O_row_ptr, O_row_offset,
    // O_data,
    //                                     O_blk_row_num, Z, output, Obias,
    //                                     seq_len / 16);
    __kernel_blk_mmul_blk_bias_smem(smem, O_row_ptr, O_row_offset, O_data,
                                    O_blk_row_num, Z, output, Obias,
                                    seq_len / 16);
}

struct {
    cudaDeviceProp prop;
    AutoExec ae{[this] {
        int device = 0;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
    }};
} cuda_helper;

void Multihead_atttion_fuseO(half *Q, half *K, half *V, half *mask, half *Z,
                             int *O_row_ptr, int *O_row_offset, half *O_data,
                             half *O_bias, std::shared_ptr<Model_t> model,
                             half *out) {
    auto smem_size = [&](int n) {
        const auto attn = 16 * (model->kdim / model->nhead + model->seq_len);
        const auto spmm = (n / 32) * 256;
        return sizeof(half) * std::max(attn, spmm);
    };
    int num_thd, _num_blk, numBlocksPerSm;
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &_num_blk, &num_thd, __kernel_multi_head_full_fuseO, smem_size);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, __kernel_multi_head_full_fuseO, num_thd,
        smem_size(num_thd));
    dim3 dimBlock(num_thd, 1, 1);
    dim3 dimGrid(cuda_helper.prop.multiProcessorCount * numBlocksPerSm, 1, 1);
    int kdim = model->kdim, vdim = model->vdim, seq_len = model->seq_len,
        nhead = model->nhead, blk_row_num = model->emdim / 16;
    void *kernelArgs[] = {(void *)&Q,
                          (void *)&K,
                          (void *)&V,
                          (void *)&Z,
                          (void *)&kdim,
                          (void *)&vdim,
                          (void *)&seq_len,
                          (void *)&nhead,
                          (void *)&mask,
                          (void *)&O_row_ptr,
                          (void *)&O_row_offset,
                          (void *)&O_data,
                          (void *)&blk_row_num,
                          (void *)&out,
                          (void *)&O_bias};
    cudaLaunchCooperativeKernel((void *)__kernel_multi_head_full_fuseO, dimGrid,
                                dimBlock, kernelArgs, smem_size(num_thd));
}