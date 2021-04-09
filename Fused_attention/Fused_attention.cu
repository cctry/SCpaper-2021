#include "Fused_attention.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <mma.hpp>
namespace cg = cooperative_groups;

using frag_t = culib::mma::mma_t<16, 16, 16>;

__global__ void __kernel_fused_attention(
    const half *__restrict__ mat_q, const half *__restrict__ mat_k,
    const half *__restrict__ mat_vo, const half *__restrict__ mask,
    half *__restrict__ output, const int kdim, const int emdim,
    const int seq_len, const int num_head) {
    /*
     * temp_row: seq_len * 16
     * Q_row: head_dim * 16
     * VO_temp: num_warp * 256
     * total_size = temp_row + max(Q_row, VO_temp)
     */
    extern __shared__ half smem[];
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    const auto warp_id = warp.meta_group_rank();
    const auto lane_id = warp.thread_rank();
    const auto num_warp = warp.meta_group_size();
    auto temp_row = smem;
    auto Q_row = &temp_row[seq_len * 16];
    const auto head_id = blockIdx.y;
    const auto blk_row_id = blockIdx.x;
    const auto head_dim = kdim / num_head;
    // load and scale Q
    const half scale = hsqrt(__int2half_rn(head_dim));
    auto Q_ptr = &mat_q[16 * blk_row_id * kdim + head_dim * head_id];
    for (int r = warp_id; r < 16; r += num_warp) {
        auto dst = &Q_row[r * head_dim];
        auto src = &Q_ptr[r * kdim];
        cg::memcpy_async(warp, dst, src, sizeof(half) * head_dim);
        auto dst2 = reinterpret_cast<half2 *>(dst);
        for (int i = lane_id; i < head_dim / 2; i += 32) {
            dst2[i] = __h2div(dst2[i], half2{scale, scale});
        }
    }
    cta.sync();
    // compute block row of S
    frag_t::a_t<wmma::row_major> a_frag;
    frag_t::b_t<wmma::col_major> b_frag;
    frag_t::c_t<half> c_frag;
    for (int KR = warp_id; KR < seq_len / 16; KR += num_warp) {
        auto K_ptr = &mat_k[16 * KR * kdim + head_dim * head_id];
        wmma::fill_fragment(c_frag, half_zero);
        for (int i = 0; i < head_dim; i += 16) {
            wmma::load_matrix_sync(a_frag, Q_row + i, head_dim);
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
    for (int i = cta.thread_rank(); i < seq_len * 8; i += cta.size()) {
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
            row_ptr[i] = __hdiv(row_ptr[i], sum);
        }
    }
    cta.sync();
    // compute output
    auto VO_temp = &Q_row[warp_id * 256];
    for (int VOC = warp_id * 16; VOC < emdim; VOC += num_warp * 16) {
        frag_t::b_t<wmma::row_major> b_frag;
        wmma::fill_fragment(c_frag, half_zero);
        for (int i = 0; i < seq_len; i += 16) {
            wmma::load_matrix_sync(a_frag, temp_row + i, seq_len);
            auto src = &mat_vo[(head_id * seq_len + i) * emdim + VOC];
            wmma::load_matrix_sync(b_frag, src, emdim);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(VO_temp, c_frag, 16, wmma::mem_row_major);
        auto res = &output[blk_row_id * 16 * emdim + VOC];
        auto subg = cg::tiled_partition<8>(warp);
#pragma unroll 4
        for (int k = subg.meta_group_rank(); k < 16; k += 4) {
            auto dst = reinterpret_cast<half2 *>(&res[k * emdim]);
            auto src = reinterpret_cast<half2 *>(&VO_temp[k * 16]);
            const auto tid = subg.thread_rank();
            atomicAdd(&dst[tid], src[tid]);
        }
    }
}

__global__ void __kernel_fused_attention_NOTF(
    const half *__restrict__ mat_q, const half *__restrict__ mat_k,
    const half *__restrict__ mask, half *__restrict__ output, const int kdim,
    const int emdim, const int seq_len, const int num_head) {
    /*
     * temp_row: seq_len * 16
     * Q_row: head_dim * 16
     * VO_temp: num_warp * 256
     * total_size = temp_row + max(Q_row, VO_temp)
     */
    extern __shared__ half smem[];
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    const auto warp_id = warp.meta_group_rank();
    const auto lane_id = warp.thread_rank();
    const auto num_warp = warp.meta_group_size();
    auto temp_row = smem;
    auto Q_row = &temp_row[seq_len * 16];
    const auto head_id = blockIdx.y;
    const auto blk_row_id = blockIdx.x;
    const auto head_dim = kdim / num_head;
    // load and scale Q
    const half scale = hsqrt(__int2half_rn(head_dim));
    auto Q_ptr = &mat_q[16 * blk_row_id * kdim + head_dim * head_id];
    for (int r = warp_id; r < 16; r += num_warp) {
        auto dst = &Q_row[r * head_dim];
        auto src = &Q_ptr[r * kdim];
        cg::memcpy_async(warp, dst, src, sizeof(half) * head_dim);
        auto dst2 = reinterpret_cast<half2 *>(dst);
        for (int i = lane_id; i < head_dim / 2; i += 32) {
            dst2[i] = __h2div(dst2[i], half2{scale, scale});
        }
    }
    cta.sync();
    // compute block row of S
    frag_t::a_t<wmma::row_major> a_frag;
    frag_t::b_t<wmma::col_major> b_frag;
    frag_t::c_t<half> c_frag;
    for (int KR = warp_id; KR < seq_len / 16; KR += num_warp) {
        auto K_ptr = &mat_k[16 * KR * kdim + head_dim * head_id];
        wmma::fill_fragment(c_frag, half_zero);
        for (int i = 0; i < head_dim; i += 16) {
            wmma::load_matrix_sync(a_frag, Q_row + i, head_dim);
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
    for (int i = cta.thread_rank(); i < seq_len * 8; i += cta.size()) {
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
            row_ptr[i] = __hdiv(row_ptr[i], sum);
        }
        warp.sync();
        // write S
        auto dst = &output[(blk_row_id * 16 + row) * num_head * seq_len +
                           head_id * seq_len];
        cg::memcpy_async(warp, dst, row_ptr, sizeof(half) * seq_len);
        cg::wait(warp);
    }
}
template <>
void Fused_Multihead_attention<NOTF_fused_attn>(
    half *mat_q, half *mat_k, half *mat_vo, half *mat_qk,
    const culib::CUDA_ptr<half> *mask, half *output,
    std::shared_ptr<Model_t> model, cublasHandle_t handle,
    cudaStream_t streamVO) {
    auto smem_size = [&](int n) {
        auto temp_row = model->seq_len * 16;
        auto Q_row = (model->kdim / model->nhead) * 16;
        auto total_size = temp_row + Q_row;
        return sizeof(half) * total_size;
    };
    int num_thd, _num_blk;
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &_num_blk, &num_thd, __kernel_fused_attention_NOTF, smem_size);
    const auto num_blk = dim3(model->seq_len / 16, model->nhead);
    __kernel_fused_attention_NOTF<<<num_blk, num_thd, smem_size(num_thd)>>>(
        mat_q, mat_k, mask->get(), mat_qk, model->kdim, model->emdim,
        model->seq_len, model->nhead);
    cudaStreamSynchronize(streamVO);
    cuBLAS_mmul(handle, mat_qk, mat_vo, output, model->seq_len, model->emdim,
                model->nhead * model->seq_len, false, false);
}

template <>
void Fused_Multihead_attention<OTF_fused_attn>(
    half *mat_q, half *mat_k, half *mat_vo, half *mat_qk,
    const culib::CUDA_ptr<half> *mask, half *output,
    std::shared_ptr<Model_t> model, cublasHandle_t handle,
    cudaStream_t streamVO) {
    auto smem_size = [&](int n) {
        auto temp_row = model->seq_len * 16;
        auto Q_row = (model->kdim / model->nhead) * 16;
        auto VO_temp = (n / 32) * 256;
        auto total_size = temp_row + std::max(Q_row, VO_temp);
        return sizeof(half) * total_size;
    };
    int num_thd, _num_blk;
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &_num_blk, &num_thd, __kernel_fused_attention, smem_size,
        model->seq_len * 2);
    const auto num_blk = dim3(model->seq_len / 16, model->nhead);
    cudaStreamSynchronize(streamVO);
    __kernel_fused_attention<<<num_blk, num_thd, smem_size(num_thd)>>>(
        mat_q, mat_k, mat_vo, mask->get(), output, model->kdim, model->emdim,
        model->seq_len, model->nhead);
}

__global__ void __decompress_row_mat(const half __restrict__ *cols,
                                     const int *idx, half __restrict__ *res,
                                     const int len_idx, const int len_col,
                                     const int ncol) {
    extern __shared__ int smem2[];
    auto cta = cg::this_thread_block();
    cg::memcpy_async(cta, smem2, idx, sizeof(int) * len_idx);
    const auto grid = cg::this_grid();
    const auto tid = grid.thread_rank();
    cg::wait(cta);
    for (int i = tid; i < len_idx * len_col; i += grid.size()) {
        const auto r = i / len_idx;
        const auto offset = i % len_idx;
        const auto c = smem2[offset];
        res[r * ncol + c] = cols[r * len_idx + offset];
    }
}

void Fused_prune_Multihead_attention(
    half *mat_q, half *mat_k, half *mat_vo, half *mat_qk,
    const culib::CUDA_ptr<half> *mask, half *output, half *mat_vos,
    std::shared_ptr<Model_t> model, cublasHandle_t handle,
    Linear_VO_prune *linearVO, cudaStream_t streamVO) {
    cudaStream_t temp_stream;
    cublasGetStream(handle, &temp_stream);
    col_mat &mat = linearVO->weight;

    auto smem_size = [&](int n) {
        auto temp_row = model->seq_len * 16;
        auto Q_row = (model->kdim / model->nhead) * 16;
        auto VO_temp = (n / 32) * 256;
        auto total_size = temp_row + std::max(Q_row, VO_temp);
        return sizeof(half) * total_size;
    };
    int num_thd, _num_blk;
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &_num_blk, &num_thd, __kernel_fused_attention, smem_size,
        model->seq_len * 2);
    const auto num_blk = dim3(model->seq_len / 16, model->nhead);
    __kernel_fused_attention_NOTF<<<num_blk, num_thd, smem_size(num_thd),
                                    temp_stream>>>(
        mat_q, mat_k, mask->get(), mat_qk, model->kdim, model->emdim,
        model->seq_len, model->nhead);
    cudaStreamSynchronize(streamVO);
    cuBLAS_mmul(handle, mat_qk, mat_vo, mat_vos, model->seq_len,
                mat.col_id->size, model->nhead * model->seq_len, false, false);

    auto shared_size = mat.col_id->size * sizeof(int);
    cudaOccupancyMaxPotentialBlockSize(&_num_blk, &num_thd,
                                       __decompress_row_mat, shared_size);

    __decompress_row_mat<<<_num_blk, num_thd, shared_size, temp_stream>>>(
        mat_vos, mat.col_id->get(), output, mat.col_id->size, model->seq_len,
        model->emdim);
}