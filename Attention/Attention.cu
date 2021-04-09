#include "Attention.h"
#include "OTFkernel_full.h"
#include "kernel_fuseO.h"
#include "kernels.h"
#include "mixed-precision/mixed_attn.h"
#include "prune_attn.h"
#include <array>
template <>
void Multihead_atttion<Norm_attn>(ptr_t *mat_q, ptr_t *mat_k, ptr_t *mat_v,
                                  ptr_t *mat_qk, const ptr_t *mask,
                                  ptr_t *mat_z,
                                  std::shared_ptr<Model_t> model) {
    const auto head_dim = model->kdim / model->nhead;
    const half scale = __float2half(sqrtf(1.0f / head_dim));
    auto scale_op = [scale] __device__(half * data, int i) -> half {
        return data[i] * scale;
    };
    culib::cuda_map(mat_q->get(), model->kdim * model->seq_len, scale_op, 0);
    const auto mask_c = mask->get();
    const auto size = model->seq_len;
    auto mask_op = [mask_c, size] __device__(half * data, int i) -> half {
        const auto row_id = (i % (size * size)) % size;
        const auto col_id = (i % (size * size)) / size;
        return data[i] + mask_c[col_id + row_id * size];
    };
    cuBLAS_bmmul(mat_q->get(), mat_k->get(), mat_qk->get(), model->seq_len,
                 model->seq_len, head_dim, model->nhead, false, true);
    culib::cuda_map(mat_qk->get(),
                    model->seq_len * model->seq_len * model->nhead, mask_op, 0);
    __kernel_batch_softmax<<<dim3(model->nhead, model->seq_len),
                             culib::util::cuda_num_thd(model->seq_len / 2),
                             2 * model->seq_len * sizeof(half)>>>(
        mat_qk->get(), model->nhead, model->seq_len, model->seq_len);
    cuBLAS_bmmul(mat_qk->get(), mat_v->get(), mat_z->get(), model->seq_len,
                 model->vdim / model->nhead, model->seq_len, model->nhead);
}

template <>
void Multihead_atttion<OTF_attn>(ptr_t *mat_q, ptr_t *mat_k, ptr_t *mat_v,
                                 ptr_t *mat_qk, const ptr_t *mask, ptr_t *mat_z,
                                 std::shared_ptr<Model_t> model) {
    const int n = model->seq_len;
    auto get_smem_size = [n](int num_thd) {
        return sizeof(half) * (256 * n / 16 + n * num_thd / 32);
    };
    int num_thd, _num_blk;
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &_num_blk, &num_thd, __kernel_multi_head, get_smem_size, 32 * n / 16);
    const auto num_blk = dim3(model->vdim / 16, model->seq_len / 16);
    __kernel_multi_head<<<num_blk, num_thd, get_smem_size(num_thd)>>>(
        mat_q->get(), mat_k->get(), mat_v->get(), mat_z->get(), model->kdim,
        model->vdim, model->seq_len, model->nhead, mask->get());
}

template <>
void Multihead_atttion<Prune_attn>(ptr_t *mat_q, ptr_t *mat_k, col_mat *mat_v,
                                   ptr_t *mat_qk, const ptr_t *mask,
                                   ptr_t *mat_z,
                                   std::shared_ptr<Model_t> model) {
    const int seq_len = model->seq_len;
    const int head_dim = model->kdim / model->nhead;
    auto get_smem_size = [seq_len, head_dim](int num_thd) {
        const auto temp_row = 16 * (seq_len + FP16_skew) * sizeof(half);
        const auto temp_Q = 16 * (head_dim + FP16_skew) * sizeof(half);
        const auto temp_blk = 16 * ((num_thd / 2) + FP16_skew) * sizeof(half);
        const auto temp_id = 16 * (num_thd / 32) * sizeof(int);
        return temp_row + std::max(temp_Q, temp_blk + temp_id);
    };
    int num_thd, _num_blk;
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &_num_blk, &num_thd, __kernel_multi_head_prune, get_smem_size);
    const auto num_blk = dim3(seq_len / 16, model->nhead);
    __kernel_multi_head_prune<<<num_blk, num_thd, get_smem_size(num_thd)>>>(
        mat_q->get(), mat_k->get(), mat_v->cols->get(), mat_z->get(),
        model->kdim, model->vdim, model->seq_len, model->nhead,
        mat_v->col_id->get(), mat_v->head_ptr->get(), mat_v->nnz_col,
        mask->get());
}

template <>
void Multihead_atttion<OTF_attn_full>(ptr_t *mat_q, ptr_t *mat_k, ptr_t *mat_v,
                                      ptr_t *mat_qk, const ptr_t *mask,
                                      ptr_t *mat_z,
                                      std::shared_ptr<Model_t> model) {
    const int n = model->seq_len;
    constexpr int num_thd = 256;
    auto tempQ_size = 16 * ((model->kdim / model->nhead) + FP16_skew);
    auto temp_row_size = 16 * (n + FP16_skew);
    auto smem_size = sizeof(half) * (tempQ_size + temp_row_size);
    const auto num_blk = dim3(n / 16, model->nhead);
    __kernel_multi_head_full_skew_warpSFM<num_thd>
        <<<num_blk, num_thd, smem_size>>>(
            mat_q->get(), mat_k->get(), mat_v->get(), mat_z->get(), model->kdim,
            model->vdim, model->seq_len, model->nhead, mask->get());
}

template <>
void Multihead_atttion<OTF_attn_tile>(ptr_t *mat_q, ptr_t *mat_k, ptr_t *mat_v,
                                      ptr_t *mat_qk, const ptr_t *mask,
                                      ptr_t *mat_z,
                                      std::shared_ptr<Model_t> model) {
    constexpr int tile_ratio = 4;
    auto vhead_dim = (model->vdim / model->nhead);
    auto tile_size = (vhead_dim / 16) / tile_ratio;
    const int n = model->seq_len;
    auto get_smem_size = [&](int num_thd) {
        return sizeof(half) * (16 * (model->kdim / model->nhead + n));
    };
    int num_thd, _num_blk;
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &_num_blk, &num_thd, __kernel_multi_head, get_smem_size, 16 * 32);
    const auto num_blk = dim3(n / 16, model->nhead, tile_ratio);
    __kernel_multi_head_tile<<<num_blk, num_thd, get_smem_size(num_thd)>>>(
        mat_q->get(), mat_k->get(), mat_v->get(), mat_z->get(), model->kdim,
        model->vdim, model->seq_len, model->nhead, mask->get(), tile_size);
}

template <>
void Multihead_atttion<OTF_attn_sharedQK>(ptr_t *mat_q, ptr_t *mat_k,
                                          ptr_t *mat_v, ptr_t *mat_qk,
                                          const ptr_t *mask, ptr_t *mat_z,
                                          std::shared_ptr<Model_t> model) {
    const auto head_dim = model->kdim / model->nhead;
    // const half scale = __float2half(sqrtf(1.0f / head_dim));
    // auto scale_op = [scale] __device__(half * data, int i) -> half {
    //     return data[i] * scale;
    // };
    // culib::cuda_map(mat_q->get(), model->kdim * model->seq_len, scale_op, 0);
    cuBLAS_bmmul(mat_q->get(), mat_k->get(), mat_qk->get(), model->seq_len,
                 model->seq_len, head_dim, model->nhead, false, true);
    // int num_thd, _num_blk;
    // cudaOccupancyMaxPotentialBlockSize(
    //     &_num_blk, &num_thd, __kernel_multi_head_sharedQK,
    //     sizeof(half) * 16 * model->seq_len, 16 * 32);
    constexpr int num_thd = 256;
    const auto num_blk = dim3(model->seq_len / 16, model->nhead);
    __kernel_multi_head_sharedQK<num_thd>
        <<<num_blk, num_thd,
           sizeof(half) * 16 * (model->seq_len + FP16_skew)>>>(
            mat_qk->get(), mat_v->get(), mat_z->get(), model->vdim,
            model->seq_len, model->nhead, mask->get());
}
