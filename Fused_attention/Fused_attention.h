#pragma once
#include "../Linear/Linear.h"
#include "../Model.h"
#include "../mycublas/cublas.h"
#include "Linear_VO.h"
#include <Attention/Attention.h>
#include <cublas.h>
#include <cuda_map.cuh>
#include <memory>
#include <type_traits>

template <typename attn_t>
void Fused_Multihead_attention(half *mat_q, half *mat_k, half *mat_vo,
                               half *mat_qk, const culib::CUDA_ptr<half> *mask,
                               half *output, std::shared_ptr<Model_t> model,
                               cublasHandle_t handle, cudaStream_t streamVO) {
    assert(-1 && "Unkown attention");
}
void Fused_prune_Multihead_attention(
    half *mat_q, half *mat_k, half *mat_vo, half *mat_qk,
    const culib::CUDA_ptr<half> *mask, half *output, half *mat_vos,
    std::shared_ptr<Model_t> model, cublasHandle_t handle,
    Linear_VO_prune *linearVO, cudaStream_t streamVO);

struct OTF_fused_attn {};
struct NOTF_fused_attn {};
struct NOTF_fused_prune_attn {};

template <typename T> struct __VO_helper { using type = Linear_VO; };
template <> struct __VO_helper<NOTF_fused_prune_attn> {
    using type = Linear_VO_prune;
};

template <typename _config> class FusedAttention {
    using ptr_t = culib::CUDA_ptr<half>;
    using config = _config;
    using linearQ_t = Linear<typename config::WQ_t>;
    using linearK_t = Linear<typename config::WK_t>;
    parallel_t<config::isPara, 3> para_handles;

  private:
    cublasHandle_t handle;
    std::unique_ptr<linearQ_t> linearQ;
    std::unique_ptr<linearK_t> linearK;

    using LinearVO_t = typename __VO_helper<typename config::attn_t>::type;
    std::unique_ptr<LinearVO_t> linearVO;

    ptr_t biasO;

    std::unique_ptr<ptr_t> mat_k;
    std::unique_ptr<ptr_t> mat_q;
    std::unique_ptr<ptr_t> mat_qk;
    std::unique_ptr<ptr_t> mat_vo;
    std::unique_ptr<ptr_t> mat_vos;
    std::shared_ptr<Model_t> model;

    ptr_t mask;
    int head_dim;

  public:
    FusedAttention(std::unique_ptr<linearQ_t> _linearQ,
                   std::unique_ptr<linearK_t> _linearK,
                   std::unique_ptr<LinearVO_t> _linearVO, const ptr_t &_biasO,
                   std::shared_ptr<Model_t> _model)
        : linearQ(std::move(_linearQ)), linearK(std::move(_linearK)),
          linearVO(std::move(_linearVO)), model(_model), biasO(_biasO),
          head_dim(model->kdim * model->seq_len),
          mask(model->seq_len * model->seq_len) {
        const auto len = model->seq_len;
        std::vector<half> h_mask(len * len);
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                h_mask[i * len + j] = (j <= i) ? half_zero : half_ninf;
            }
        }
        cudaMemcpy(mask.get(), h_mask.data(), sizeof(half) * len * len,
                   cudaMemcpyHostToDevice);
        mat_k = std::make_unique<ptr_t>(model->seq_len * model->kdim);
        mat_q = std::make_unique<ptr_t>(model->seq_len * model->kdim);
        mat_qk = std::make_unique<ptr_t>(model->seq_len * model->seq_len *
                                         model->nhead);
        mat_vo = std::make_unique<ptr_t>(model->emdim * model->nhead *
                                         model->seq_len);
        if constexpr (std::is_same_v<typename config::attn_t,
                                     NOTF_fused_prune_attn>) {
            mat_vos = std::make_unique<ptr_t>(linearVO->weight.col_id->size *
                                              model->seq_len);
            assert(mat_vos->get());
        }
        cublasCreate(&handle);
    }
    void forward(half *out, half *query, half *key, half *value) {
        linearVO->forward(mat_vo->get(), value, para_handles.streams[2]);
        linearQ->forward(mat_q->get(), query, para_handles.streams[0]);
        linearK->forward(mat_k->get(), key, para_handles.streams[1]);
        culib::util::sync_streams(para_handles.streams, 2);
        if constexpr (std::is_same_v<typename config::attn_t,
                                     NOTF_fused_prune_attn>) {
            Fused_prune_Multihead_attention(
                mat_q->get(), mat_k->get(), mat_vo->get(), mat_qk->get(), &mask,
                out, mat_vos->get(), model, handle, linearVO.get(),
                para_handles.streams[2]);
        } else {
            Fused_Multihead_attention<typename config::attn_t>(
                mat_q->get(), mat_k->get(), mat_vo->get(), mat_qk->get(), &mask,
                out, model, handle, para_handles.streams[2]);
        }
        auto bias_temp = biasO.get();
        const auto stride = model->emdim;
        const auto add_bias = [bias_temp, stride] __device__(half * data,
                                                             int i) -> half {
            return data[i] + bias_temp[i % stride];
        };
        culib::cuda_map(out, model->emdim * model->seq_len, add_bias);
    }
    ~FusedAttention() { cublasDestroy(handle); }
};
template <>
void Fused_Multihead_attention<OTF_fused_attn>(
    half *mat_q, half *mat_k, half *mat_vo, half *mat_qk,
    const culib::CUDA_ptr<half> *mask, half *output,
    std::shared_ptr<Model_t> model, cublasHandle_t handle,
    cudaStream_t streamVO);
template <>
void Fused_Multihead_attention<NOTF_fused_attn>(
    half *mat_q, half *mat_k, half *mat_vo, half *mat_qk,
    const culib::CUDA_ptr<half> *mask, half *output,
    std::shared_ptr<Model_t> model, cublasHandle_t handle,
    cudaStream_t streamVO);
