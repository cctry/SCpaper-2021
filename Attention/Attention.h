#pragma once
#include "../Linear/Linear.h"
#include "../Model.h"
#include "kernel_fuseO.h"
#include <cublas.h>
#include <memory>
#include <type_traits>
using ptr_t = culib::CUDA_ptr<half>;

template <typename _WQ_t, typename _WK_t, typename _WV_t, typename _WO_t,
          typename _attn_t, bool _isSingleKernel = false, bool _isPara = true,
          typename =
              typename std::enable_if<!(_isSingleKernel && _isPara)>::type>
struct Attention_config {
    using WQ_t = _WQ_t;
    using WK_t = _WK_t;
    using WV_t = _WV_t;
    using WO_t = _WO_t;
    using attn_t = _attn_t;
    static constexpr bool isPara = _isPara;
    static constexpr bool isSingleKernel = _isSingleKernel;
    using W_t =
        typename culib::util::type_if<_isSingleKernel, _WQ_t, void>::type;
};

struct OTF_attn {};
struct Norm_attn {};
struct Prune_attn {};
struct OTF_attn_full {}; // reuse Q and finish V
struct OTF_attn_tile {}; // reuse Q and compute V with tiling
struct OTF_attn_sharedQK {};
struct OTF_attn_full_mixed {};
struct OTF_attn_Ofused {};

template <typename _attn_type>
void Multihead_atttion(ptr_t *mat_q, ptr_t *mat_k, ptr_t *mat_v, ptr_t *mat_qk,
                       const ptr_t *mask, ptr_t *mat_z,
                       std::shared_ptr<Model_t> model) {
    static_assert(-1 && "Unkown attention");
}

template <typename _attn_type>
void Multihead_atttion(ptr_t *mat_q, ptr_t *mat_k, col_mat *mat_v,
                       ptr_t *mat_qk, const ptr_t *mask, ptr_t *mat_z,
                       std::shared_ptr<Model_t> model);

template <typename T> struct __matv_helper { using type = ptr_t; };
template <> struct __matv_helper<Prune_attn> { using type = col_mat; };

template <bool _enable, int _nstream> struct parallel_t {
    static constexpr auto nstream = _nstream;
    using para_handle_t = int;
    para_handle_t streams[_nstream] = {0};
};

template <typename _config, typename = void> class Attention {};

template <typename _config>
class Attention<_config, std::enable_if_t<!_config::isSingleKernel>> {
    using config = _config;
    using linearQ_t = Linear<typename config::WQ_t>;
    using linearK_t = Linear<typename config::WK_t>;
    using linearV_t = Linear<typename config::WV_t>;
    using linearO_t = Linear<typename config::WO_t>;
    parallel_t<config::isPara, 3> para_handles;

  private:
    std::unique_ptr<linearQ_t> linearQ;
    std::unique_ptr<linearK_t> linearK;
    std::unique_ptr<linearV_t> linearV;
    std::unique_ptr<linearO_t> linearO;

    std::unique_ptr<ptr_t> mat_k;
    std::unique_ptr<ptr_t> mat_q;
    std::unique_ptr<ptr_t> mat_qk;
    std::unique_ptr<ptr_t> mat_z;
    using mat_v_t = typename __matv_helper<typename config::attn_t>::type;
    std::unique_ptr<mat_v_t> mat_v;

    std::shared_ptr<Model_t> model;

    ptr_t mask;
    int head_dim;

  public:
    Attention(std::unique_ptr<linearQ_t> _linearQ,
              std::unique_ptr<linearK_t> _linearK,
              std::unique_ptr<linearV_t> _linearV,
              std::unique_ptr<linearO_t> _linearO,
              std::shared_ptr<Model_t> _model)
        : linearQ(std::move(_linearQ)), linearK(std::move(_linearK)),
          linearV(std::move(_linearV)), linearO(std::move(_linearO)),
          model(_model), head_dim(model->kdim * model->seq_len),
          mask(model->seq_len * model->seq_len) {
        static_assert(std::is_same_v<typename config::W_t, void>);
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
        mat_z = std::make_unique<ptr_t>(model->seq_len * model->vdim);
        if constexpr (std::is_same_v<typename config::attn_t, Prune_attn>) {
            static_assert(std::is_same_v<decltype(linearV->weight), row_mat>);
            mat_v = std::make_unique<col_mat>(model->seq_len, model->vdim,
                                              linearV->weight.row_id->size);
            mat_v->set_metadata(*linearV->weight.row_id.get(),
                                linearV->weight.row_id->size, model->nhead,
                                model->vdim);
        } else {
            mat_v = std::make_unique<ptr_t>(model->seq_len * model->vdim);
        }
    }
    void forward(half *out, half *query, half *key, half *value) {
        linearQ->forward(mat_q->get(), query, para_handles.streams[0]);
        linearK->forward(mat_k->get(), key, para_handles.streams[1]);
        linearV->forward(mat_v->get(), value, para_handles.streams[2]);
        if constexpr (config::isPara)
            culib::util::sync_streams(para_handles.streams,
                                      decltype(para_handles)::nstream);
        if constexpr (std::is_same_v<OTF_attn_Ofused,
                                     typename config::attn_t>) {
            Multihead_atttion_fuseO(
                mat_q->get(), mat_k->get(), mat_v->get(), mask.get(),
                mat_z->get(), linearO->weight.row_ptr.get(),
                linearO->weight.row_offset.get(), linearO->weight.data.get(),
                linearO->bias.get(), model, out);
        } else {
            Multihead_atttion<typename config::attn_t>(
                mat_q.get(), mat_k.get(), mat_v.get(), mat_qk.get(), &mask,
                mat_z.get(), model);
            linearO->forward(out, mat_z->get());
        }
    }
};

template <>
void Multihead_atttion<OTF_attn>(ptr_t *mat_q, ptr_t *mat_k, ptr_t *mat_v,
                                 ptr_t *mat_qk, const ptr_t *mask, ptr_t *mat_z,
                                 std::shared_ptr<Model_t> model);

template <>
void Multihead_atttion<Norm_attn>(ptr_t *mat_q, ptr_t *mat_k, ptr_t *mat_v,
                                  ptr_t *mat_qk, const ptr_t *mask,
                                  ptr_t *mat_z, std::shared_ptr<Model_t> model);

template <>
void Multihead_atttion<Prune_attn>(ptr_t *mat_q, ptr_t *mat_k, col_mat *mat_v,
                                   ptr_t *mat_qk, const ptr_t *mask,
                                   ptr_t *mat_z,
                                   std::shared_ptr<Model_t> model);

template <>
void Multihead_atttion<OTF_attn_full>(ptr_t *mat_q, ptr_t *mat_k, ptr_t *mat_v,
                                      ptr_t *mat_qk, const ptr_t *mask,
                                      ptr_t *mat_z,
                                      std::shared_ptr<Model_t> model);

template <>
void Multihead_atttion<OTF_attn_tile>(ptr_t *mat_q, ptr_t *mat_k, ptr_t *mat_v,
                                      ptr_t *mat_qk, const ptr_t *mask,
                                      ptr_t *mat_z,
                                      std::shared_ptr<Model_t> model);
template <>
void Multihead_atttion<OTF_attn_sharedQK>(ptr_t *mat_q, ptr_t *mat_k,
                                          ptr_t *mat_v, ptr_t *mat_qk,
                                          const ptr_t *mask, ptr_t *mat_z,
                                          std::shared_ptr<Model_t> model);

template <>
void Multihead_atttion<OTF_attn_full_mixed>(ptr_t *mat_q, ptr_t *mat_k,
                                            ptr_t *mat_v, ptr_t *mat_qk,
                                            const ptr_t *mask, ptr_t *mat_z,
                                            std::shared_ptr<Model_t> model);

template <int _nstream> struct parallel_t<true, _nstream> {
    static constexpr auto nstream = _nstream;
    cudaStream_t streams[_nstream];
    parallel_t() {
        for (int i = 0; i < _nstream; i++) {
            cudaStreamCreate(&streams[i]);
        }
    }
    ~parallel_t() {
        for (int i = 0; i < _nstream; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }
};

__global__ void __kernel_multi_head_full_single_kernel(
    const half *__restrict__ Q, const half *__restrict__ K,
    const half *__restrict__ V, half *__restrict__ Z, const int kdim,
    const int vdim, const int seq_len, const int num_head,
    const half *__restrict__ mask, const int ldm);

template <typename _config>
class Attention<_config, std::enable_if_t<_config::isSingleKernel>> {
    using config = _config;
    using linearQKV_t = Linear<typename config::W_t>;
    using linearO_t = Linear<typename config::WO_t>;

  private:
    std::unique_ptr<linearQKV_t> linearQKV;
    std::unique_ptr<linearO_t> linearO;

    std::unique_ptr<ptr_t> mat_qkv;
    std::unique_ptr<ptr_t> mat_qk;
    std::unique_ptr<ptr_t> mat_z;

    std::shared_ptr<Model_t> model;

    ptr_t mask;
    int head_dim;

  public:
    Attention(std::unique_ptr<linearQKV_t> _linearQKV,
              std::unique_ptr<linearO_t> _linearO,
              std::shared_ptr<Model_t> _model)
        : linearQKV(std::move(_linearQKV)), linearO(std::move(_linearO)),
          model(_model), head_dim(model->kdim * model->seq_len),
          mask(model->seq_len * model->seq_len) {
        static_assert(!std::is_same_v<typename config::W_t, void>);
        const auto len = model->seq_len;
        std::vector<half> h_mask(len * len);
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                h_mask[i * len + j] = (j <= i) ? half_zero : half_ninf;
            }
        }
        cudaMemcpy(mask.get(), h_mask.data(), sizeof(half) * len * len,
                   cudaMemcpyHostToDevice);
        mat_qkv = std::make_unique<ptr_t>(model->seq_len *
                                          (model->kdim * 2 + model->vdim));
        mat_qk = std::make_unique<ptr_t>(model->seq_len * model->seq_len *
                                         model->nhead);
        mat_z = std::make_unique<ptr_t>(model->seq_len * model->vdim);
    }
    void forward(half *out, half *query, half *key, half *value) {
        assert(query == key && key == value);
        linearQKV->forward(mat_qkv->get(), query);
        constexpr auto FP16_skew = 16;
    auto tempQ_size = 16 * ((model->kdim / model->nhead) + FP16_skew);
    auto temp_row_size = 16 * (model->seq_len + FP16_skew);
    auto smem_size = sizeof(half) * (tempQ_size + temp_row_size);
        int num_thd, _num_blk;
        cudaChk(cudaOccupancyMaxPotentialBlockSize(
            &_num_blk, &num_thd, __kernel_multi_head_full_single_kernel,
            smem_size));
        const auto num_blk = dim3(model->seq_len / 16, model->nhead);
        half *Q = mat_qkv->get();
        half *K = Q + model->kdim;
        half *V = K + model->vdim;
        __kernel_multi_head_full_single_kernel<<<num_blk, num_thd, smem_size>>>(
            Q, K, V, mat_z->get(), model->kdim, model->vdim, model->seq_len,
            model->nhead, mask.get(), model->kdim * 2 + model->vdim);
        linearO->forward(out, mat_z->get());
    }
};