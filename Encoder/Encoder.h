#pragma once
#include "../Attention/Attention.h"
#include "../Fused_attention/Fused_attention.h"
#include "../LayerNorm/LayerNorm.h"
#include "../Linear/Linear.h"
#include "../Model.h"
#include "../SkipLayerNorm/SkipLayerNorm.h"
#include <Attention/Attention_single.h>
#include <utils.h>
template <typename _attn, typename _l1, typename _l2, typename _act_t>
struct Encoder_config {
    using attn_config = _attn;
    using linear1_config = _l1;
    using linear2_config = _l2;
    using act_t = _act_t;
};

struct RELU_OP {};

struct GELU_OP {};

namespace GELU {
void computeGelu(cudaStream_t stream, int n, const float *input, float *output);
void computeGelu(cudaStream_t stream, int n, const half *input, half *output);
} // namespace GELU

template <typename ACT_OP_t> void act(half *data, int size) {
    if constexpr (std::is_same_v<ACT_OP_t, RELU_OP>) {
        auto relu = [] __device__(half * data, int i) -> half {
            const half th = half_zero;
            return data[i] > th ? data[i] : th;
        };
        culib::cuda_map(data, size, relu);
    } else if constexpr (std::is_same_v<ACT_OP_t, GELU_OP>) {
        GELU::computeGelu(0, size, data, data);
    }
}

template <typename _config> class Encoder {
    using config = _config;
    using attn_t = Attention<typename config::attn_config>;
    using linear1_t = Linear<typename config::linear1_config>;
    using linear2_t = Linear<typename config::linear2_config>;
    using ACT_OP_t = typename config::act_t;

  private:
    std::unique_ptr<attn_t> attn;
    std::unique_ptr<SkipLayerNorm> LN1;
    std::unique_ptr<SkipLayerNorm> LN2;
    std::unique_ptr<linear1_t> linear1;
    std::unique_ptr<linear2_t> linear2;
    std::shared_ptr<Model_t> model;

  public:
    culib::CUDA_ptr<half> temp_linear;
    culib::CUDA_ptr<half> temp_attn;
    culib::CUDA_ptr<half> temp_LN;

    Encoder(std::unique_ptr<attn_t> _attn, std::unique_ptr<SkipLayerNorm> _LN1,
            std::unique_ptr<SkipLayerNorm> _LN2,
            std::unique_ptr<linear1_t> _linear1,
            std::unique_ptr<linear1_t> _linear2,
            std::shared_ptr<Model_t> _model)
        : model(_model), attn(std::move(_attn)), LN1(std::move(_LN1)),
          LN2(std::move(_LN2)), linear1(std::move(_linear1)),
          linear2(std::move(_linear2)),
          temp_attn(model->seq_len * model->emdim),
          temp_linear(model->seq_len * model->dimFF),
          temp_LN(model->seq_len * model->emdim) {}

    void forward(half *out, half *src) {
        attn->forward(temp_attn.get(), src, src, src);
        LN1->forward(temp_LN.get(), temp_attn.get(), src, model->seq_len);
        linear1->forward(temp_linear.get(), temp_LN.get());
        // act<ACT_OP_t>(temp_linear.get(), model->dimFF * model->seq_len);
        linear2->forward(temp_attn.get(), temp_linear.get());
        LN2->forward(out, temp_attn.get(), temp_LN.get(), model->seq_len);
    }
};

template <typename _attn_config> class Encoder_fuse {
    using attn_t = FusedAttention<_attn_config>;
    using linear1_t = Linear<tile_mat>;
    using linear2_t = Linear<tile_mat>;
    using ACT_OP_t = GELU_OP;

  private:
    std::unique_ptr<attn_t> attn;
    std::unique_ptr<SkipLayerNorm> LN1;
    std::unique_ptr<SkipLayerNorm> LN2;
    std::unique_ptr<linear1_t> linear1;
    std::unique_ptr<linear2_t> linear2;
    std::shared_ptr<Model_t> model;

  public:
    culib::CUDA_ptr<half> temp_linear;
    culib::CUDA_ptr<half> temp_attn;
    culib::CUDA_ptr<half> temp_LN;

    Encoder_fuse(std::unique_ptr<attn_t> _attn,
                 std::unique_ptr<SkipLayerNorm> _LN1,
                 std::unique_ptr<SkipLayerNorm> _LN2,
                 std::unique_ptr<linear1_t> _linear1,
                 std::unique_ptr<linear1_t> _linear2,
                 std::shared_ptr<Model_t> _model)
        : model(_model), attn(std::move(_attn)), LN1(std::move(_LN1)),
          LN2(std::move(_LN2)), linear1(std::move(_linear1)),
          linear2(std::move(_linear2)),
          temp_attn(model->seq_len * model->emdim),
          temp_linear(model->seq_len * model->dimFF),
          temp_LN(model->seq_len * model->emdim) {}

    void forward(half *out, half *src) {
        attn->forward(temp_attn.get(), src, src, src);
        LN1->forward(temp_LN.get(), temp_attn.get(), src, model->seq_len);
        linear1->forward(temp_linear.get(), temp_LN.get());
        act<ACT_OP_t>(temp_linear.get(), model->dimFF * model->seq_len);
        linear2->forward(temp_attn.get(), temp_linear.get());
        LN2->forward(out, temp_attn.get(), temp_LN.get(), model->seq_len);
    }
};
