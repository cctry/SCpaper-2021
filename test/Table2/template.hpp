#include "../../Fused_attention/Fused_attention.h"
#include <Attention/Attention.h>
#include <CUDA_ptr.hpp>
#include <Encoder/Encoder.h>
#include <Linear/Linear.h>
#include <Model.h>
#include <bits/stdc++.h>
#include <wtime.h>
using namespace culib;
using model_para_t = std::shared_ptr<Model_t>;

template <class mat_t>
double test_uniform(model_para_t para, float sparsity, int nlayer) {
    using attn_config = Attention_config<mat_t, mat_t, mat_t, mat_t,
                                         OTF_attn_full, true, false>;
    using encoder_config = Encoder_config<attn_config, mat_t, mat_t, GELU_OP>;
    std::vector<std::unique_ptr<Encoder<encoder_config>>> model(nlayer);
    for (int i = 0; i < model.size(); i++) {
        auto LQKV = gen_sparse_linear<mat_t>(
            para->kdim * 2 + para->vdim, para->emdim, para->seq_len, sparsity);
        auto LO = gen_sparse_linear<mat_t>(para->emdim, para->vdim,
                                           para->seq_len, sparsity);
        auto attn = std::make_unique<Attention<attn_config>>(
            std::move(LQKV), std::move(LO), para);

        auto LN1 = gen_skLN(para->emdim);
        auto LN2 = gen_skLN(para->emdim);
        auto L1 = gen_sparse_linear<mat_t>(para->dimFF, para->emdim,
                                           para->seq_len, sparsity);
        auto L2 = gen_sparse_linear<mat_t>(para->emdim, para->dimFF,
                                           para->seq_len, sparsity);

        model[i] = std::make_unique<Encoder<encoder_config>>(
            std::move(attn), std::move(LN1), std::move(LN2), std::move(L1),
            std::move(L2), para);
    }
    CUDA_ptr<half> IN(para->emdim * para->seq_len, __float2half_rn(0.2));
    CUDA_ptr<half> d_OUT(para->emdim * para->seq_len);
    for (int i = 0; i < model.size(); i++) {
        model[i]->forward(d_OUT.get(), IN.get());
    }
    cudaDeviceSynchronize();
    auto time = wtime_new(
        10,
        [&]() {
            for (int i = 0; i < nlayer; i++) {
                model[i]->forward(d_OUT.get(), IN.get());
            }
        },
        []() { cudaDeviceSynchronize(); });
    return time;
}

inline double test_prun_attn1(model_para_t para, float sparsity, int nlayer) {
    using attn_config = Attention_config<tile_mat, tile_mat, row_mat, tile_mat,
                                         Prune_attn, false, true>;
    using encoder_config =
        Encoder_config<attn_config, tile_mat, tile_mat, GELU_OP>;
    std::vector<std::unique_ptr<Encoder<encoder_config>>> model(nlayer);
    for (int i = 0; i < model.size(); i++) {
        auto LQ = gen_sparse_linear<tile_mat>(para->kdim, para->emdim,
                                              para->seq_len, sparsity);
        auto LK = gen_sparse_linear<tile_mat>(para->kdim, para->emdim,
                                              para->seq_len, sparsity);
        auto LV = gen_sparse_linear<row_mat>(para->vdim, para->emdim,
                                             para->seq_len, sparsity);
        auto LO = gen_sparse_linear<tile_mat>(para->emdim, para->vdim,
                                              para->seq_len, sparsity);
        auto attn = std::make_unique<Attention<attn_config>>(
            std::move(LQ), std::move(LK), std::move(LV), std::move(LO), para);
        auto LN1 = gen_skLN(para->emdim);
        auto LN2 = gen_skLN(para->emdim);
        auto L1 = gen_sparse_linear<tile_mat>(para->dimFF, para->emdim,
                                              para->seq_len, sparsity);
        auto L2 = gen_sparse_linear<tile_mat>(para->emdim, para->dimFF,
                                              para->seq_len, sparsity);

        model[i] = std::make_unique<Encoder<encoder_config>>(
            std::move(attn), std::move(LN1), std::move(LN2), std::move(L1),
            std::move(L2), para);
    }
    CUDA_ptr<half> IN(para->emdim * para->seq_len, __float2half_rn(0.2));
    CUDA_ptr<half> d_OUT(para->emdim * para->seq_len);
    for (int i = 0; i < model.size(); i++) {
        model[i]->forward(d_OUT.get(), IN.get());
    }
    cudaDeviceSynchronize();
    auto time = wtime_new(
        10,
        [&]() {
            for (int i = 0; i < nlayer; i++) {
                model[i]->forward(d_OUT.get(), IN.get());
            }
        },
        []() { cudaDeviceSynchronize(); });
    return time;
}

// inline double test_prun_attn2(model_para_t para, float sparsity, int nlayer)
// {
//     using attn_config = Attention_config<tile_mat, tile_mat, tile_mat,
//     tile_mat,
//                                          NOTF_fused_prune_attn, false, true>;
//     using encoder_config =
//         Encoder_config<attn_config, tile_mat, tile_mat, GELU_OP>;
//     std::vector<std::unique_ptr<Encoder_fuse<attn_config>>> model(nlayer);
//     for (int i = 0; i < model.size(); i++) {
//         auto LQ = gen_sparse_linear<tile_mat>(para->kdim, para->emdim,
//                                               para->seq_len, sparsity);
//         auto LK = gen_sparse_linear<tile_mat>(para->kdim, para->emdim,
//                                               para->seq_len, sparsity);
//         // build LVO
//         auto weight = col_mat::gen_sparse_mat(para->nhead * para->emdim,
//                                               para->emdim, sparsity);
//         CUDA_ptr<half> bias(para->nhead * weight.col_id->size,
//                             __float2half_rn(0.2));
//         auto LVO = std::make_unique<Linear_VO_prune>(weight, bias, para);
//         CUDA_ptr<half> biasO(para->emdim);
//         FusedAttention<attn_config> attn(std::move(LQ), std::move(LK),
//                                          std::move(LVO), biasO, para);
//         auto LN1 = gen_skLN(para->emdim);
//         auto LN2 = gen_skLN(para->emdim);
//         auto L1 = gen_sparse_linear<tile_mat>(para->dimFF, para->emdim,
//                                               para->seq_len, sparsity);
//         auto L2 = gen_sparse_linear<tile_mat>(para->emdim, para->dimFF,
//                                               para->seq_len, sparsity);

//         model[i] = std::make_unique<Encoder_fuse<attn_config>>(
//             std::move(attn), std::move(LN1), std::move(LN2), std::move(L1),
//             std::move(L2), para);
//     }
//     CUDA_ptr<half> IN(para->emdim * para->seq_len, __float2half_rn(0.2));
//     CUDA_ptr<half> d_OUT(para->emdim * para->seq_len);
//     for (int i = 0; i < model.size(); i++) {
//         model[i]->forward(d_OUT.get(), IN.get());
//     }
//     cudaDeviceSynchronize();
//     auto time = wtime(
//         10,
//         [&]() {
//             for (int i = 0; i < model.size(); i++) {
//                 model[i]->forward(d_OUT.get(), IN.get());
//             }
//             cudaDeviceSynchronize();
//         },
//         []() {});
//     return time;
// }