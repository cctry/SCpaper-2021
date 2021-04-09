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
constexpr int decoder_size = 28800;

template <class mat_t>
double test_uniform(model_para_t para, float sparsity) {
    using attn_config = Attention_config<mat_t, mat_t, mat_t, mat_t,
                                         OTF_attn_full, true, false>;
    using encoder_config = Encoder_config<attn_config, mat_t, mat_t, GELU_OP>;
    auto LQKV = gen_sparse_linear<mat_t>(para->kdim * 2 + para->vdim,
                                         para->emdim, para->seq_len, sparsity);
    auto LO = gen_sparse_linear<mat_t>(para->emdim, para->vdim, para->seq_len,
                                       sparsity);
    auto attn = std::make_unique<Attention<attn_config>>(std::move(LQKV),
                                                         std::move(LO), para);

    auto LN1 = gen_skLN(para->emdim);
    auto LN2 = gen_skLN(para->emdim);
    auto L1 = gen_sparse_linear<mat_t>(para->dimFF, para->emdim, para->seq_len,
                                       sparsity);
    auto L2 = gen_sparse_linear<mat_t>(para->emdim, para->dimFF, para->seq_len,
                                       sparsity);

    auto decoder = gen_sparse_linear<mat_t>(decoder_size, para->emdim,
                                            para->seq_len, sparsity);
    auto encoder = std::make_unique<Encoder<encoder_config>>(
        std::move(attn), std::move(LN1), std::move(LN2), std::move(L1),
        std::move(L2), para);
    CUDA_ptr<half> IN(para->emdim * para->seq_len, __float2half_rn(0.2));
    CUDA_ptr<half> temp(para->emdim * para->seq_len);
    CUDA_ptr<half> d_OUT(para->emdim * decoder_size);
    encoder->forward(temp.get(), IN.get());
    decoder->forward(d_OUT.get(), temp.get());
    cudaDeviceSynchronize();
    auto time = wtime_new(
        10,
        [&]() {
            encoder->forward(temp.get(), IN.get());
            decoder->forward(d_OUT.get(), temp.get());
        },
        []() { cudaDeviceSynchronize(); });
    return time;
}

inline double test_prun_attn1(model_para_t para, float sparsity) {
    using attn_config = Attention_config<tile_mat, tile_mat, row_mat, tile_mat,
                                         Prune_attn, false, true>;
    using encoder_config =
        Encoder_config<attn_config, tile_mat, tile_mat, GELU_OP>;
    auto LQ = gen_sparse_linear<tile_mat>(para->kdim, para->emdim,
                                          para->seq_len, sparsity);
    auto LK = gen_sparse_linear<tile_mat>(para->kdim, para->emdim,
                                          para->seq_len, sparsity);
    auto LV = gen_sparse_linear<row_mat>(para->vdim, para->emdim, para->seq_len,
                                         sparsity);
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

    auto decoder = gen_sparse_linear<tile_mat>(decoder_size, para->emdim,
                                            para->seq_len, sparsity);
    auto encoder = std::make_unique<Encoder<encoder_config>>(
        std::move(attn), std::move(LN1), std::move(LN2), std::move(L1),
        std::move(L2), para);
    CUDA_ptr<half> IN(para->emdim * para->seq_len, __float2half_rn(0.2));
    CUDA_ptr<half> temp(para->emdim * para->seq_len);
    CUDA_ptr<half> d_OUT(para->emdim * decoder_size);
    encoder->forward(temp.get(), IN.get());
    decoder->forward(d_OUT.get(), temp.get());
    cudaDeviceSynchronize();
    auto time = wtime_new(
        10,
        [&]() {
            encoder->forward(temp.get(), IN.get());
            decoder->forward(d_OUT.get(), temp.get());
        },
        []() { cudaDeviceSynchronize(); });
    return time;
}