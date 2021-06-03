#include "weights.hpp"
#include <Attention/Attention.h>
#include <CUDA_ptr.hpp>
#include <Encoder/Encoder.h>
#include <Linear/Linear.h>
#include <Model.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using namespace culib;

using mat_t = tile_mat;
using attn_t = OTF_attn_full;
using attn_config =
    Attention_config<mat_t, mat_t, mat_t, mat_t, attn_t, true, false>;
using encoder_config = Encoder_config<attn_config, mat_t, mat_t, RELU_OP>;

CUDA_ptr<half> random_input(int nrow, int ncol);
std::vector<half> vec_F2H(const std::vector<float> &arr);

int main(int ac, char **av) {
    const auto weight_file = std::string(av[1]);
    const auto weights_path = std::string(av[2]);
    // Transformer
    constexpr auto TEXT_size = 1800 * 16;
    auto para =
        std::make_shared<Model_t>(Model_t{800, 800, 128, 4, 208, 800, 2});
    // Input
    auto IN = random_input(para->seq_len, para->emdim);
    // Output
    auto OUT1 = CUDA_ptr<half>(para->seq_len * para->emdim);
    auto OUT2 = CUDA_ptr<half>(para->seq_len * para->emdim);
    auto OUT = CUDA_ptr<half>(para->seq_len * TEXT_size);
    // Build model
    Weights weights(weight_file, weights_path);
    for (auto name : weights.layer_names) {
        std::cout << name << std::endl;
    }
    std::vector<std::unique_ptr<Encoder<encoder_config>>> model(para->nlayer);

    {
        // QKV
        const auto QKVname =
            "transformer_encoder.layers.0.self_attn.in_proj_.npz";
        tile_mat WQKV(weights.load_mat<int>("row_ptr", QKVname),
                      weights.load_mat<int>("column", QKVname),
                      vec_F2H(weights.load_mat<float>("data", QKVname)),
                      para->emdim / 16);
        culib::CUDA_ptr<half> BQKV(
            vec_F2H(weights.load_mat<float>("B", QKVname)));
        auto LQKV = std::make_unique<Linear<mat_t>>(para->emdim, para->kdim * 3,
                                                    WQKV, BQKV, para->seq_len);
        // O
        const auto Oname =
            "transformer_encoder.layers.0.self_attn.out_proj..npz";
        tile_mat WO(weights.load_mat<int>("row_ptr", Oname),
                    weights.load_mat<int>("column", Oname),
                    vec_F2H(weights.load_mat<float>("data", Oname)),
                    para->vdim / 16);
        culib::CUDA_ptr<half> BO(vec_F2H(weights.load_mat<float>("B", Oname)));
        auto LO = std::make_unique<Linear<mat_t>>(para->vdim, para->emdim, WO,
                                                  BO, para->seq_len);
        auto attn = std::make_unique<Attention<attn_config>>(
            std::move(LQKV), std::move(LO), para);
        attn->forward(OUT.get(), IN.get(), IN.get(), IN.get());
        // LN1
        const auto LN1name = "transformer_encoder.layers.0.norm1..npz";
        auto LN1 = std::make_unique<SkipLayerNorm>(
            vec_F2H(weights.load_mat<float>("W", LN1name)).data(),
            vec_F2H(weights.load_mat<float>("B", LN1name)).data(), para->emdim);
        // LN2
        const auto LN2name = "transformer_encoder.layers.0.norm2..npz";
        auto LN2 = std::make_unique<SkipLayerNorm>(
            vec_F2H(weights.load_mat<float>("W", LN2name)).data(),
            vec_F2H(weights.load_mat<float>("B", LN2name)).data(), para->emdim);
        // L1
        const auto L1name = "transformer_encoder.layers.0.linear1..npz";
        tile_mat WL1(weights.load_mat<int>("row_ptr", L1name),
                     weights.load_mat<int>("column", L1name),
                     vec_F2H(weights.load_mat<float>("data", L1name)),
                     para->emdim / 16);
        culib::CUDA_ptr<half> BL1(
            vec_F2H(weights.load_mat<float>("B", L1name)));
        auto L1 = std::make_unique<Linear<mat_t>>(para->emdim, para->dimFF, WL1,
                                                  BL1, para->seq_len);
        // L2
        const auto L2name = "transformer_encoder.layers.0.linear2..npz";
        tile_mat WL2(weights.load_mat<int>("row_ptr", L2name),
                     weights.load_mat<int>("column", L2name),
                     vec_F2H(weights.load_mat<float>("data", L2name)),
                     para->dimFF / 16);
        culib::CUDA_ptr<half> BL2(
            vec_F2H(weights.load_mat<float>("B", L2name)));
        auto L2 = std::make_unique<Linear<mat_t>>(para->dimFF, para->emdim, WL2,
                                                  BL2, para->seq_len);
        model[0] = std::make_unique<Encoder<encoder_config>>(
            std::move(attn), std::move(LN1), std::move(LN2), std::move(L1),
            std::move(L2), para);
    }

    {
        // QKV
        const auto QKVname =
            "transformer_encoder.layers.1.self_attn.in_proj_.npz";
        tile_mat WQKV(weights.load_mat<int>("row_ptr", QKVname),
                      weights.load_mat<int>("column", QKVname),
                      vec_F2H(weights.load_mat<float>("data", QKVname)),
                      para->emdim / 16);
        culib::CUDA_ptr<half> BQKV(
            vec_F2H(weights.load_mat<float>("B", QKVname)));
        auto LQKV = std::make_unique<Linear<mat_t>>(para->emdim, para->kdim * 3,
                                                    WQKV, BQKV, para->seq_len);
        // O
        const auto Oname =
            "transformer_encoder.layers.1.self_attn.out_proj..npz";
        tile_mat WO(weights.load_mat<int>("row_ptr", Oname),
                    weights.load_mat<int>("column", Oname),
                    vec_F2H(weights.load_mat<float>("data", Oname)),
                    para->vdim / 16);
        culib::CUDA_ptr<half> BO(vec_F2H(weights.load_mat<float>("B", Oname)));
        auto LO = std::make_unique<Linear<mat_t>>(para->vdim, para->emdim, WO,
                                                  BO, para->seq_len);
        auto attn = std::make_unique<Attention<attn_config>>(
            std::move(LQKV), std::move(LO), para);
        attn->forward(OUT.get(), IN.get(), IN.get(), IN.get());
        // LN1
        const auto LN1name = "transformer_encoder.layers.1.norm1..npz";
        auto LN1 = std::make_unique<SkipLayerNorm>(
            vec_F2H(weights.load_mat<float>("W", LN1name)).data(),
            vec_F2H(weights.load_mat<float>("B", LN1name)).data(), para->emdim);
        // LN2
        const auto LN2name = "transformer_encoder.layers.1.norm2..npz";
        auto LN2 = std::make_unique<SkipLayerNorm>(
            vec_F2H(weights.load_mat<float>("W", LN2name)).data(),
            vec_F2H(weights.load_mat<float>("B", LN2name)).data(), para->emdim);
        // L1
        const auto L1name = "transformer_encoder.layers.1.linear1..npz";
        tile_mat WL1(weights.load_mat<int>("row_ptr", L1name),
                     weights.load_mat<int>("column", L1name),
                     vec_F2H(weights.load_mat<float>("data", L1name)),
                     para->emdim / 16);
        culib::CUDA_ptr<half> BL1(
            vec_F2H(weights.load_mat<float>("B", L1name)));
        auto L1 = std::make_unique<Linear<mat_t>>(para->emdim, para->dimFF, WL1,
                                                  BL1, para->seq_len);
        // L2
        const auto L2name = "transformer_encoder.layers.1.linear2..npz";
        tile_mat WL2(weights.load_mat<int>("row_ptr", L2name),
                     weights.load_mat<int>("column", L2name),
                     vec_F2H(weights.load_mat<float>("data", L2name)),
                     para->dimFF / 16);
        culib::CUDA_ptr<half> BL2(
            vec_F2H(weights.load_mat<float>("B", L2name)));
        auto L2 = std::make_unique<Linear<mat_t>>(para->dimFF, para->emdim, WL2,
                                                  BL2, para->seq_len);
        model[1] = std::make_unique<Encoder<encoder_config>>(
            std::move(attn), std::move(LN1), std::move(LN2), std::move(L1),
            std::move(L2), para);
    }
    // Decoder
    const auto Decodername = "decoder..npz";
    tile_mat WD(weights.load_mat<int>("row_ptr", Decodername),
                weights.load_mat<int>("column", Decodername),
                vec_F2H(weights.load_mat<float>("data", Decodername)),
                para->emdim / 16);
    culib::CUDA_ptr<half> BD(
        vec_F2H(weights.load_mat<float>("B", Decodername)));
    auto LD = std::make_unique<Linear<mat_t>>(para->emdim, TEXT_size, WD, BD,
                                              para->seq_len);

    // Inference
    model[0]->forward(OUT1.get(), IN.get());
    model[1]->forward(OUT2.get(), OUT1.get());
    LD->forward(OUT.get(), OUT2.get());
    cudaDeviceSynchronize();
    std::vector<half> h_out(OUT.size);
    OUT.dump(h_out.data());
    std::cout << __half2float(h_out[0]) << std::endl;
}

CUDA_ptr<half> random_input(int nrow, int ncol) {
    const auto size = nrow * ncol;
    std::vector<half> h_data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1, 1);
    std::generate_n(std::back_inserter(h_data), size,
                    [&]() { return __float2half_rn(dis(gen)); });
    return CUDA_ptr<half>(h_data);
}

std::vector<half> vec_F2H(const std::vector<float> &arr) {
    std::vector<half> res(arr.size());
    std::transform(arr.begin(), arr.end(), res.begin(),
                   [](float f) { return __float2half_rn(f); });
    return res;
}