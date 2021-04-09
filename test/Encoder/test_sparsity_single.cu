#include <Attention/Attention.h>
#include <Attention/Attention_single.h>
#include <CUDA_ptr.hpp>
#include <Encoder/Encoder.h>
#include <Linear/Linear.h>
#include <Model.h>
#include <bits/stdc++.h>
#include <wtime.h>
using namespace culib;
auto para = std::make_shared<Model_t>(Model_t{768, 768, 128, 12, 768 * 4, 768});

void test_single_kernel() {
    constexpr int num_encoders = 1;
    using w_mat_t = tile_mat;
    using attn_config =
        Attention_config<w_mat_t, w_mat_t, w_mat_t, w_mat_t, void, true, false>;
    using encoder_config =
        Encoder_config<attn_config, w_mat_t, w_mat_t, GELU_OP>;
    for (int i = 0; i < 20; i++) {
        float sparsity = 0.05 * i;
        std::array<std::unique_ptr<Encoder<encoder_config>>, num_encoders>
            model;
        for (int i = 0; i < model.size(); i++) {
            auto LQKV = gen_sparse_linear<w_mat_t>(para->kdim * 2 + para->vdim,
                                                   para->emdim, para->seq_len,
                                                   sparsity);
            auto LO = gen_sparse_linear<w_mat_t>(para->emdim, para->vdim,
                                                 para->seq_len, sparsity);
            auto attn = std::make_unique<Attention<attn_config>>(
                std::move(LQKV), std::move(LO), para);

            auto LN1 = gen_skLN(para->emdim);
            auto LN2 = gen_skLN(para->emdim);
            auto L1 = gen_sparse_linear<w_mat_t>(para->dimFF, para->emdim,
                                                 para->seq_len, sparsity);
            auto L2 = gen_sparse_linear<w_mat_t>(para->emdim, para->dimFF,
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

        double time = 0.0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i : culib::range(0, 100)) {
                for (int i = 0; i < model.size(); i++) {
                    model[i]->forward(d_OUT.get(), IN.get());
                }
        }
        cudaChk(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count() /
            100.0;

        std::cout << time << "\n";
        std::vector<half> OUT(d_OUT.size);
        d_OUT.dump(OUT.data());
        // std::cout << __half2float(OUT[0]) << std::endl;
    }
}

int main() { test_single_kernel(); }