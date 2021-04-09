#include "../Attention/Attention.h"
#include "../Encoder/Encoder.h"
#include "../Linear/Linear.h"
#include "../Model.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <wtime.h>
using namespace culib;

int main(int ac, char **av) {
    float sparsity = std::atof(av[1]);
    constexpr int num_encoders = 1;
    using w_mat_t = tile_mat;
    using AL_t = tile_mat;
    using mat_v_t = w_mat_t;
    using attn_t = OTF_attn_full;
    auto para =
        std::make_shared<Model_t>(Model_t{768, 768, 128, 12, 768 * 4, 768});
    using attn_config = Attention_config<w_mat_t, w_mat_t, mat_v_t, w_mat_t,
                                         attn_t, true, false>;
    using encoder_config = Encoder_config<attn_config, AL_t, AL_t, GELU_OP>;

    std::array<std::unique_ptr<Encoder<encoder_config>>, num_encoders> model;
    for (int i = 0; i < model.size(); i++) {
        auto LQKV = gen_sparse_linear<w_mat_t>(
            para->kdim * 2 + para->vdim, para->emdim, para->seq_len, sparsity);
        auto LO = gen_sparse_linear<w_mat_t>(para->emdim, para->vdim,
                                             para->seq_len, sparsity);
        auto attn = std::make_unique<Attention<attn_config>>(
            std::move(LQKV), std::move(LO), para);

        auto LN1 = gen_skLN(para->emdim);
        auto LN2 = gen_skLN(para->emdim);

        auto L1 = gen_sparse_linear<AL_t>(para->dimFF, para->emdim,
                                          para->seq_len, sparsity);
        auto L2 = gen_sparse_linear<AL_t>(para->emdim, para->dimFF,
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
            for (int i = 0; i < model.size(); i++) {
                model[i]->forward(d_OUT.get(), IN.get());
            }
        },
        []() { cudaDeviceSynchronize(); });
    std::cout << "Time: " << time << " us\n";
    std::vector<half> OUT(d_OUT.size);
    d_OUT.dump(OUT.data());
    std::cout << __half2float(OUT[0]) << std::endl;
}
