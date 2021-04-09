#include "../../Linear/Linear.h"
#include "../../Model.h"
#include "../Fused_attention.h"
#include "../Linear_VO.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <col_mat/col_mat.hpp>
#include <wtime.h>

using namespace culib;
using w_mat_t = tile_mat;
#define TIME

void test(CUDA_ptr<half> &output, CUDA_ptr<half> &input,
          std::shared_ptr<Model_t> para) {
    using attn_config = Attention_config<w_mat_t, w_mat_t, w_mat_t, w_mat_t,
                                         NOTF_fused_prune_attn>;
    CUDA_ptr<half> biasO(para->emdim);
    float sparsity_base = 0.05;
    for (int i = 0; i < 20; i++) {
        float sparsity = i * sparsity_base;
        auto LQ =
            gen_dense_linear<w_mat_t>(para->kdim, para->emdim, para->seq_len);
        auto LK =
            gen_dense_linear<w_mat_t>(para->kdim, para->emdim, para->seq_len);
        // build LVO
        auto weight = col_mat::gen_sparse_mat(para->nhead * para->emdim,
                                              para->emdim, sparsity);
        CUDA_ptr<half> bias(para->nhead * weight.col_id->size,
                            __float2half_rn(0.2));
        auto LVO = std::make_unique<Linear_VO_prune>(weight, bias, para);
        FusedAttention<attn_config> attn(std::move(LQ), std::move(LK),
                                         std::move(LVO), biasO, para);
        attn.forward(output.get(), input.get(), input.get(), input.get());
        cudaChk(cudaDeviceSynchronize());
#ifdef TIME
        double time = wtime(
            10,
            [&]() {
                attn.forward(output.get(), input.get(), input.get(),
                             input.get());
                cudaChk(cudaDeviceSynchronize());
            },
            [&]() {});
        std::cout << time << std::endl;
#endif
    }
    std::vector<half> h_res(para->emdim * para->seq_len);
    output.dump(h_res.data());
    // printf("%f\n", __half2float(h_res[0]));
}

void prof_one(CUDA_ptr<half> &output, CUDA_ptr<half> &input,
              std::shared_ptr<Model_t> para) {
    using attn_config = Attention_config<w_mat_t, w_mat_t, w_mat_t, w_mat_t,
                                         NOTF_fused_prune_attn>;
    CUDA_ptr<half> biasO(para->emdim);
    float sparsity = 0.15;
    auto LQ = gen_sparse_linear<w_mat_t>(para->kdim, para->emdim, para->seq_len,
                                         sparsity);
    auto LK = gen_sparse_linear<w_mat_t>(para->kdim, para->emdim, para->seq_len,
                                         sparsity);
    // build LVO
    auto weight = col_mat::gen_sparse_mat(para->nhead * para->emdim,
                                          para->emdim, sparsity);
    CUDA_ptr<half> bias(para->nhead * weight.col_id->size,
                        __float2half_rn(0.2));
    auto LVO = std::make_unique<Linear_VO_prune>(weight, bias, para);
    FusedAttention<attn_config> attn(std::move(LQ), std::move(LK),
                                     std::move(LVO), biasO, para);
    attn.forward(output.get(), input.get(), input.get(), input.get());
    cudaChk(cudaDeviceSynchronize());
}

int main() {
    auto para =
        std::make_shared<Model_t>(Model_t{1024, 1024, 128, 4, 4096, 1024});
    CUDA_ptr<half> input(para->seq_len * para->emdim, half_one);
    CUDA_ptr<half> output(para->seq_len * para->emdim);
    test(output, input, para);
    // prof_one(output,input, para);
}