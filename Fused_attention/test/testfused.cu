#include "../../Linear/Linear.h"
#include "../../Model.h"
#include "../Fused_attention.h"
#include "../Linear_VO.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <wtime.h>
using namespace culib;
using w_mat_t = base_mat;
#define TIME

void test_normal(CUDA_ptr<half> &output, CUDA_ptr<half> &input,
                 std::shared_ptr<Model_t> para);

void test_fused_NOTF(CUDA_ptr<half> &output, CUDA_ptr<half> &input,
                     std::shared_ptr<Model_t> para);
int main(int ac, char **av) {
    int nhead = std::atoi(av[1]);
   int d_model = std::atoi(av[2]);
    auto para = std::make_shared<Model_t>(
        Model_t{d_model, d_model, 128, nhead, d_model * 4, d_model});
    if (para->kdim % nhead != 0 || para->kdim / nhead % 16 != 0) {
        puts("nhead error");
        return -1;
    }
    CUDA_ptr<half> input(para->seq_len * para->emdim, half_one);
    CUDA_ptr<half> output(para->seq_len * para->emdim);

    test_normal(output, input, para);
    test_fused_NOTF(output, input, para);
}

void test_fused_NOTF(CUDA_ptr<half> &output, CUDA_ptr<half> &input,
                     std::shared_ptr<Model_t> para) {
    float sparsity = 0.8;
    using attn_config = Attention_config<w_mat_t, w_mat_t, w_mat_t, w_mat_t,
                                         NOTF_fused_prune_attn>;
    CUDA_ptr<half> biasO(para->emdim);
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
    std::vector<half> h_res(para->emdim * para->seq_len);
    cudaChk(cudaDeviceSynchronize());
#ifdef TIME
    double time = wtime_new(
        10,
        [&]() {
            attn.forward(output.get(), input.get(), input.get(), input.get());
        },
        [&]() { cudaChk(cudaDeviceSynchronize()); });
    std::cout << "Time: " << time << " us" << std::endl;
#endif
    output.dump(h_res.data());
    printf("NOTF_fused_attn: %f\n", __half2float(h_res[0]));
}

void test_normal(CUDA_ptr<half> &output, CUDA_ptr<half> &input,
                 std::shared_ptr<Model_t> para) {
    float sparsity = 0.5;
    using attn_config =
        Attention_config<w_mat_t, w_mat_t, w_mat_t, w_mat_t, OTF_attn_full>;
    auto LQ = gen_sparse_linear<w_mat_t>(para->kdim, para->emdim, para->seq_len,
                                         sparsity);
    auto LK = gen_sparse_linear<w_mat_t>(para->kdim, para->emdim, para->seq_len,
                                         sparsity);
    auto LV = gen_sparse_linear<w_mat_t>(para->vdim, para->emdim, para->seq_len,
                                         sparsity);
    auto LO = gen_sparse_linear<w_mat_t>(para->emdim, para->vdim, para->seq_len,
                                         sparsity);
    auto attn = std::make_unique<Attention<attn_config>>(
        std::move(LQ), std::move(LK), std::move(LV), std::move(LO), para);
    attn->forward(output.get(), input.get(), input.get(), input.get());
    cudaChk(cudaDeviceSynchronize());
#ifdef TIME
    double time = wtime_new(
        10,
        [&]() {
            attn->forward(output.get(), input.get(), input.get(), input.get());
        },
        [&]() { cudaChk(cudaDeviceSynchronize()); });
    std::cout << "Time: " << time << " us" << std::endl;
#endif
    std::vector<half> h_res(para->emdim * para->seq_len);
    cudaChk(cudaDeviceSynchronize());
    output.dump(h_res.data());
    printf("OTF_attn_full: %f\n", __half2float(h_res[0]));
}