#include <Attention/Attention.h>
#include <CUDA_ptr.hpp>
#include <Linear/Linear.h>
#include <Model.h>
#include <bits/stdc++.h>
#include <wtime.h>
using namespace culib;
using w_mat_t = base_mat;
using mat_v_t = w_mat_t;

template <typename _attn_T> void test() {
    using attn_config =
        Attention_config<w_mat_t, w_mat_t, mat_v_t, w_mat_t, _attn_T>;
    constexpr float sparsity = 0.6;
    std::vector<double> res;
    for (int seq_len = 16; seq_len <= 512; seq_len += 16) {
        auto para = std::make_shared<Model_t>(
            Model_t{768, 768, seq_len, 12, 4096, 768});
        CUDA_ptr<half> IN(para->emdim * para->seq_len, __float2half_rn(0.2));
        CUDA_ptr<half> d_OUT(para->emdim * para->seq_len);
        auto LQ = gen_sparse_linear<w_mat_t>(para->kdim, para->emdim,
                                             para->seq_len, sparsity);
        auto LK = gen_sparse_linear<w_mat_t>(para->kdim, para->emdim,
                                             para->seq_len, sparsity);
        auto LV = gen_sparse_linear<mat_v_t>(para->vdim, para->emdim,
                                             para->seq_len, sparsity);
        auto LO = gen_sparse_linear<w_mat_t>(para->emdim, para->vdim,
                                             para->seq_len, sparsity);
        auto attn = std::make_unique<Attention<attn_config>>(
            std::move(LQ), std::move(LK), std::move(LV), std::move(LO), para);
        attn->forward(d_OUT.get(), IN.get(), IN.get(), IN.get());
        cudaChk(cudaDeviceSynchronize());
        auto time = wtime(
            10,
            [&]() {
                attn->forward(d_OUT.get(), IN.get(), IN.get(), IN.get());
                cudaDeviceSynchronize();
            },
            []() {});
        d_OUT.clear();
        res.push_back(time);
    }
    std::cout << typeid(_attn_T).name() << std::endl;
    for (auto i : res) {
        printf("%lf\n", i);
    }
}

int main(int ac, char **av) {
    test<OTF_attn_full>();
    test<OTF_attn_sharedQK>();
}
