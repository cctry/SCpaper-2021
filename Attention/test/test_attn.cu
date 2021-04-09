#include "../../Linear/Linear.h"
#include "../Attention.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
using namespace culib;
int main() {
    using w_mat_t = CUDA_ptr<half>;
    using attn_config =
        Attention_config<w_mat_t, w_mat_t, w_mat_t, w_mat_t, OTF_attn>;
    auto para = std::make_shared<Model_t>(Model_t{768, 768, 384, 12, 3072, 768});

    auto LQ = gen_dense_linear<w_mat_t>(para->kdim, para->emdim, para->seq_len);
    auto LK = gen_dense_linear<w_mat_t>(para->kdim, para->emdim, para->seq_len);
    auto LV = gen_dense_linear<w_mat_t>(para->vdim, para->emdim, para->seq_len);
    auto LO = gen_dense_linear<w_mat_t>(para->emdim, para->vdim, para->seq_len);
    auto attn = std::make_unique<Attention<attn_config>>(
        std::move(LQ), std::move(LK), std::move(LV), std::move(LO), para);

    CUDA_ptr<half> IN(para->emdim * para->seq_len);
    CUDA_ptr<half> d_OUT(para->emdim * para->seq_len);
    attn->forward(d_OUT.get(), IN.get(), IN.get(), IN.get());
    cudaDeviceSynchronize();
    std::vector<half> OUT(para->emdim * para->seq_len);
    d_OUT.dump(OUT.data());
    std::cout << __half2float(OUT[0]) << std::endl;
}