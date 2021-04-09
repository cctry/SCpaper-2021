#include "../Attention.h"
#include "../kernels.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <wtime.h>
using namespace culib;
using kernel_t = OTF_attn_full;
auto para = std::make_shared<Model_t>(Model_t{1024, 1024, 128, 16, 4096, 1024});
void test(int seq_len) {
    half val = __float2half_rn(0.5);
    CUDA_ptr<half> matQ(para->seq_len * para->kdim, val),
        matK(para->seq_len * para->kdim, val),
        matV(para->seq_len * para->vdim, val),
        mask(para->seq_len * para->seq_len, val),
        matZ(para->seq_len * para->vdim),
        matQK(para->seq_len * para->seq_len * para->nhead);

    Multihead_atttion<kernel_t>(&matQ, &matK, &matV, &matQK, &mask, &matZ,
                                para);
    cudaDeviceSynchronize();
    auto time = wtime(
        10,
        [&]() {
            Multihead_atttion<kernel_t>(&matQ, &matK, &matV, &matQK, &mask,
                                        &matZ, para);
            cudaDeviceSynchronize();
        },
        []() {});
    std::cout << time << std::endl;
    std::vector<half> OUT(para->vdim * para->seq_len);
    matZ.dump(OUT.data());
}

int main() {
    for (int i = 16; i <= 512; i += 16) {
        test<OTF_attn_sharedQK>(i);
    }
    puts("");
    for (int i = 16; i <= 512; i += 16) {
        test<OTF_attn_full>(i);
    }
}