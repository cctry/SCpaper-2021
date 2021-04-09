#include "../Attention.h"
#include "../kernels.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <wtime.h>
using namespace culib;
template <typename kernel_t> void test(int seq_len) {
    auto para =
        std::make_shared<Model_t>(Model_t{768, 768, seq_len, 12, 768 * 4, 768});
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
    double time = wtime_new(
        100,
        [&]() {
            Multihead_atttion<kernel_t>(&matQ, &matK, &matV, &matQK, &mask,
                                        &matZ, para);
        },
        []() { cudaDeviceSynchronize(); });

    std::cout << typeid(kernel_t{}).name() << "\n" << time << std::endl;
    std::vector<half> OUT(para->vdim * para->seq_len);
    matZ.dump(OUT.data());
    // std::cout << __half2float(OUT[0]) << std::endl;

    // Multihead_atttion<OTF_attn_full>(&matQ, &matK, &matV, &matQK, &mask,
    // &matZ,
    //                                  para);
    // std::vector<half> OUT2(para->vdim * para->seq_len);
    // matZ.dump(OUT2.data());

    // for (int i = 0; i < OUT.size(); i++) {
    //     auto v1 = __half2float(OUT[i]);
    //     auto v2 = __half2float(OUT2[i]);
    //     if (abs(v1 - v2) > 0.5) {
    //         printf("id:%i\t%f\t%f\n", i, v1, v2);
    //     }
    // }
}

int main(int ac, char **av) {

    // for (int i = 16; i <= 512; i += 16) {
    //     test<OTF_attn_full>(i);
    // }
    // puts("");
    // for (int i = 16; i <= 512; i += 16) {
    //     test<OTF_attn_full_mixed>(i);
    // }
    int seq_len = std::atoi(av[1]);
    test<OTF_attn_sharedQK>(seq_len);
    test<OTF_attn_full>(seq_len);
}