#include "../../Model.h"
#include "../Linear_VO.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
using namespace culib;

int main() {
    auto para =
        std::make_shared<Model_t>(Model_t{1024, 1024, 128, 16, 4096, 1024});
    CUDA_ptr<half> weight(para->emdim * para->emdim * para->nhead,
                          __float2half(0.2));
    CUDA_ptr<half> bias(para->nhead * para->emdim, __float2half_rn(0.2));
    CUDA_ptr<half> input(para->seq_len * para->emdim, half_one);
    CUDA_ptr<half> output(para->nhead * para->seq_len * para->emdim);
    Linear_VO layer(weight, bias, para);
    layer.forward(output.get(), input.get());
    std::vector<half> h_res(output.size);
    cudaChk(cudaDeviceSynchronize());
    output.dump(h_res.data());
    printf("%f\n", __half2float(h_res[0]));
}