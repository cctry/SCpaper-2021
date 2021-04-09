#include "../Linear.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <wtime.h>
using namespace culib;

int main() {
    constexpr int num = 128;
    constexpr int in_size = 1024;
    constexpr int out_size = 1024 * 3;
    constexpr int num_row = out_size;
    CUDA_ptr<half> IN(num * in_size, half_one);
    CUDA_ptr<half> OUT(out_size * num);
    auto layer = gen_sparse_linear<row_mat>(out_size, in_size, num, 0.05);
    layer->forward(OUT.get(), IN.get());
    OUT.clear();
    double time = wtime(
        10,
        [&]() {
            layer->forward(OUT.get(), IN.get());
            cudaChk(cudaDeviceSynchronize());
        },
        [&]() { OUT.clear(); });
    std::cout << time << std::endl;
}