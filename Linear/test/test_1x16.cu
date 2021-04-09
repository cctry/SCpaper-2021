#include "../Linear.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <wtime.h>
using namespace culib;

int main(int ac, char **av) {
    constexpr int seq_len = 128;
    constexpr int in_size = 1024;
    constexpr int out_size = 1024 * 3;
    CUDA_ptr<half> IN(seq_len * in_size, half_one);
    CUDA_ptr<float> OUT(out_size * seq_len);
    // CUDA_ptr<half> OUT(out_size * seq_len);

    // auto layer = gen_sparse_linear<mat1x16>(out_size, in_size, seq_len, 0);
    // layer->forward(OUT.get(), IN.get());
    // cudaChk(cudaDeviceSynchronize());

    // CUDA_ptr<float> OUT_f(out_size * seq_len);
    // layer->forward(OUT_f.get(), IN.get());
    // cudaChk(cudaDeviceSynchronize());

    for (int i = 0; i < 20; i++) {
        float sp = i * 0.05;
        auto layer = gen_sparse_linear<mat1x16>(out_size, in_size, seq_len, sp);
        layer->forward(OUT.get(), IN.get());
        OUT.clear();
        cudaChk(cudaDeviceSynchronize());
        double time = wtime(
            10,
            [&]() {
                layer->forward(OUT.get(), IN.get());
                cudaChk(cudaDeviceSynchronize());
            },
            [&]() { cudaChk(cudaDeviceSynchronize()); });
        printf("%f, %lf\n", sp, time);
    }
}