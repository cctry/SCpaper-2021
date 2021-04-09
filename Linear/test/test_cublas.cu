#include "../Linear.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <wtime.h>
using namespace culib;

int main() {
    constexpr int num = 128;
    constexpr int in_size = 1024;
    constexpr int out_size = 1024 * 3;

    CUDA_ptr<half> W(out_size * in_size, half_one); // weight
    CUDA_ptr<half> IN(num * in_size, half_one);     // input
    CUDA_ptr<half> OUT(out_size * num);             // output
    CUDA_ptr<half> bias(out_size, __float2half_rn(0.5));

    Linear<CUDA_ptr<half>> linear(in_size, out_size, std::move(W), bias.get(),
                                  num);
    linear.forward(OUT.get(), IN.get());
    std::vector<half> res(out_size * num);
    OUT.dump(res.data());

    auto time = wtime(
        10,
        [&]() {
            linear.forward(OUT.get(), IN.get());
            cudaDeviceSynchronize();
        },
        [&]() { OUT.clear(); });
    OUT.clear();
    std::cout << "Time: " << time << std::endl;
}