#include "../Linear.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <mma.hpp>
#include <wtime.h>
using namespace culib;

int main(int ac, char **av) {
    auto in_size = std::atoi(av[1]);
    auto out_size = std::atoi(av[2]);
    auto seq_len = std::atoi(av[3]);
    float sparsity = std::atof(av[4]);

    CUDA_ptr<half> IN(seq_len * in_size, half_one); // input
    CUDA_ptr<half> OUT(out_size * seq_len);         // output
    using mat_t = tile_mat;
    auto W = mat_t::gen_sparse_mat(in_size, out_size, sparsity);
    std::vector<half> bias(out_size, __float2half_rn(0.5));
    auto layer = std::make_unique<Linear<mat_t>>(in_size, out_size, std::move(W),
                                               bias.data(), seq_len);

    layer->forward(OUT.get(), IN.get());
    OUT.clear();
    double time = wtime(
        100,
        [&]() {
            layer->forward(OUT.get(), IN.get());
            cudaChk(cudaDeviceSynchronize());
        },
        [&]() {});
    std::cout << typeid(mat_t).name() << ":\t" << time << std::endl;
    std::vector<half> res(out_size * seq_len);
    OUT.dump(res.data());
    std::cout << __half2float(res[128 * 1000]) << ",";
    puts("");
}