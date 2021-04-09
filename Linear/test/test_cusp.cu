#include "../Linear.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <csr_mat/csr_mat.h>
#include <wtime.h>
using namespace culib;

int main(int ac, char **av) {
    auto in_size = std::atoi(av[1]);
    auto out_size = std::atoi(av[2]);
    auto seq_len = std::atoi(av[3]);
    float sparsity = std::atof(av[4]);

    CUDA_ptr<half> IN(seq_len * in_size, half_one); // input
    CUDA_ptr<half> OUT(out_size * seq_len);         // output

    auto linear =
        gen_sparse_linear<csr_mat>(out_size, in_size, seq_len, sparsity);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    linear->forward(OUT.get(), IN.get(), stream);
    auto time = wtime_new(
        10, [&]() { linear->forward(OUT.get(), IN.get(), stream); },
        []() { cudaChk(cudaDeviceSynchronize()); });
    std::vector<half> res(out_size * seq_len);
    OUT.dump(res.data());
    std::cout << time << std::endl;
    // for (auto i : res) {
    //     std::cout << __half2float(i) << ",";
    // }
    // puts("");
}