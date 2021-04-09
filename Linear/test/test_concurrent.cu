#include "../Linear.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <constexpr_loop.hpp>
#include <type_list.hpp>
#include <typeinfo>
#include <utils.h>
#include <wtime.h>
using namespace culib;
int in_size;
int out_size;
int seq_len;
using res_t = std::pair<float, double>;
template <typename mat_t> void test(int out_size, int in_size, int seq_len) {
    constexpr float sparsity_base = 0.05;
    std::vector<res_t> res;
    CUDA_ptr<half> IN(seq_len * in_size, half_one);
    CUDA_ptr<half> OUT1(out_size * seq_len);
    CUDA_ptr<half> OUT2(out_size * seq_len);
    CUDA_ptr<half> OUT3(out_size * seq_len);
    cudaStream_t streams[3];
    for (size_t i = 0; i < 3; i++) {
        cudaStreamCreate(streams + i);
    }
    for (size_t i = 0; i < 20; i++) {
        float sparsity = sparsity_base * i;
        auto layer1 =
            gen_sparse_linear<mat_t>(out_size, in_size, seq_len, sparsity);
        auto layer2 =
            gen_sparse_linear<mat_t>(out_size, in_size, seq_len, sparsity);
        auto layer3 =
            gen_sparse_linear<mat_t>(out_size, in_size, seq_len, sparsity);
        layer1->forward(OUT1.get(), IN.get());
        cudaChk(cudaDeviceSynchronize());
        double time = wtime(
            10,
            [&]() {
                layer1->forward(OUT1.get(), IN.get(), streams[0]);
                layer2->forward(OUT2.get(), IN.get(), streams[1]);
                layer3->forward(OUT3.get(), IN.get(), streams[2]);
                culib::util::sync_streams(streams, 3);
            },
            [&]() { cudaChk(cudaDeviceSynchronize()); });
        res.emplace_back(sparsity, time);
    }
    for (auto &e : res) {
        printf(" %lf\n", e.second);
    }
    for (size_t i = 0; i < 3; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

using list = type_list::list<base_mat, row_mat, col_mat, tile_mat>;

template <std::size_t N, typename = void> struct Func {
    void operator()() {
        using mat_t = typename type_list::get_type<N, list>::type;
        std::cout << typeid(mat_t).name() << std::endl;
        test<mat_t>(::in_size, ::out_size, ::seq_len);
    }
};

template <class mat_t>
void prof_one(const int out_size, const int in_size, const int seq_len, float sparsity) {
    CUDA_ptr<half> IN(seq_len * in_size, half_one);
    CUDA_ptr<half> OUT1(out_size * seq_len);
    CUDA_ptr<half> OUT2(out_size * seq_len);
    CUDA_ptr<half> OUT3(out_size * seq_len);
    cudaStream_t streams[3];
    auto layer1 =
        gen_sparse_linear<mat_t>(out_size, in_size, seq_len, sparsity);
    auto layer2 =
        gen_sparse_linear<mat_t>(out_size, in_size, seq_len, sparsity);
    auto layer3 =
        gen_sparse_linear<mat_t>(out_size, in_size, seq_len, sparsity);
    for (size_t i = 0; i < 3; i++) {
        cudaStreamCreate(streams + i);
    }
    layer1->forward(OUT1.get(), IN.get(), streams[0]);
    layer2->forward(OUT2.get(), IN.get(), streams[1]);
    layer3->forward(OUT3.get(), IN.get(), streams[2]);
    cudaChk(cudaDeviceSynchronize());
    for (size_t i = 0; i < 3; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

int main(int ac, char **av) {
    if (ac < 4) {
        puts("./a.out in_size out_size seq_len");
        return -1;
    }
    ::in_size = std::atoi(av[1]);
    ::out_size = std::atoi(av[2]);
    ::seq_len = std::atoi(av[3]);
    float sparsity = std::atof(av[4]);
    auto loop = make_loop<Func, void, list::size>();
    loop.run();
    // prof_one<tile_mat>(::in_size, ::out_size, ::seq_len, sparsity);
    // prof_one<base_mat>(::in_size, ::out_size, ::seq_len);
}
