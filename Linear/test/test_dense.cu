#include "../Linear.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <constexpr_loop.hpp>
#include <type_list.hpp>
#include <typeinfo>
#include <utils.h>
#include <wtime.h>
using namespace culib;

template <typename mat_t> void test(int out_size, int in_size, int seq_len) {
    auto layer = gen_dense_linear<mat_t>(out_size, in_size, seq_len);
    CUDA_ptr<half> IN(seq_len * in_size, half_one);
    CUDA_ptr<half> OUT(out_size * seq_len);
    layer->forward(OUT.get(), IN.get());
    OUT.clear();
    double time = wtime_new(
        10,
        [&]() {
            layer->forward(OUT.get(), IN.get());
        },
        [&]() { cudaChk(cudaDeviceSynchronize()); });
    std::cout << typeid(mat_t).name() << ":\t" << time << std::endl;
}

int in_size;
int out_size;
int seq_len;

// using list = type_list::list<base_mat, row_mat, col_mat, csr_mat, tile_mat>;
using list = type_list::list<base_mat>;
template <std::size_t N, typename = void> struct Func {
    void operator()() {
        using mat_t = typename type_list::get_type<N, list>::type;
        test<mat_t>(::in_size, ::out_size, ::seq_len);
    }
};

int main(int ac, char **av) {
    ::in_size = std::atoi(av[1]);
    ::out_size = std::atoi(av[2]);
    ::seq_len = std::atoi(av[3]);

    auto loop = make_loop<Func, void, list::size>();
    loop.run();
}