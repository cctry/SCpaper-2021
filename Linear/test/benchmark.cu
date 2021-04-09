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
    float sparsity_base = 0.05;
    std::vector<res_t> res;
    CUDA_ptr<half> IN(seq_len * in_size, half_one);
    CUDA_ptr<half> OUT(out_size * seq_len);
    for (size_t i = 0; i < 20; i++) {
        float sparsity = sparsity_base * i;
        auto layer =
            gen_sparse_linear<mat_t>(out_size, in_size, seq_len, sparsity);
        layer->forward(OUT.get(), IN.get());
        OUT.clear();
        double time = wtime_new(
            10, [&]() { layer->forward(OUT.get(), IN.get()); },
            [&]() { cudaChk(cudaDeviceSynchronize()); });
        res.emplace_back(sparsity, time);
    }
    for (auto &e : res) {
        printf("%lf\n", e.second);
    }
}

using list = type_list::list<base_mat, col_mat, row_mat,tile_mat>;

// using list = type_list::list<mat1x16>;

template <std::size_t N, typename = void> struct Func {
    void operator()() {
        using mat_t = typename type_list::get_type<N, list>::type;
        std::cout << typeid(mat_t).name() << std::endl;
        test<mat_t>(::in_size, ::out_size, ::seq_len);
    }
};

int main(int ac, char **av) {
    if (ac < 4) {
        puts("./a.out in_size out_size seq_len");
        return -1;
    }
    ::in_size = std::atoi(av[1]);
    ::out_size = std::atoi(av[2]);
    ::seq_len = std::atoi(av[3]);

    auto loop = make_loop<Func, void, list::size>();
    loop.run();
}
