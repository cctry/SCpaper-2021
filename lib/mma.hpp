#pragma once
#include <mma.h>
using namespace nvcuda;
namespace culib {
namespace mma {
template <int M, int N, int K, typename T = half> struct mma_t {
    template <typename layout>
    using a_t = wmma::fragment<wmma::matrix_a, M, N, K, T, layout>;
    template <typename layout>
    using b_t = wmma::fragment<wmma::matrix_b, M, N, K, T, layout>;
    template <typename V>
    using c_t = wmma::fragment<wmma::accumulator, M, N, K, V>;
};
} // namespace mma
} // namespace culib