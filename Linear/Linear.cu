#include "Linear.h"
#include <algorithm>
#include <numeric>
#include <utils.h>
#include <vector>

template <>
std::unique_ptr<Linear<base_mat>> gen_dense_linear<base_mat>(int nrow, int ncol,
                                                             int size) {
    culib::CUDA_ptr<half> W(nrow * ncol, __float2half_rn(1.0));
    culib::CUDA_ptr<half> bias(nrow, __float2half_rn(0.5));
    auto res = std::make_unique<Linear<base_mat>>(ncol, nrow, std::move(W),
                                                  bias.get(), size);
    return std::move(res);
}

template <>
std::unique_ptr<Linear<base_mat>>
gen_sparse_linear<base_mat>(int nrow, int ncol, int size, float sparsity) {
    auto res = gen_dense_linear<base_mat>(nrow, ncol, size);
    return std::move(res);
}
