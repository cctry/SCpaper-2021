#include "row_mat.hpp"
#include <algorithm>
#include <numeric>
#include <vector>
row_mat row_mat::gen_dense_mat(int nrow, int ncol) {
    culib::CUDA_ptr<half> rows(ncol * nrow, __float2half_rn(1.0));
    std::vector<int> h_row_id(nrow);
    std::iota(h_row_id.begin(), h_row_id.end(), 0);
    culib::CUDA_ptr<int> row_id(h_row_id);
    row_mat W(std::move(row_id), std::move(rows), nrow, ncol);
    return W;
}

row_mat row_mat::gen_sparse_mat(int nrow, int ncol, float sparsity) {
    int nn_row = nrow * (1.0 - sparsity);
    nn_row = ((nn_row / 16) + ((nn_row % 16) != 0)) * 16; // pad to 16
    culib::CUDA_ptr<half> rows(ncol * nn_row, __float2half_rn(1.0));
    std::vector<int> h_row_id(nrow);
    std::iota(h_row_id.begin(), h_row_id.end(), 0);
    std::random_shuffle(h_row_id.begin(), h_row_id.end());
    h_row_id.erase(h_row_id.begin() + nn_row, h_row_id.end());
    std::sort(h_row_id.begin(), h_row_id.end());
    culib::CUDA_ptr<int> row_id(h_row_id.data(), nn_row);
    return row_mat(std::move(row_id), std::move(rows), nrow, ncol);
}
