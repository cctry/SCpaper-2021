#include "col_mat.hpp"
#include <algorithm>
#include <numeric>
#include <vector>

template <int N> int roundTo(int n) {
    auto r = n % N;
    if (r == 0)
        return n;
    else
        return n + 16 - r;
}

col_mat col_mat::gen_dense_mat(int nrow, int ncol) {
    culib::CUDA_ptr<half> cols(ncol * nrow, __float2half_rn(1.0));
    std::vector<int> h_col_id(ncol);
    std::iota(h_col_id.begin(), h_col_id.end(), 0);
    culib::CUDA_ptr<int> col_id(h_col_id);
    return col_mat(std::move(col_id), std::move(cols), nrow, ncol, ncol);
}

col_mat col_mat::gen_sparse_mat(int nrow, int ncol, float sparsity) {
    int nn_col = ncol * (1.0 - sparsity);
    nn_col = 16 * ceil(nn_col / 16);
    culib::CUDA_ptr<half> cols(nrow * nn_col, __float2half_rn(0.5));
    std::vector<int> h_col_id(ncol);
    std::iota(h_col_id.begin(), h_col_id.end(), 0);
    std::random_shuffle(h_col_id.begin(), h_col_id.end());
    h_col_id.erase(h_col_id.begin() + nn_col, h_col_id.end());
    culib::CUDA_ptr<int> col_id(h_col_id.data(), nn_col);
    return col_mat(std::move(col_id), std::move(cols), nrow, ncol, nn_col);
}

void col_mat::set_metadata(const culib::CUDA_ptr<int> &_col_id, int _nnz_col,
                           int nhead, int full_dim) {
    auto old = col_id.release();
    auto head_dim = full_dim / nhead;
    if (old)
        delete old;
    std::vector<int> h_col_id(_col_id.size);
    _col_id.dump(h_col_id.data());
    std::vector<int> _head_id = {0};
    for (int head = 0; head < nhead; head++) {
        auto range_high = (head + 1) * (head_dim);
        int i;
        for (i = _head_id.back(); h_col_id[i] < range_high && i < _nnz_col; i++)
            ;
        int end = i;
        _head_id.push_back(end);
    }
    assert(_head_id.size() == nhead + 1);
    if (_head_id.back() != _nnz_col)
        printf("%d\t%d\n", _head_id.back(), _nnz_col);
    assert(_head_id.back() == _nnz_col);
    nnz_col = _nnz_col;
    head_ptr = std::make_unique<culib::CUDA_ptr<int>>(_head_id);
    col_id = std::make_unique<culib::CUDA_ptr<int>>(_col_id);
}
