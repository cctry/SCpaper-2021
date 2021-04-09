#pragma once
#include <CUDA_ptr.hpp>
#include <memory>
#include <vector>
class mat1x16 {
    template <typename T> using weight_ptr = culib::CUDA_ptr<T>;

  public:
    const int npart;
    const int nrow;
    const int ncol;
    const int ntile;
    weight_ptr<int> tile_idx; // nrow/16 * (npart + 1)
    weight_ptr<int> row_idx;  // ntile
    weight_ptr<half> data;    // ntile * 16
    mat1x16(int _nrow, int _ncol, const std::vector<int> &_tile_idx,
            const std::vector<int> &_row_idx, const std::vector<half> &_data)
        : nrow(_nrow), ncol(_ncol), npart(_ncol / 16), ntile(_row_idx.size()),
          row_idx(_row_idx), tile_idx(_tile_idx), data(_data) {}
    static mat1x16 gen_dense_mat(int, int);
    static mat1x16 gen_sparse_mat(int, int, float);
};
