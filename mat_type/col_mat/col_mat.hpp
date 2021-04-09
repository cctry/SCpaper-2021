#pragma once
#include <CUDA_ptr.hpp>
#include <IO.hpp>
#include <cstdint>
#include <memory>
#include <utils.h>
#include <vector>

// cols is row-major
class col_mat {

  public:
    int nrow;
    int ncol;
    int nnz_col;
    int ldm;
    std::unique_ptr<culib::CUDA_ptr<int>> head_ptr;
    std::unique_ptr<culib::CUDA_ptr<int>> col_id;
    std::unique_ptr<culib::CUDA_ptr<half>> cols;
    void set_metadata(const culib::CUDA_ptr<int> &_col_id, int _nnz_col,
                      int nhead, int full_dim);
    col_mat(const std::string &filename) {
        std::ifstream f(filename);
        auto meta = load_vec<int>(f, 3);
        nrow = meta[0];
        ncol = meta[1];
        auto len_id = meta[2];
        col_id =
            std::make_unique<culib::CUDA_ptr<int>>(load_vec<int>(f, len_id));
        auto cols_float =
            culib::CUDA_ptr<float>(load_vec<float>(f, nrow * len_id));
        cols = std::make_unique<culib::CUDA_ptr<half>>(nrow * len_id);
        culib::util::to_half_devptr(cols->get(), cols_float.get(),
                                    nrow * len_id);
        f.close();
    }
    col_mat(const culib::CUDA_ptr<int> &_col_id,
            const culib::CUDA_ptr<half> &_cols, int _nrow, int _ncol,
            int _nnz_col)
        : nrow(_nrow), ncol(_ncol), nnz_col(_nnz_col) {
        col_id = std::make_unique<culib::CUDA_ptr<int>>(_col_id);
        cols = std::make_unique<culib::CUDA_ptr<half>>(_cols);
        ldm = cols->size / nrow;
    }
    col_mat(culib::CUDA_ptr<int> &&_col_id, culib::CUDA_ptr<half> &&_cols,
            const int _nrow, const int _ncol, int _nnz_col)
        : nrow(_nrow), ncol(_ncol), nnz_col(_nnz_col) {
        col_id = std::make_unique<culib::CUDA_ptr<int>>(std::move(_col_id));
        cols = std::make_unique<culib::CUDA_ptr<half>>(std::move(_cols));
        ldm = cols->size / nrow;
    }
    col_mat(const col_mat &mat) {
        col_id = std::make_unique<culib::CUDA_ptr<int>>(*mat.col_id.get());
        cols = std::make_unique<culib::CUDA_ptr<half>>(*mat.cols.get());
        ldm = cols->size / mat.nrow;
    }
    col_mat(col_mat &&mat) {
        col_id = std::make_unique<culib::CUDA_ptr<int>>(
            std::move(*mat.col_id.get()));
        cols =
            std::make_unique<culib::CUDA_ptr<half>>(std::move(*mat.cols.get()));
        ldm = cols->size / mat.nrow;
    }
    col_mat(int _nrow, int _ncol, int _nn_col)
        : nrow(_nrow), ncol(_ncol), nnz_col(_nn_col) {
        col_id = std::make_unique<culib::CUDA_ptr<int>>(_nn_col);
        cols = std::make_unique<culib::CUDA_ptr<half>>(_nrow * _nn_col);
        ldm = cols->size / nrow;
    }
    col_mat *get() { return this; }
    static col_mat gen_dense_mat(int nrow, int ncol);
    static col_mat gen_sparse_mat(int nrow, int ncol, float sparsity);
};
