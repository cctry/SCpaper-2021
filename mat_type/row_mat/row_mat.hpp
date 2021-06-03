#pragma once
#include <CUDA_ptr.hpp>
#include <IO.hpp>
#include <cstdint>
#include <memory>
#include <utils.h>
#include <vector>

class row_mat {
  public:
    int nrow;
    int ncol;
    std::unique_ptr<culib::CUDA_ptr<int>> row_id;
    std::unique_ptr<culib::CUDA_ptr<half>> rows;
    row_mat(const std::string &filename) {
        std::ifstream f(filename);
        auto meta = load_vec<int>(f, 3);
        nrow = meta[0];
        ncol = meta[1];
        auto len_id = meta[2];
        row_id = std::make_unique<culib::CUDA_ptr<int>>(
            load_vec<int>(f, len_id));
        auto rows_float =
            culib::CUDA_ptr<float>(load_vec<float>(f, ncol * len_id));
        rows = std::make_unique<culib::CUDA_ptr<half>>(ncol * len_id);
        culib::util::to_half_devptr(rows->get(), rows_float.get(),
                                    ncol * len_id);
        f.close();
    }
    row_mat(const culib::CUDA_ptr<int> &_row_id,
            const culib::CUDA_ptr<half> &_rows, const int _nrow,
            const int _ncol)
        : nrow(_nrow), ncol(_ncol) {
        row_id = std::make_unique<culib::CUDA_ptr<int>>(_row_id);
        rows = std::make_unique<culib::CUDA_ptr<half>>(_rows);
    }
    row_mat(culib::CUDA_ptr<int> &&_row_id, culib::CUDA_ptr<half> &&_rows,
            const int _nrow, const int _ncol)
        : nrow(_nrow), ncol(_ncol) {
        row_id =
            std::make_unique<culib::CUDA_ptr<int>>(std::move(_row_id));
        rows = std::make_unique<culib::CUDA_ptr<half>>(std::move(_rows));
    }
    row_mat(const row_mat &mat) : nrow(mat.nrow), ncol(mat.ncol) {
        row_id = std::make_unique<culib::CUDA_ptr<int>>(*mat.row_id.get());
        rows = std::make_unique<culib::CUDA_ptr<half>>(*mat.rows.get());
    }
    row_mat(row_mat &&mat) : nrow(mat.nrow), ncol(mat.ncol) {
        row_id = std::make_unique<culib::CUDA_ptr<int>>(
            std::move(*mat.row_id.get()));
        rows =
            std::make_unique<culib::CUDA_ptr<half>>(std::move(*mat.rows.get()));
    }
    static row_mat gen_dense_mat(int nrow, int ncol);
    static row_mat gen_sparse_mat(int nrow, int ncol, float sparsity);
};
