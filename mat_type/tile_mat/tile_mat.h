#pragma once
#include <CUDA_ptr.hpp>
class tile_mat {
  public:
    static constexpr int tile_m = 16;
    static constexpr int tile_n = 16;
    static constexpr int tile_k = 16;
    static constexpr int tile_size = tile_m * tile_k;
    int blk_row_num;
    int blk_col_num;
    int num_blk;
    culib::CUDA_ptr<int> row_ptr;
    culib::CUDA_ptr<int> row_offset;
    culib::CUDA_ptr<half> data;
    tile_mat *dev_ptr;

    tile_mat(const int _blk_row_num, const int _blk_col_num, const int _num_blk,
             const int *_row_ptr, const int *_row_offset, const half *_data)
        : blk_row_num(_blk_row_num), blk_col_num(_blk_col_num),
          num_blk(_num_blk), row_ptr(_row_ptr, _blk_row_num + 1),
          row_offset(_row_offset, _num_blk), data(_num_blk * tile_size) {
        cudaChk(cudaMalloc(&dev_ptr, sizeof(tile_mat)));
        cudaChk(cudaMemcpy(dev_ptr, this, sizeof(tile_mat),
                           cudaMemcpyHostToDevice));
    }
    tile_mat(const culib::CUDA_ptr<int> &_row_ptr,
             const culib::CUDA_ptr<int> &_row_offset,
             const culib::CUDA_ptr<half> &_data, int _blk_col_num)
        : row_ptr(_row_ptr), row_offset(_row_offset), data(_data),
          blk_row_num(_row_ptr.size - 1), blk_col_num(_blk_col_num),
          num_blk(_data.size / tile_size) {
        cudaChk(cudaMalloc(&dev_ptr, sizeof(tile_mat)));
        cudaChk(cudaMemcpy(dev_ptr, this, sizeof(tile_mat),
                           cudaMemcpyHostToDevice));
    }
    tile_mat(const std::vector<int> &_row_ptr,
             const std::vector<int> &_row_offset,
             const std::vector<half> &_data, int _blk_col_num)
        : row_ptr(_row_ptr), row_offset(_row_offset), data(_data),
          blk_row_num(_row_ptr.size() - 1), blk_col_num(_blk_col_num),
          num_blk(_data.size() / tile_size) {
        cudaChk(cudaMalloc(&dev_ptr, sizeof(tile_mat)));
        cudaChk(cudaMemcpy(dev_ptr, this, sizeof(tile_mat),
                           cudaMemcpyHostToDevice));
    }
    tile_mat(culib::CUDA_ptr<int> &&_row_ptr,
             culib::CUDA_ptr<int> &&_row_offset, culib::CUDA_ptr<half> &&_data,
             int _blk_col_num)
        : row_ptr(std::move(_row_ptr)), row_offset(std::move(_row_offset)),
          data(std::move(_data)), blk_row_num(_row_ptr.size - 1),
          blk_col_num(_blk_col_num), num_blk(_data.size / tile_size) {
        cudaChk(cudaMalloc(&dev_ptr, sizeof(tile_mat)));
        cudaChk(cudaMemcpy(dev_ptr, this, sizeof(tile_mat),
                           cudaMemcpyHostToDevice));
    }
    static tile_mat gen_sparse_mat(int nrow, int ncol, float sparsity);
    static tile_mat gen_dense_mat(int nrow, int ncol);
};