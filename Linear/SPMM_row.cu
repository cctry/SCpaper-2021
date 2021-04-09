#include "Linear.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cublas.h>
#include <iostream>
namespace cg = cooperative_groups;

/*
 * cols, // arranged as row-major
 * len_idx, // length of idx, number of col compressed
 * len_col, // length of each column in res matrix
 * ncol          // number of col in res matrix
 */
__global__ void decompress_row_mat(const half __restrict__ *cols,
                                   const int *idx, half __restrict__ *res,
                                   const int len_idx, const int len_col,
                                   const int ncol) {
    extern __shared__ int smem[];
    auto cta = cg::this_thread_block();
    cg::memcpy_async(cta, smem, idx, sizeof(int) * len_idx);
    const auto grid = cg::this_grid();
    const auto tid = grid.thread_rank();
    cg::wait(cta);
    for (int i = tid; i < len_idx * len_col; i += grid.size()) {
        const auto r = i / len_idx;
        const auto offset = i % len_idx;
        const auto c = smem[offset];
        res[r * ncol + c] = cols[r * len_idx + offset];
    }
}

void SPMM_row(const cublasHandle_t &handle, const row_mat &mat,
              const half *dense_mat, half *output, const int num,
              const int out_size, const int in_size, half *workspace) {
    cuBLAS_mmul(handle, dense_mat, mat.rows->get(), workspace, num,
                mat.row_id->size, in_size, false, true);
    cudaStream_t temp_stream;
    cublasGetStream(handle, &temp_stream);
    int num_thd, num_blk;
    auto smem_size = mat.row_id->size * sizeof(int);
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &num_blk, &num_thd, decompress_row_mat,
        [&](int n) { return smem_size; });
    decompress_row_mat<<<num_blk, num_thd, smem_size, temp_stream>>>(
        workspace, mat.row_id->get(), output, mat.row_id->size, num, out_size);
}

Linear<row_mat>::Linear(int _in_size, int _out_size, const row_mat &w,
                        const half *b, int _size)
    : weight(w), bias(b, _out_size), in_size(_in_size), out_size(_out_size),
      size(_size), workspace(_size * w.row_id->size) {
    cublasCreate(&handle);
};

Linear<row_mat>::Linear(int _in_size, int _out_size, row_mat &&w, const half *b,
                        int _size)
    : weight(std::move(w)), bias(b, _out_size), in_size(_in_size),
      out_size(_out_size), size(_size), workspace(_size * w.row_id->size) {
    cublasCreate(&handle);
};

void Linear<row_mat>::forward(half *output, half *input,
                              cudaStream_t stream) const {
    cublasSetStream(handle, stream);

    auto bias_temp = this->bias.get();
    auto stride = out_size;
    const auto add_bias = [bias_temp, stride] __device__(half * data,
                                                         int i) -> half {
        return data[i] + bias_temp[i % stride];
    }; // unpruned bias
    SPMM_row(handle, weight, input, output, size, out_size, in_size,
             workspace.get());
    culib::cuda_map(output, size * out_size, add_bias, stream);
};
Linear<row_mat>::~Linear() { cublasDestroy(handle); }

void Linear<row_mat>::forward(col_mat *output, half *input,
                              cudaStream_t stream) const {
    cublasSetStream(handle, stream);

    auto bias_temp = this->bias.get();
    auto col_idx = weight.row_id->get();
    int stride = weight.row_id->size;
    const auto add_bias = [bias_temp, stride,
                           col_idx] __device__(half * data, int i) -> half {
        const auto c = col_idx[i % stride];
        return data[i] + bias_temp[c];
    }; // unpruned bias

    cuBLAS_mmul(handle, input, weight.rows->get(), output->cols->get(), size,
                weight.row_id->size, in_size, false, true);

    // culib::cuda_map(output->cols->get(), size * stride, add_bias, stream);
};