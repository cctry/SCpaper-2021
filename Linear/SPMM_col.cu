#include "Linear.h"
#include <col_mat/col_mat.hpp>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cublas.h>
namespace cg = cooperative_groups;

__global__ void __kernel_compress(const half __restrict__ *mat,
                                  const int *col_id, half __restrict__ *res,
                                  const int nCOL, const int height,
                                  const int width) {
    extern __shared__ int smem[];
    auto cta = cg::this_thread_block();
    cg::memcpy_async(cta, smem, col_id, sizeof(int) * nCOL);
    auto grid = cg::this_grid();
    cg::wait(cta);
    for (auto i = grid.thread_rank(); i < nCOL * height; i += grid.size()) {
        const auto pos_x = i % nCOL;
        const auto pos_y = i / nCOL;
        const auto temp = mat[smem[pos_x] + pos_y * width];
        res[pos_x + pos_y * nCOL] = temp;
    }
}

void SPMM_col(const cublasHandle_t &handle, const col_mat &mat,
              const half *dense_mat, half *output, const int M, const int N,
              const int K, half *workspace) {
    cudaStream_t temp_stream;
    cublasGetStream(handle, &temp_stream);
    int num_thd, num_blk;
    auto smem_size = mat.col_id->size * sizeof(uint32_t);
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &num_blk, &num_thd, __kernel_compress,
        [&](int n) { return smem_size; });
    __kernel_compress<<<num_blk, num_thd, smem_size, temp_stream>>>(
        dense_mat, mat.col_id->get(), workspace, mat.col_id->size, M, K);
    cuBLAS_mmul(handle, workspace, mat.cols->get(), output, M, N,
                mat.col_id->size, false, true);
}

Linear<col_mat>::~Linear() { cublasDestroy(handle); }

Linear<col_mat>::Linear(int _in_size, int _out_size, const col_mat &w,
                        const half *b, int _size)
    : weight(w), bias(b, _out_size), in_size(_in_size), out_size(_out_size),
      size(_size), workspace(w.cols->size * _size) {
    cublasCreate(&handle);
}

Linear<col_mat>::Linear(int _in_size, int _out_size, col_mat &&w, const half *b,
                        int _size)
    : weight(std::move(w)), bias(b, _out_size), in_size(_in_size),
      out_size(_out_size), size(_size), workspace(w.cols->size * _size) {
    cublasCreate(&handle);
}

void Linear<col_mat>::forward(half *output, half *input,
                              cudaStream_t stream) const {
    cublasSetStream(handle, stream);
    auto bias_temp = this->bias.get();
    auto stride = out_size;
    const auto add_bias = [bias_temp, stride] __device__(half * data,
                                                         int i) -> half {
        return data[i] + bias_temp[i % stride];
    }; // unpruned bias
    SPMM_col(handle, weight, input, output, size, out_size, in_size,
             workspace.get());
    culib::cuda_map(output, size * out_size, add_bias, stream);
}
