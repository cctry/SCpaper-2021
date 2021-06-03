#pragma once
#include "cublas.h"
#include <CUDA_ptr.hpp>
#include <cublas.h>
#include <cuda_map.cuh>
#include <cusparse.h>
#include <mat_type.h>
#include <memory>
using base_mat = culib::CUDA_ptr<half>;

/*
 * Applies a linear transformation to the incoming data: y = xA^T + b
 * x is a row vector
 * A is a row-major matrix. Height: in_size. Width: out_size
 */
template <typename _MAT_T> class Linear {
  public:
    const int in_size;
    const int out_size;
    const int dim;
    cublasHandle_t handle;
    Linear(int _in_size, int _out_size, const _MAT_T &w, const half *b,
           int _size) {
        cublasCreate(&handle);
    };
    Linear(int _in_size, int _out_size, _MAT_T &&w, const half *b, int _size) {
        static_assert(-1);
    };
    void forward(half *output, half *input, cudaStream_t stream = 0) const {};
};

template <> class Linear<base_mat> {
  public:
    const int in_size;
    const int out_size;
    base_mat weight;
    culib::CUDA_ptr<half> bias;
    cublasHandle_t handle;

    int size;

    Linear(int _in_size, int _out_size, const base_mat &w, const half *b,
           int _size)
        : weight(w), bias(b, _out_size), in_size(_in_size), out_size(_out_size),
          size(_size) {
        cublasCreate(&handle);
    };
    Linear(int _in_size, int _out_size, base_mat &&w, const half *b, int _size)
        : weight(std::move(w)), bias(b, _out_size), in_size(_in_size),
          out_size(_out_size), size(_size) {
        cublasCreate(&handle);
    };
    void forward(half *output, half *input, cudaStream_t stream = 0) const {
        cublasSetStream(handle, stream);
        auto bias_temp = this->bias.get();
        auto stride = out_size;
        const auto add_bias = [bias_temp, stride] __device__(half * data,
                                                             int i) -> half {
            return data[i] + bias_temp[i % stride];
        };
        cuBLAS_mmul(handle, input, weight.get(), output, size, out_size,
                    in_size, false, true);
        culib::cuda_map(output, size * out_size, add_bias, stream);
    };
    ~Linear() { cublasDestroy(handle); }
};

// The transposed matrix are column pruned
template <> class Linear<row_mat> {
  public:
    const uint32_t in_size;
    const uint32_t out_size;
    row_mat weight;
    culib::CUDA_ptr<half> bias;
    cublasHandle_t handle;

    int size;
    culib::CUDA_ptr<half> workspace;

    Linear(int _in_size, int _out_size, const row_mat &w, const half *b,
           int _size);
    Linear(int _in_size, int _out_size, row_mat &&w, const half *b, int _size);
    void forward(half *output, half *input, cudaStream_t stream = 0) const;
    void forward(col_mat *output, half *input, cudaStream_t stream = 0) const;
    ~Linear();
};

template <> class Linear<csr_mat> {
  public:
    const uint32_t in_size;
    const uint32_t out_size;
    const int size;

    culib::CUDA_ptr<int> *indptr;
    culib::CUDA_ptr<int> *indices;
    culib::CUDA_ptr<half> *data;

    cusparseSpMatDescr_t spWeight;
    culib::CUDA_ptr<half> bias;

    cusparseDnMatDescr_t denmat, resmat;
    cusparseHandle_t handle;

    static const auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    static const auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    Linear(int _in_size, int _out_size, const csr_mat &w, const half *b,
           int _size);
    Linear(Linear<csr_mat> &&_linear);
    void forward(half *output, half *input, cudaStream_t stream = 0);
    ~Linear();
};

// The transposed matrix are row pruned
template <> class Linear<col_mat> {
  public:
    const uint32_t in_size;
    const uint32_t out_size;
    col_mat weight;
    culib::CUDA_ptr<half> bias;
    cublasHandle_t handle;

    int size;
    culib::CUDA_ptr<half> workspace;

    Linear(int _in_size, int _out_size, const col_mat &w, const half *b,
           int _size);
    Linear(int _in_size, int _out_size, col_mat &&w, const half *b, int _size);
    void forward(half *output, half *input, cudaStream_t stream = 0) const;
    ~Linear();
};

template <> class Linear<tile_mat> {
  public:
    const uint32_t in_size;
    const uint32_t out_size;
    const int size;

    tile_mat weight;

    culib::CUDA_ptr<half> bias;

    Linear(int _in_size, int _out_size, const tile_mat &w, const half *b,
           int _size);
    Linear(int _in_size, int _out_size, const tile_mat &w, culib::CUDA_ptr<half>& b,
           int _size);
    Linear(int _in_size, int _out_size, tile_mat &&w, const half *b, int _size);
    Linear(Linear<tile_mat> &&_linear);
    void forward(half *output, const half *const input,
                 cudaStream_t stream = 0);
};

template <> class Linear<mat1x16> {
  public:
    const uint32_t in_size;
    const uint32_t out_size;
    const int size;

    mat1x16 weight;
    culib::CUDA_ptr<half> bias;

    Linear(int _in_size, int _out_size, const mat1x16 &w, const half *b,
           int _size);
    Linear(int _in_size, int _out_size, mat1x16 &&w, const half *b, int _size);
    void forward(half *output, half *input, cudaStream_t stream = 0);
    void forward(float *output, half *input, cudaStream_t stream = 0);
};

#include "gen_linear.hpp"
