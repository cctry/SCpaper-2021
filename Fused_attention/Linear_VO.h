#pragma once
#include "../Model.h"
#include <CUDA_ptr.hpp>
#include <algorithm>
#include <col_mat/col_mat.hpp>
#include <cublas.h>
#include <cuda_map.cuh>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utils.h>

__global__ void add_bias_VO(half *out, const half *__restrict__ bias,
                            const int emdim, const int seq_len,
                            const int nhead);

template <typename T> constexpr auto div_ceil(const T &x, const T &y) {
    static_assert(std::is_integral<T>::value, "Integral required.");
    return x / y + (x % y != 0);
}

class Linear_VO {

    using ptr_t = culib::CUDA_ptr<half>;

  public:
    ptr_t bias;
    ptr_t weight; // weight is fully pre-processed and row-major
    std::shared_ptr<Model_t> model;
    cublasHandle_t handle;

    Linear_VO(const ptr_t &_weight, const ptr_t &_bias,
              std::shared_ptr<Model_t> _model)
        : model(_model), bias(_bias), weight(_weight) {
        cublasCreate(&handle);
    }
    ~Linear_VO() { cublasDestroy(handle); }
    void forward(half *output, const half *input, cudaStream_t stream = 0) {
        const static half alpha = half_one;
        const static half beta = half_zero;
        cublasSetStream(handle, stream);
        const int M = model->seq_len;
        const int N = model->emdim;
        const int K = model->emdim;
        cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                  &alpha, weight.get(), N, N * K, input, K, 0,
                                  &beta, output, N, M * N, model->nhead);
        const auto smem_size = [&](int n) {
            return 2 * (n / 32) * model->emdim * sizeof(half);
        };
        int num_thd, _num_blk;
        cudaOccupancyMaxPotentialBlockSizeVariableSMem(&_num_blk, &num_thd,
                                                       add_bias_VO, smem_size);
        const auto num_blk =
            div_ceil(model->seq_len * model->nhead, num_thd / 32);
        add_bias_VO<<<num_blk, num_thd, smem_size(num_thd), stream>>>(
            output, bias.get(), model->emdim, model->seq_len, model->nhead);
    }
};

class Linear_VO_prune {

    using ptr_t = culib::CUDA_ptr<half>;

  public:
    ptr_t bias;
    col_mat weight; // weight is fully pre-processed and row-major
    int head_dim;
    std::shared_ptr<Model_t> model;
    cublasHandle_t handle;

    Linear_VO_prune(const col_mat &_weight, const ptr_t &_bias,
                    std::shared_ptr<Model_t> _model)
        : model(_model), bias(_bias), weight(_weight) {
        cublasCreate(&handle);
    }
    ~Linear_VO_prune() { cublasDestroy(handle); }
    void forward(half *output, const half *input, cudaStream_t stream = 0) {
        const static half alpha = half_one;
        const static half beta = half_zero;
        cublasSetStream(handle, stream);
        const int M = model->seq_len;
        const int N = weight.col_id->size;
        const int K = model->emdim;
        cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                  &alpha, weight.cols->get(), N, N * K, input,
                                  K, 0, &beta, output, N, M * N, model->nhead);
        int num_thd, _num_blk;
        cudaOccupancyMaxPotentialBlockSize(&_num_blk, &num_thd,
                                                       add_bias_VO);
        const auto num_blk =
            div_ceil(model->seq_len * model->nhead, num_thd / 32);
        add_bias_VO<<<num_blk, num_thd, 0, stream>>>(
            output, bias.get(), N, model->seq_len, model->nhead);
    }
};