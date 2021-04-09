#include "cublas.h"
#include <assert.h>
#include <utils.h>
cublasStatus_t cuBLAS_mmul(const cublasHandle_t &cublasHandle, const half *A,
                           const half *B, half *C, const int M, const int N,
                           const int K, const bool trans_A, const bool trans_B,
                           cublasGemmAlgo_t algo) {
    const static half alpha = (half)1.0f;
    const static half beta = (half)0.0f;
    const auto OP_A = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto OP_B = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto lda = trans_A ? M : K;
    const auto ldb = trans_B ? K : N;
    const auto ldc = N;
    // auto status = cublasHgemm(cublasHandle, OP_B, OP_A, N, M, K, &alpha, B,
    // ldb, A, lda, &beta, C, ldc);
    auto status = cublasGemmEx(cublasHandle, OP_B, OP_A, N, M, K, &alpha, B,
                               CUDA_R_16F, ldb, A, CUDA_R_16F, lda, &beta, C,
                               CUDA_R_16F, ldc, CUDA_R_16F, algo);
    assert(status == CUBLAS_STATUS_SUCCESS);
    return status;
}

/*
 * This function assume the each batch is row-major.
 * E.g. For two 2*2 matrices:
 * mat1 = [1 2;3 4]; mat2 = [5 6;7 8]
 * The layout:
 * 1 2 | 5 6
 * 3 4 | 7 8
 * Storage: 1 2 3 4 5 6 7 8
 */
void cuBLAS_bmmul(const half *A, const half *B, half *C, const int M,
                  const int N, const int K, int num_batch, const bool trans_A,
                  const bool trans_B) {
    static cublasHandle_t handle;
    static bool first_time = true;
    if (first_time) {
        cublasCreate(&handle);
        first_time = false;
    }
    const static half alpha = half_one;
    const static half beta = half_zero;
    const auto OP_A = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto OP_B = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto lda = (trans_A ? M : K) * num_batch;
    const auto ldb = (trans_B ? K : N) * num_batch;
    const auto ldc = N * num_batch;
    const auto strideA = trans_A ? M : K;
    const auto strideB = trans_B ? K : N;
    const auto strideC = N;
    cublasGemmStridedBatchedEx(handle, OP_B, OP_A, N, M, K, &alpha, B, CUDA_R_16F,ldb,
                              strideB, A, CUDA_R_16F, lda, strideA, &beta, C, CUDA_R_16F, ldc, strideC,
                              num_batch, CUDA_R_16F, CUBLAS_GEMM_ALGO0_TENSOR_OP);
}