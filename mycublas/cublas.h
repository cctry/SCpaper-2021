#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>

cublasStatus_t cuBLAS_mmul(const cublasHandle_t &cublasHandle, const half *A,
                 const half *B, half *C, const int M, const int N, const int K,
                 const bool trans_A, const bool trans_B,
                 cublasGemmAlgo_t = CUBLAS_GEMM_DEFAULT);

void cuBLAS_bmmul(const half *A, const half *B, half *C, const int M,
                  const int N, const int K, int num_batch,
                  const bool trans_A = false, const bool trans_B = false);
