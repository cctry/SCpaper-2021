#pragma once
#include <reduce.cuh>
#include <utils.h>
__global__ void __kernel_multi_head_prune(
    const half *Q, const half *K, const half *V, half *Z, const int kdim,
    const int vdim, const int seq_len, const int num_head, const int *v_col_id,
    const int *v_head_ptr, const int nnz_col_v, const half *mask);