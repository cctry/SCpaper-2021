#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <cuda_runtime_api.h>

#ifdef __CUDACC__
#define DEVICE_CALLABLE __host__ __device__
#else
#define DEVICE_CALLABLE
#endif

#ifndef __NVCC__
#define __device__
#define __global__
#endif

struct AutoExec {
    template <typename Callable> AutoExec(Callable &&callable) { callable(); }
};

struct cuda_helper_t {
    cudaDeviceProp prop;
    AutoExec ae{[this] {
        int device = 0;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
    }};
};

#define cudaChk(stat)                                                          \
    { culib::util::cudaErrCheck_((stat), __FILE__, __LINE__); }
#define cudaSMEMCheck(prop, size)                                              \
    { assert(size <= prop.sharedMemPerBlock); }
#define half_one (__half_raw{.x = 0x3c00})
#define half_zero (__half_raw{.x = 0})
#define half_inf (__half_raw{.x = 0x7c00})
#define half_ninf (__half_raw{.x = 0xFc00})
#define grid_tid (blockDim.x * blockIdx.x + threadIdx.x)
namespace culib {
namespace util {
template <bool Cond, typename T, typename U> struct type_if {};
template <typename T, typename U> struct type_if<true, T, U> {
    using type = T;
};
template <typename T, typename U> struct type_if<false, T, U> {
    using type = U;
};

void sync_streams(cudaStream_t *streams, int num);
void to_half_devptr(half *dst, const float *src, const size_t len);
void cuda_free_safe(void *p);
void cudaErrCheck_(cudaError_t stat, const char *file, int line);
void cuda_chk_launch(int num_thd, int num_blk, size_t smem);
int cuda_num_thd(int num);
__device__ unsigned dynamic_smem_size();
template <typename T>
__device__ T warp_scan(T val, const int logic_id,
                       const uint32_t mask = 0xFFFFFFFF) {
    T temp;
    const auto size = __popc(mask);
#pragma unroll
    for (int i = 1; i < size; i *= 2) {
        temp = __shfl_up_sync(mask, val, i);
        if (logic_id >= i)
            val += temp;
    }
    return val;
}
} // namespace util
} // namespace culib
