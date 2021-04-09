#include "range.hpp"
#include "utils.h"
#include <algorithm>
#include <cassert>
namespace culib {
namespace util {

struct cuda_helper_t cuda_helper;

void cuda_chk_launch(int num_thd, int num_blk, size_t smem) {
    assert(num_thd <= cuda_helper.prop.maxThreadsPerBlock && "num_thd error");
    assert(num_blk <= 65535 && "num_blk error");
    assert(cuda_helper.prop.sharedMemPerBlock >= smem && "smem error");
}

int cuda_num_thd(int num) {
    return std::min(cuda_helper.prop.maxThreadsPerBlock, num);
}

void cuda_free_safe(void *p) { cudaChk(cudaFree(p)); }

__device__ unsigned dynamic_smem_size() {
    unsigned ret;
    asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    return ret;
}

void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    assert(stat == cudaSuccess);
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat),
                file, line);
        exit(1);
    }
}
__global__ void __kernel_to_half(half *dst, const float *src,
                                 const size_t len) {
    for (auto i : grid_stride_range(std::size_t(0), len)) {
        dst[i] = __float2half_rn(src[i]);
    }
}

void to_half_devptr(half *dst, const float *src, const size_t len) {
    int num_blk, num_thd;
    cudaOccupancyMaxPotentialBlockSize(&num_blk, &num_thd, __kernel_to_half);
    __kernel_to_half<<<num_blk, num_thd>>>(dst, src, len);
}

void sync_streams(cudaStream_t *streams, int num) {
    for (size_t i = 0; i < num; i++) {
        cudaChk(cudaStreamSynchronize(streams[i]));
    }
}

} // namespace util
} // namespace culib