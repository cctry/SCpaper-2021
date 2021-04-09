#include "kernels.h"
#include "prune_attn.h"
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <mma.hpp>
#include <cuda_fp16.h>
namespace cg = cooperative_groups;

using frag_t = culib::mma::mma_t<16, 16, 16>;

template <typename _Tg>
__device__ void clear_smem(const _Tg &group, void *smem, const int N) {
    auto ptr = reinterpret_cast<int *>(smem);
#pragma unroll
    for (int i = group.thread_rank(); i < N; i += group.size()) {
        ptr[i] = 0;
    }
    group.sync();
}

