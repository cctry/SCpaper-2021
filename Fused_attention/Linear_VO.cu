#include "Linear_VO.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;

// __global__ void add_bias_VO(half *out, const half *__restrict__ bias,
//                             const int emdim, const int seq_len,
//                             const int nhead) {
//     // sizeof(half) * emdim * num_warp * 2
//     extern __shared__ half smem[];
//     auto grid = cg::this_grid();
//     const auto row = grid.thread_rank() / 32;
//     if (row > nhead * seq_len)
//         return;
//     auto cta = cg::this_thread_block();
//     auto warp = cg::tiled_partition<32>(cta);
//     const auto warp_id = warp.meta_group_rank();
//     auto bias_smem = &smem[warp_id * 2 * emdim];
//     auto bias_smem2 = reinterpret_cast<half2 *>(bias_smem);
//     auto row_smem = &bias_smem2[emdim / 2];
//     const auto head_id = row / seq_len;
//     cg::memcpy_async(warp, bias_smem, &bias[head_id * emdim],
//                      sizeof(half) * emdim);
//     auto src = reinterpret_cast<half2 *>(&out[row * emdim]);
//     cg::memcpy_async(warp, row_smem, src, sizeof(half) * emdim);
//     cg::wait(warp);
//     for (int i = warp.thread_rank(); i < emdim / 2; i += 32) {
//         src[i] = __hadd2(row_smem[i], bias_smem2[i]);
//     }
// }

__global__ void add_bias_VO(half *out, const half *__restrict__ bias,
                            const int emdim, const int seq_len,
                            const int nhead) {
    auto grid = cg::this_grid();
    const auto row = grid.thread_rank() / 32;
    if (row >= nhead * seq_len)
        return;
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    const auto head_id = row / seq_len;
    for (int i = warp.thread_rank(); i < emdim; i += 32) {
        out[row * emdim + i] = out[row * emdim + i] + bias[head_id * emdim + i];
    }
}