#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.hpp>
#include <utils.h>
namespace cg = cooperative_groups;

__global__ void
__kernel_multi_head_full_single_kernel(const half *__restrict__ Q, const half *__restrict__ K,
                         const half *__restrict__ V, half *__restrict__ Z,
                         const int kdim, const int vdim, const int seq_len,
                         const int num_head, const half *__restrict__ mask,
                         const int ldm) {
    // blockIdx.x: block row id
    // blockIdx.y: head_id
    using frag_t = culib::mma::mma_t<16, 16, 16>;
    extern __shared__ half smem[];
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    const auto warp_id = warp.meta_group_rank();
    const auto lane_id = warp.thread_rank();
    const auto num_warp = warp.meta_group_size();
    const auto head_dim = kdim / num_head;
    auto Q_ptr = &Q[16 * blockIdx.x * ldm + head_dim * blockIdx.y];
    auto temp_Q = smem;
    const half scale = hsqrt(__int2half_rn(head_dim));
    for (int r = warp_id; r < 16; r += num_warp) {
        auto dst = &temp_Q[r * head_dim];
        auto src = &Q_ptr[r * ldm];
        cg::memcpy_async(warp, dst, src, sizeof(half) * head_dim);
        // auto dst2 = reinterpret_cast<half2 *>(dst);
        // for (int i = lane_id; i < head_dim / 2; i += 32) {
        //     dst2[i] = __h2div(dst2[i], half2{scale, scale});
        // }
    }
    cta.sync();
    frag_t::a_t<wmma::row_major> a_frag;
    frag_t::b_t<wmma::col_major> b_frag;
    frag_t::c_t<half> c_frag;
    auto temp_row = &smem[16 * head_dim];
    for (int KR = warp_id; KR < seq_len / 16; KR += num_warp) {
        auto K_ptr = &K[16 * KR * ldm + head_dim * blockIdx.y];
        wmma::fill_fragment(c_frag, half_zero);
        for (int i = 0; i < head_dim; i += 16) {
            wmma::load_matrix_sync(a_frag, smem + i, head_dim);
            wmma::load_matrix_sync(b_frag, K_ptr + i, ldm);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(temp_row + KR * 16, c_frag, seq_len,
                                wmma::mem_row_major);
    }
    cta.sync();
    // mask
    const auto mask_base =
        reinterpret_cast<const half2 *>(&mask[(blockIdx.x * 16) * seq_len]);
    auto temp_row_2 = reinterpret_cast<half2 *>(temp_row);
    for (int i = threadIdx.x; i < seq_len * 8; i += blockDim.x) {
        temp_row_2[i] += mask_base[i];
    }
    cta.sync();
    for (int row = warp_id; row < 16; row += num_warp) {
        auto row_ptr = temp_row + row * seq_len;
        // find the max
        half val_max = half_zero, temp;
        for (auto i = warp.thread_rank(); i < seq_len; i += warp.size()) {
            temp = row_ptr[i];
            val_max = val_max > temp ? val_max : temp;
        }
        warp.sync();
        const auto max = cg::reduce(warp, val_max, cg::greater<half>());
        // compute the sum of exp-ed and shifted array
        half val_sum = half_zero;
        for (auto i = warp.thread_rank(); i < seq_len; i += warp.size()) {
            temp = hexp(row_ptr[i] - max);
            val_sum += temp;
            row_ptr[i] = temp;
        }
        warp.sync();
        const auto sum = cg::reduce(warp, val_sum, cg::plus<half>());
        // update with softmax scaling
        for (auto i = warp.thread_rank(); i < seq_len; i += warp.size()) {
            row_ptr[i] = row_ptr[i] / sum;
        }
    }
    cta.sync();
    const auto vhead_dim = vdim / num_head;
    for (int VC = warp_id; VC < vhead_dim / 16; VC += num_warp) {
        frag_t::b_t<wmma::row_major> b_frag;
        wmma::fill_fragment(c_frag, half_zero);
        for (int i = 0; i < seq_len; i += 16) {
            auto V_ptr = &V[ldm * i + blockIdx.y * vhead_dim + VC * 16];
            wmma::load_matrix_sync(a_frag, temp_row + i, seq_len);
            wmma::load_matrix_sync(b_frag, V_ptr, vdim);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        auto res =
            &Z[blockIdx.x * 16 * vdim + blockIdx.y * vhead_dim + VC * 16];
        wmma::store_matrix_sync(res, c_frag, vdim, wmma::mem_row_major);
    }
}