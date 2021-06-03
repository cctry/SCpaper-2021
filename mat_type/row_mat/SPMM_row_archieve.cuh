#include "row_mat.hpp"
#include <assert.h>
#include <cooperative_groups.h>
#include <mma.hpp>
#include <utils.h>
namespace cg = cooperative_groups;

template <typename T, int num_thd, int tile_h, int tile_w>
__device__ void memcpy_blk(T *__restrict__ dst, const T *__restrict__ src,
                           const int stride) {
    constexpr int factor = (tile_h * tile_w) / num_thd;
#pragma unroll factor
    for (int i = threadIdx.x; i < (tile_h * tile_w); i += num_thd) {
        const auto idx = i % tile_w;
        const auto idy = i / tile_w;
        dst[idx + idy * tile_w] = src[idx + idy * stride];
    }
}

template <int num_thd, int tile_h, int tile_w, int tile_k>
__global__ void __launch_bounds__(num_thd)
    __kernel_SPMM_row(const half *__restrict__ rows,
                      const half *__restrict__ mat,
                      const uint32_t *__restrict__ row_id,
                      half *__restrict__ res, const uint32_t M,
                      const uint32_t N, const uint32_t K,
                      const uint32_t num_rows) {
    constexpr int tM = 16, tN = 16, tK = 16;
    static_assert((tile_w / tN) * (tile_h / tM) == num_thd / 32);
    __shared__ uint32_t smem_idx[tile_h];
    __shared__ half smem[tile_h * tile_k + tile_k * tile_w];
    auto smem_A = smem;
    auto smem_B = &smem_A[tile_h * tile_k];

    using frag_t = culib::mma::mma_t<tM, tN, tK>;
    frag_t::a_t<wmma::row_major> a_frag;
    frag_t::b_t<wmma::row_major> b_frag;
    frag_t::c_t<half> c_frag;

    const auto warp_id = threadIdx.x / 32;
    const auto lane_id = threadIdx.x % 32;
    const auto cx_warp = warp_id % (tile_w / tN);
    const auto cy_warp = warp_id / (tile_w / tN);

    const auto cx_blk = blockIdx.x;
    const auto cy_blk = blockIdx.y;
    if (threadIdx.x < tile_h)
        smem_idx[threadIdx.x] = row_id[cy_blk * tile_h + threadIdx.x];
    wmma::fill_fragment(c_frag, half_zero);
    for (int k = 0; k < K; k += tile_k) {
        const auto A_ptr = &rows[cy_blk * tile_h * K + k];
        const auto B_ptr = &mat[k * N + cx_blk * tile_w];
        memcpy_blk<half, num_thd, tile_h, tile_k>(smem_A, A_ptr, K);
        memcpy_blk<half, num_thd, tile_k, tile_w>(smem_B, B_ptr, N);
        __syncthreads();
        #pragma unroll tile_k / tK
        for (int j = 0; j < (tile_k / tK); j++) {
            wmma::load_matrix_sync(
                a_frag, &smem_A[cy_warp * tM * tile_k + j * tK], tile_k);
            wmma::load_matrix_sync(
                b_frag, &smem_B[j * tK * tile_w + cx_warp * tN], tile_w);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    static_assert((num_thd / 32) * 256 <= tile_h * tile_k + tile_k * tile_w);
    auto dst = &smem[cx_warp * tN + cy_warp * tM * tile_w];
    wmma::store_matrix_sync(dst, c_frag, tile_w, wmma::mem_row_major);
    __syncthreads();
    for (int tr = warp_id; tr < tile_h; tr += (num_thd / 32)) {
        const auto r = smem_idx[tr];
        auto dst = &res[r * N + cx_blk * tile_w];
        for (int c = lane_id; c < tile_w; c += 32) {
            dst[c] = smem[tr * tile_w + c];
        }
    }

    // auto dst_base = &res[cx_blk * tile_w + cy_blk * tile_h * N];
    // auto dst = &dst_base[cx_warp * tN + cy_warp * tM * N];
    // wmma::store_matrix_sync(dst, c_frag, N, wmma::mem_row_major);
}

template <int num_thd>
__global__ void __launch_bounds__(num_thd)
    __kernel_SPMM_row_plain(const half *__restrict__ rows,
                            const half *__restrict__ mat,
                            const uint32_t *__restrict__ row_id,
                            half *__restrict__ res, const uint32_t M,
                            const uint32_t N, const uint32_t K,
                            const uint32_t num_rows) {
    constexpr int tM = 16, tN = 16, tK = 16;
    constexpr int num_warp = num_thd / 32;
    __shared__ half smem[tM * tN * num_warp];
    __shared__ uint32_t smem_idx[tM];
    if (threadIdx.x < tM)
        smem_idx[threadIdx.x] = row_id[blockIdx.y * tM + threadIdx.x];
    const auto warp_id = threadIdx.x / 32;

    using frag_t = culib::mma::mma_t<tM, tN, tK>;
    frag_t::a_t<wmma::row_major> a_frag;
    frag_t::b_t<wmma::row_major> b_frag;
    frag_t::c_t<half> c_frag;
    wmma::fill_fragment(c_frag, half_zero);

    for (int i = warp_id; i < (K / tK); i += num_warp) {
        wmma::load_matrix_sync(a_frag, &rows[i * tK + blockIdx.y * tM * K], K);
        wmma::load_matrix_sync(b_frag, &mat[blockIdx.x * tN + i * tK * N], N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    auto dst = &smem[warp_id * tM * tN];
    wmma::store_matrix_sync(dst, c_frag, tN, wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < tM * tN; i += num_thd) {
        half sum = half_zero;
        for (int j = 0; j < num_warp; j++) {
            sum += smem[i + j * tM * tN];
        }
        const auto dx = i % tN;
        const auto dy = i / tN;
        const auto r = smem_idx[dx];
        const auto c = blockIdx.x * tN + dy;
        res[c + r * N] = sum;
    }
}

// ncol > 256
template <int num_thd, int tile_size>
__global__ void __launch_bounds__(num_thd)
    __kernel_SPMM_row_tile(const half *__restrict__ rows,
                           const half *__restrict__ mat,
                           const uint32_t *__restrict__ row_id,
                           half *__restrict__ res, const uint32_t M,
                           const uint32_t N, const uint32_t K,
                           const uint32_t num_rows) {
    __shared__ half smem_rows[tile_size];
}