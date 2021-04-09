#include "Linear.h"
#include <cassert>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <mma.hpp>
#include <omp.h>
#include <utils.h>
namespace cg = cooperative_groups;
using frag_t = culib::mma::mma_t<16, 16, 16>;

/*
 * It returns the pointer of the top-left corner of give block in a matrix.
 * Assume the matrix is stored in a row-major array.
 * It needs the number of columns of the matrix (leading dimension).
 */
template <typename T, int SIZE = 16>
__device__ T *get_blk_start(T *data, const int row_blk, const int col_blk,
                            const int stride) {
    return &data[row_blk * SIZE * stride + SIZE * col_blk];
}

template <typename TyGroup, typename T>
__device__ void memcpy2D(const TyGroup &g, T *__restrict__ dst,
                         const T *__restrict__ src, const int size,
                         const int ldm, const int ld_dst, const int ld_src) {
    for (size_t i = g.thread_rank(); i < size; i += g.size()) {
        const auto r = i / ldm;
        const auto c = i % ldm;
        dst[r * ld_dst + c] = src[r * ld_src + c];
    }
    g.sync();
}

__global__ void __kernel_SPMM1x16(const int *__restrict__ tile_idx,
                                  const int *__restrict__ row_idx,
                                  const half *__restrict__ data,
                                  const int npart,
                                  const half *__restrict__ dense,
                                  half *__restrict__ res, const int size,
                                  const int in_size, const int out_size) {
    // (sizeof(half) * 256 + sizeof(int) * (npart+1)) * num_warp
    extern __shared__ half smem[];
    auto grid = cg::this_grid();
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    // global warp id
    const auto warp_id = warp.meta_group_rank() + blockIdx.x * blockDim.x / 32;
    // total warp in this grid
    const auto num_warp = grid.size() / 32;
    auto blk_temp = &smem[256 * warp.meta_group_rank()];
    auto idx_temp = reinterpret_cast<int *>(blk_temp + 256);
    frag_t::a_t<wmma::row_major> a_frag;
    frag_t::b_t<wmma::col_major> b_frag;
    frag_t::c_t<half> c_frag;
    for (int i = warp_id; i < (out_size * size) / 256; i += num_warp) {
        const auto row_set = i / (size / 16);
        const auto dense_set = i % (size / 16);
        const auto idx_start = &tile_idx[row_set * (npart + 1)];
        cg::memcpy_async(warp, idx_temp, idx_start, sizeof(int) * (npart + 1));
        wmma::fill_fragment(c_frag, 0);
        cg::wait(warp);
        for (int part_id = 0; part_id < npart; part_id++) {
            auto workgroup = cg::tiled_partition<2>(warp);
            const auto n_valid_row = idx_temp[part_id + 1] - idx_temp[part_id];
            if (n_valid_row == 0)
                continue;
            const auto group_id = workgroup.meta_group_rank();
            if (group_id < n_valid_row) {
                // index for this tile
                const auto idx = idx_temp[part_id] + group_id;
                // relative row in 16x16 block
                const auto row_id = row_idx[idx] % 16;
                auto src = &data[16 * idx];
                auto dst = &blk_temp[16 * row_id];
                cg::memcpy_async(workgroup, dst, src, sizeof(half) * 16);
            }
            cg::wait(warp);
            wmma::load_matrix_sync(a_frag, blk_temp, 16);
            auto src = get_blk_start(dense, dense_set, part_id, in_size);
            wmma::load_matrix_sync(b_frag, src, in_size);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        const auto lane_id = warp.thread_rank();
    }
}

void Linear<mat1x16>::forward(half *output, half *input,
                              cudaStream_t stream) {
    //     int num_thd, num_blk;
    //     auto get_smem = [](int n) {
    //         return sizeof(half) * 256 * (1 + n / 32) + sizeof(int) * 16 * (n
    //         / 32);
    //     };
    //     cudaOccupancyMaxPotentialBlockSizeVariableSMem(&num_blk, &num_thd,
    //                                                    __kernel_SPMM1x16,
    //                                                    get_smem);
    //     cudaStream_t _stream;
    //     if (stream)
    //         _stream = *stream;
    //     else
    //         _stream = 0;
    //     __kernel_SPMM1x16<<<dim3(size / 16, weight.npart), num_thd,
    //                         get_smem(num_thd), _stream>>>(
    //         weight.tile_idx.get(), weight.tile_row_idx.get(),
    //         weight.data.get(), input, output, size, in_size, out_size);
    //     auto bias_temp = this->bias.get();
    //     auto stride = out_size;
    //     const auto add_bias = [bias_temp, stride] __device__(half * data,
    //                                                          int i) -> half {
    //         return data[i] + bias_temp[i % stride];
    //     };
    //     culib::cuda_map(output, size * out_size, add_bias, stream);
}

Linear<mat1x16>::Linear(int _in_size, int _out_size, const mat1x16 &w,
                        const half *b, int _size)
    : in_size(_in_size), out_size(_out_size), size(_size), bias(b, _out_size),
      weight(w) {}
Linear<mat1x16>::Linear(int _in_size, int _out_size, mat1x16 &&w, const half *b,
                        int _size)
    : in_size(_in_size), out_size(_out_size), size(_size), bias(b, _out_size),
      weight(w) {}
