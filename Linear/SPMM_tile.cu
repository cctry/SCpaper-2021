#include "Linear.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <mma.hpp>

namespace cg = cooperative_groups;
/*
 * It returns the pointer of the top-left corner of give block in a matrix.
 * Assume the matrix is stored in a row-major array.
 * It needs the number of columns of the matrix (leading dimension).
 */
template <typename T, int SIZE = 16>
__device__ T *get_blk_start(T *data, const int row_blk, const int col_blk,
                            const int stride) {
    auto res = &data[row_blk * SIZE * stride + SIZE * col_blk];
    return res;
}

// A*B = C
__global__ void __kernel_blk_mmul_blk_bias_smem(
    const int *__restrict__ A_row_ptr, const int *__restrict__ A_row_offset,
    const half *__restrict__ A_data, const int A_blk_row_num,
    const half *__restrict__ B, half *__restrict__ C,
    const half *__restrict__ bias, const int out_row_blk_num,
    const int in_col_blk_num) {
    // num_warp * 256
    // temp_blk
    extern __shared__ half smem[];
    auto grid = cg::this_grid();
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    auto tile_temp = &smem[warp.meta_group_rank() * 256];
    const auto gwarp_id = grid.thread_rank() >> 5;
    const auto total_warp = grid.size() / 32;
    const auto total_tile = A_blk_row_num * out_row_blk_num;
    for (int t = gwarp_id; t < total_tile; t += total_warp) {
        // const auto out_col_blk_num = A_blk_row_num;
        // out_row and out_col are untransposed positions
        const auto out_row = gwarp_id / A_blk_row_num;
        const auto out_col = gwarp_id % A_blk_row_num;
        using frag_t = culib::mma::mma_t<16, 16, 16>;
        frag_t::a_t<wmma::row_major> a;
        frag_t::b_t<wmma::col_major> b;
        frag_t::c_t<half> c;
        wmma::fill_fragment(c, 0);
        for (auto i = A_row_ptr[out_col]; i < A_row_ptr[out_col + 1]; i++) {
            wmma::load_matrix_sync(a, &A_data[i * 256], 16);
            const half *src =
                get_blk_start(B, out_row, A_row_offset[i], in_col_blk_num * 16);
            wmma::load_matrix_sync(b, src, in_col_blk_num * 16);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(tile_temp, c, 16, wmma::mem_col_major);
        const auto bias_value = bias[out_col * 16 + warp.thread_rank() % 16];
#pragma unroll 8
        for (int i = warp.thread_rank(); i < 256; i += warp.size()) {
            tile_temp[i] += bias_value;
        }
        const auto dst = get_blk_start(C, out_row, out_col, A_blk_row_num * 16);
        wmma::load_matrix_sync(c, tile_temp, 16, wmma::mem_col_major);
        wmma::store_matrix_sync(dst, c, A_blk_row_num * 16,
                                wmma::mem_col_major);
    }
}

Linear<tile_mat>::Linear(int _in_size, int _out_size, const tile_mat &w,
                         const half *b, int _size)
    : weight(w), bias(b, _out_size), in_size(_in_size), out_size(_out_size),
      size(_size) {}

Linear<tile_mat>::Linear(int _in_size, int _out_size, tile_mat &&w,
                         const half *b, int _size)
    : weight(std::move(w)), bias(b, _out_size), in_size(_in_size),
      out_size(_out_size), size(_size) {}

Linear<tile_mat>::Linear(Linear<tile_mat> &&_linear)
    : in_size(_linear.in_size), out_size(_linear.out_size), size(_linear.size),
      weight(std::move(_linear.weight)), bias(std::move(_linear.bias)) {}

template <typename _Tg>
__device__ void clear_smem(const _Tg &group, void *smem, const int N) {
    auto ptr = reinterpret_cast<int *>(smem);
#pragma unroll
    for (int i = group.thread_rank(); i < N / sizeof(int); i += group.size()) {
        ptr[i] = 0;
    }
    group.sync();
}

// A*B = C
template <int num_thd>
__global__ void __kernel_blk_mmul_blk_bias_smem_blk(
    const int *__restrict__ A_row_ptr, const int *__restrict__ A_row_offset,
    const half *__restrict__ A_data, const int A_blk_row_num,
    const half *__restrict__ B, half *__restrict__ C,
    const half *__restrict__ bias, const int out_row_blk_num,
    const int in_col_blk_num) {
    // num_warp * 256
    // temp_blk
    extern __shared__ half smem[];
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    auto tile_temp = &smem[warp.meta_group_rank() * 256];
    const auto out_row = blockIdx.y;
    const auto out_col = blockIdx.x;
    const auto warp_id = cta.thread_rank() >> 5;
    constexpr auto num_warp = num_thd >> 5;
    // clear_smem(cta, smem, num_warp * 256 * sizeof(half));
    // const auto out_col_blk_num = A_blk_row_num;
    // out_row and out_col are untransposed positions
    using frag_t = culib::mma::mma_t<16, 16, 16>;
    frag_t::a_t<wmma::row_major> a;
    frag_t::b_t<wmma::col_major> b;
    frag_t::c_t<half> c;
    wmma::fill_fragment(c, 0);
    const auto ldm = in_col_blk_num << 4;
#pragma unroll
    for (auto i = A_row_ptr[out_col] + warp_id; i < A_row_ptr[out_col + 1];
         i += num_warp) {
        wmma::load_matrix_sync(a, &A_data[i << 8], 16);
        auto src = &B[(out_row << 4) * ldm + (A_row_offset[i] << 4)];
        wmma::load_matrix_sync(b, src, ldm);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(tile_temp, c, 16, wmma::mem_col_major);
    cta.sync();
    // reduce across warp
    int e = cta.thread_rank();
    auto base = &smem[e];
    auto sum = *base;
#pragma unroll
    for (int i = 0; i < num_warp; i++) {
        sum += base[i << 8];
    }
    *base = sum;
    cta.sync();
    // add bias
    if (warp_id == 0) {
        const auto bias_value = bias[(out_col << 4) + warp.thread_rank() % 16];
#pragma unroll 8
        for (int i = warp.thread_rank(); i < 256; i += warp.size()) {
            tile_temp[i] += bias_value;
        }
        const auto dst = get_blk_start(C, out_row, out_col, A_blk_row_num << 4);
        wmma::load_matrix_sync(c, tile_temp, 16, wmma::mem_col_major);
        wmma::store_matrix_sync(dst, c, A_blk_row_num << 4,
                                wmma::mem_col_major);
    }
}

// void Linear<tile_mat>::forward(half *output, const half *const input,
//                                cudaStream_t stream) {
//     constexpr int num_thd = 256;
//     auto smem_size = [=](int n) { return sizeof(half) * ((n / 32) * 256); };
//     __kernel_blk_mmul_blk_bias_smem_blk<num_thd>
//         <<<dim3(out_size / 16, size / 16), num_thd, smem_size(num_thd),
//            stream>>>(weight.row_ptr.get(), weight.row_offset.get(),
//                      weight.data.get(), weight.blk_row_num, input, output,
//                      bias.get(), size / 16, in_size / 16);
// }

// void Linear<tile_mat>::forward(half *output, const half *const input,
//                                cudaStream_t stream) {
//     int num_blk = 160, num_thd = 576;
//     auto smem_size = [=](int n) { return sizeof(half) * ((n / 32) * 256); };

//     cudaOccupancyMaxPotentialBlockSizeVariableSMem(
//         &num_blk, &num_thd, __kernel_blk_mmul_blk_bias_smem, smem_size);
//     __kernel_blk_mmul_blk_bias_smem<<<num_blk, num_thd, smem_size(num_thd),
//                                       stream>>>(
//         weight.row_ptr.get(), weight.row_offset.get(), weight.data.get(),
//         weight.blk_row_num, input, output, bias.get(), size / 16, in_size /
//         16);
// }

// A*B = C
// A is CSC, B is transposed
template <int num_thd>
__global__ void __kernel_blk_mmul_blk_bias_smem_blk_outproduct(
    const int *__restrict__ A_row_ptr, const int *__restrict__ A_row_offset,
    const half *__restrict__ A_data, const half *__restrict__ B,
    const int in_col_blk_num, half *__restrict__ C,
    const half *__restrict__ bias, const int ldB, const int ldC) {
    // num_warp * 256
    // temp_blk
    extern __shared__ half smem[];
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    auto eight_thd = cg::tiled_partition<8>(warp);
    const auto K = blockIdx.x;
    const auto warp_id = cta.thread_rank() >> 5;
    constexpr auto num_warp = num_thd / 32;
    auto tile_temp = &smem[warp_id * 256];
    using frag_t = culib::mma::mma_t<16, 16, 16>;
    frag_t::a_t<wmma::row_major> a;
    frag_t::b_t<wmma::col_major> b;
    frag_t::c_t<half> c; // c is stored in col-major
    wmma::fill_fragment(c, 0);
    const auto start_idx = A_row_ptr[K];
    const auto num_nz_tile = A_row_ptr[K + 1] - start_idx;
    for (auto i = warp_id; i < num_nz_tile * in_col_blk_num; i += num_warp) {
        const auto idx = (i % num_nz_tile) + start_idx;
        const auto out_row = i / num_nz_tile;
        const auto out_col = A_row_offset[idx];
        wmma::load_matrix_sync(a, &A_data[idx << 8], 16);
        auto src = get_blk_start(B, K, out_row, ldB);
        wmma::load_matrix_sync(b, src, ldB);
        wmma::mma_sync(c, a, b, c);
        wmma::store_matrix_sync(tile_temp, c, 16, wmma::mem_col_major);
        // FIXME add bias
        const int eight_group_id = eight_thd.meta_group_rank();
        const int eight_thd_id = eight_thd.thread_rank();
        auto temp = &tile_temp[eight_group_id * 16];
        auto dst = get_blk_start(C, out_row, out_col, ldC);
        // for (int k = warp.thread_rank(); k < 256; k += 32) {
        //     auto r = k / 16, c = k % 16;
        //     atomicAdd(&dst[r * ldC + c], tile_temp[r * 16 + c]);
        // }
        auto dst2 = reinterpret_cast<half2 *>(dst);
        for (int k = warp.thread_rank(); k < 128; k += 32) {
            auto r = k / 8, c = k % 8;
            half2 src{tile_temp[r * 16 + 2 * c], tile_temp[r * 16 + 2 * c + 1]};
            dst2[r * ldC / 2 + c] += src;
            // atomicAdd(&dst2[r * ldC / 2 + c], src);
        }
    }
}

// void Linear<tile_mat>::forward(half *output, const half *const input,
//                                cudaStream_t stream) {
//     constexpr int num_thd = 256;
//     constexpr auto smem_size = sizeof(half) * ((num_thd / 32) * 256);
//     assert(weight.row_ptr.size == (in_size / 16) + 1);
//     __kernel_blk_mmul_blk_bias_smem_blk_outproduct<num_thd>
//         <<<in_size / 16, num_thd, smem_size, stream>>>(
//             weight.row_ptr.get(), weight.row_offset.get(), weight.data.get(),
//             input, size / 16, output, bias.get(), size, out_size);
// }

static constexpr int FP16_skew = 16;

// A*B = C
template <int num_thd>
__global__ void __kernel_blk_mmul_blk_bias_smem_blk_skew(
    const int *__restrict__ A_row_ptr, const int *__restrict__ A_row_offset,
    const half *__restrict__ A_data, const int A_blk_row_num,
    const half *__restrict__ B, half *__restrict__ C,
    const half *__restrict__ bias, const int out_row_blk_num,
    const int in_col_blk_num) {
    // num_warp * 256
    // temp_blk
    extern __shared__ half smem[];
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    constexpr auto num_warp = num_thd >> 5;
    const auto warp_id = cta.thread_rank() >> 5;
    auto tile_temp = &smem[warp_id * 16];
    constexpr auto tile_ldm = num_warp * 16 + FP16_skew;
    const auto out_row = blockIdx.y;
    const auto out_col = blockIdx.x;
    // clear_smem(cta, smem, num_warp * 256 * sizeof(half));
    // const auto out_col_blk_num = A_blk_row_num;
    // out_row and out_col are untransposed positions
    using frag_t = culib::mma::mma_t<16, 16, 16>;
    frag_t::a_t<wmma::row_major> a;
    frag_t::b_t<wmma::col_major> b;
    frag_t::c_t<half> c;
    wmma::fill_fragment(c, 0);
    const auto ldm = in_col_blk_num << 4;
#pragma unroll
    for (auto i = A_row_ptr[out_col] + warp_id; i < A_row_ptr[out_col + 1];
         i += num_warp) {
        wmma::load_matrix_sync(a, &A_data[i << 8], 16);
        auto src = &B[(out_row << 4) * ldm + (A_row_offset[i] << 4)];
        wmma::load_matrix_sync(b, src, ldm);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(tile_temp, c, tile_ldm, wmma::mem_col_major);
    cta.sync();
    // reduce across warp
    int e = cta.thread_rank();
    auto base = &smem[e];
    auto sum = *base;
#pragma unroll
    for (int i = 0; i < num_warp; i++) {
        sum += base[i << 8];
    }
    *base = sum;
    cta.sync();
    // add bias
    if (warp_id == 0) {
        const auto sub_group_id = warp.thread_rank() % 2;
        const auto sub_group_tid = warp.thread_rank() % 16;
        const auto bias_value = bias[(out_col << 4) + sub_group_tid];
#pragma unroll 8
        for (int r = sub_group_id; r < 16; r += 2) {
            tile_temp[r * tile_ldm + sub_group_tid] += bias_value;
        }
        const auto dst = get_blk_start(C, out_row, out_col, A_blk_row_num << 4);
        wmma::load_matrix_sync(c, tile_temp, tile_ldm, wmma::mem_col_major);
        wmma::store_matrix_sync(dst, c, A_blk_row_num << 4,
                                wmma::mem_col_major);
    }
}

void Linear<tile_mat>::forward(half *output, const half *const input,
                               cudaStream_t stream) {
    constexpr int num_thd = 256;
    auto smem_size = [=](int n) {
        const auto num_warp = n / 32;
        return sizeof(half) * 16 * (num_warp * 16 + FP16_skew);
    };
    __kernel_blk_mmul_blk_bias_smem_blk_skew<num_thd>
        <<<dim3(out_size / 16, size / 16), num_thd, smem_size(num_thd),
           stream>>>(weight.row_ptr.get(), weight.row_offset.get(),
                     weight.data.get(), weight.blk_row_num, input, output,
                     bias.get(), size / 16, in_size / 16);
}