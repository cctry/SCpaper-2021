#include "../Linear.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <mma.hpp>
#include <wtime.h>
using namespace culib;
template <typename T, int SIZE = 16>
__device__ T *get_blk_start(T *data, const int row_blk, const int col_blk,
                            const int stride) {
    return &data[row_blk * SIZE * stride + SIZE * col_blk];
}
__global__ void __kernel_dense2blk(const half *__restrict__ dense,
                                   half *__restrict__ blk, const int ncol) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto warp_id = tid / 32;
    const auto blk_col_num = ncol / 16;
    const auto row_blk = warp_id / blk_col_num;
    const auto col_blk = warp_id % blk_col_num;
    using frag_t = culib::mma::mma_t<16, 16, 16>;
    frag_t::c_t<half> frag;

    const auto src = get_blk_start(dense, row_blk, col_blk, ncol);
    wmma::load_matrix_sync(frag, src, ncol, wmma::mem_row_major);
    const auto dst = &blk[(blk_col_num * row_blk + col_blk) * 256];
    wmma::store_matrix_sync(dst, frag, 16, wmma::mem_row_major);
}

/*
 * When linear operation:
 * num_blk_row is the output_size
 * num_blk_col is the input_size
 */
std::unique_ptr<tile_mat>
gen_from_dense(const int num_blk_row, const int num_blk_col, const half *data) {
    const auto num_blk = num_blk_row * num_blk_col;
    std::vector<int> row_ptr(num_blk_row + 1), row_offset(num_blk);
    int temp = 0;
    for (int i = 0; i < num_blk_row + 1; i++) {
        row_ptr[i] = temp;
        temp += num_blk_col;
    }
    for (int i = 0; i < num_blk_row; i++) {
        int temp = 0;
        for (int j = 0; j < num_blk_col; j++) {
            row_offset[i * num_blk_col + j] = temp;
            temp++;
        }
    }

    auto res = std::make_unique<tile_mat>(
        CUDA_ptr<int>(row_ptr), CUDA_ptr<int>(row_offset),
        CUDA_ptr<half>(data, num_blk * 256), num_blk_col);
    return std::move(res);
}

int main(int ac, char **av) {
    auto in_size = std::atoi(av[1]);
    auto out_size = std::atoi(av[2]);
    auto seq_len = std::atoi(av[3]);
    float sparsity = std::atof(av[4]);
    CUDA_ptr<half> IN(seq_len * in_size, half_one); // input
    CUDA_ptr<half> OUT(out_size * seq_len);         // output
    using mat_t = tile_mat;
    auto layer = gen_sparse_linear<mat_t>(out_size, in_size, seq_len, sparsity);
    layer->forward(OUT.get(), IN.get());
    OUT.clear();
    double time = wtime_new(
        100, [&]() { layer->forward(OUT.get(), IN.get()); },
        [&]() { cudaChk(cudaDeviceSynchronize()); });
    std::cout << typeid(mat_t).name() << ":\t" << time << std::endl;
    std::vector<half> res(out_size * seq_len);
    OUT.dump(res.data());
    std::cout << __half2float(res[128 * 1000]) << ",";
    puts("");
}