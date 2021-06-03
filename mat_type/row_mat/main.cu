#include "SPMM_row.cuh"
#include "cublas.h"
#include "row_mat.hpp"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <wtime.h>
using namespace culib;
template <typename T> void prt(const std::vector<T> &vec) {
    for (auto i : vec)
        std::cout << i << ",";
    puts("");
}

__global__ void verify(const half *__restrict__ rows,
                       const half *__restrict__ mat,
                       const uint32_t *__restrict__ row_id,
                       const half *__restrict__ res, const uint32_t M,
                       const uint32_t N, const uint32_t K,
                       const uint32_t num_rows) {
    const auto tid = grid_tid;
    if (tid >= num_rows)
        return;
    half sum = half_zero;
    for (int i = 0; i < N; i++) {
        sum = __hfma(rows[tid * N + i], half_one, sum);
    }
    auto r = row_id[tid];
    printf("tid %d\t%f\t%f\n", tid, __half2float(sum),
           __half2float(res[r * N]));
}

int main(int ac, char **av) {
    std::string name(av[1]);
    row_mat rm(name);
    std::cout << rm.nrow << "\t" << rm.ncol << std::endl;
    if (rm.row_id.size() < 32) {
        prt(rm.row_id);
    }
    CUDA_ptr<float> d_rows_f(rm.rows);
    CUDA_ptr<half> d_rows_h(rm.rows.size());
    util::to_half_devptr(d_rows_h.get(), d_rows_f.get(), d_rows_f.size);
    std::cout << d_rows_h.size << std::endl;

    int N = rm.nrow;

    CUDA_ptr<half> d_B_h(rm.ncol * N, half_one);
    CUDA_ptr<half> d_res(rm.nrow * N);
    CUDA_ptr<uint32_t> d_row_id(rm.row_id);

    constexpr int tile_h = 64, tile_k = 32, tile_w = 64;
    constexpr int num_thd = (tile_w / 16) * (tile_h / 16) * 32;

    const auto t_row = rm.row_id.size() / tile_h;
    const auto t_col = rm.ncol / tile_w;

    // const auto t_row = 1;
    // const auto t_col = 1;

    __kernel_SPMM_row<num_thd, tile_h, tile_w, tile_k>
        <<<dim3(t_col, t_row), num_thd>>>(d_rows_h.get(), d_B_h.get(),
                                          d_row_id.get(), d_res.get(), rm.nrow,
                                          N, rm.ncol, d_row_id.size);

#ifdef TIME
    cudaDeviceSynchronize();
    auto time = wtime(
        10,
        [&]() {
            __kernel_SPMM_row<num_thd, tile_h, tile_w, tile_k>
                <<<dim3(t_col, t_row), num_thd>>>(
                    d_rows_h.get(), d_B_h.get(), d_row_id.get(), d_res.get(),
                    rm.nrow, N, rm.ncol, d_row_id.size);
            cudaDeviceSynchronize();
        },
        [&]() { d_res.clear(); });
    printf("time: %lf (us)\n", time);
#endif
    std::vector<half> res(rm.nrow * N);
    d_res.dump(res.data());
    std::cout << __half2float(res[0]) << std::endl;
    // verify<<<128, 128>>>(d_rows_h.get(), d_B_h.get(), d_row_id.get(),
    //                      d_res.get(), rm.nrow, N, rm.ncol, d_row_id.size);
    // cudaDeviceSynchronize();

    CUDA_ptr<half> d_A_h(rm.nrow * rm.ncol, __float2half_rn(0.5));
    cuBLAS_mmul(d_A_h.get(), d_B_h.get(), d_res.get(), rm.nrow, N, rm.ncol);
#ifdef TIME
    cudaDeviceSynchronize();
    time = wtime(
        10,
        [&]() {
            cuBLAS_mmul(d_A_h.get(), d_B_h.get(), d_res.get(), rm.nrow, N,
                        rm.ncol);
            cudaDeviceSynchronize();
        },
        [&]() { d_res.clear(); });
    printf("time: %lf (us)\n", time);
#endif
    d_res.dump(res.data());
    std::cout << __half2float(res[0]) << std::endl;
    return 0;
}

int main1(int ac, char **av) {
    std::string name(av[1]);
    row_mat rm(name);
    std::cout << rm.nrow << "\t" << rm.ncol << std::endl;
    if (rm.row_id.size() < 32) {
        prt(rm.row_id);
    }
    CUDA_ptr<float> d_rows_f(rm.rows);
    CUDA_ptr<half> d_rows_h(rm.rows.size());
    util::to_half_devptr(d_rows_h.get(), d_rows_f.get(), d_rows_f.size);
    std::cout << d_rows_h.size << std::endl;

    int N = rm.nrow;

    CUDA_ptr<half> d_B_h(rm.ncol * N, half_one);
    CUDA_ptr<half> d_res(rm.nrow * N);
    CUDA_ptr<uint32_t> d_row_id(rm.row_id);

    constexpr auto num_thd = 128;
    __kernel_SPMM_row_plain<num_thd>
        <<<dim3(rm.ncol / 16, rm.row_id.size() / 16), num_thd>>>(
            d_rows_h.get(), d_B_h.get(), d_row_id.get(), d_res.get(), rm.nrow,
            N, rm.ncol, d_row_id.size);

    #ifdef TIME
    cudaDeviceSynchronize();
    auto time = wtime(
        10,
        [&]() {
    __kernel_SPMM_row_plain<num_thd>
        <<<dim3(rm.ncol / 16, rm.row_id.size() / 16), num_thd>>>(
            d_rows_h.get(), d_B_h.get(), d_row_id.get(), d_res.get(), rm.nrow,
            N, rm.ncol, d_row_id.size);
            cudaDeviceSynchronize();
        },
        [&]() { d_res.clear(); });
    printf("time: %lf (us)\n", time);
#endif

    
    std::vector<half> res(rm.nrow * N);
    d_res.dump(res.data());
    std::cout << __half2float(res[0]) << std::endl;
    // verify<<<128, 128>>>(d_rows_h.get(), d_B_h.get(), d_row_id.get(),
    //                      d_res.get(), rm.nrow, N, rm.ncol, d_row_id.size);
    // cudaDeviceSynchronize();



    CUDA_ptr<half> d_A_h(rm.nrow * rm.ncol, __float2half_rn(0.5));
    cuBLAS_mmul(d_A_h.get(), d_B_h.get(), d_res.get(), rm.nrow/2, N, rm.ncol);
#ifdef TIME
    cudaDeviceSynchronize();
    time = wtime(
        10,
        [&]() {
            cuBLAS_mmul(d_A_h.get(), d_B_h.get(), d_res.get(), rm.nrow/2, N,
                        rm.ncol);
            cudaDeviceSynchronize();
        },
        [&]() { d_res.clear(); });
    printf("time: %lf (us)\n", time);
#endif
    d_res.dump(res.data());
    std::cout << __half2float(res[0]) << std::endl;
    return 0;
}