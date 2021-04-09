#include <CUDA_ptr.hpp>
#include <Linear/Linear.h>
#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cublas.h>
#include <mma.hpp>
#include <wtime.h>
namespace cg = cooperative_groups;
using namespace culib;

__global__ void __kernel(const half __restrict__ *mat, const int *col_id,
                         half __restrict__ *res, const int nCOL,
                         const int height, const int width) {
    extern __shared__ int smem[];
    auto cta = cg::this_thread_block();
    cg::memcpy_async(cta, smem, col_id, sizeof(int) * nCOL);
    auto grid = cg::this_grid();
    cg::wait(cta);
    for (int i = grid.thread_rank(); i < nCOL * height; i += grid.size()) {
        const int pos_x = i % nCOL;
        const int pos_y = i / nCOL;
        const int temp = mat[smem[pos_x] + pos_y * width];
        res[pos_x + pos_y * nCOL] = temp;
    }
}

double test(int in_size, int out_size, int seq_len, half *output, half *input,
            half *workspace, const col_mat &mat, cublasGemmAlgo_t algo) {
    // bias
    culib::CUDA_ptr<half> bias(out_size, __float2half_rn(0.5));
    auto bias_temp = bias.get();
    auto stride = out_size;
    const auto add_bias = [bias_temp, stride] __device__(half * data,
                                                         int i) -> half {
        return data[i] + bias_temp[i % stride];
    };
    // init
    cublasHandle_t handle;
    cublasCreate(&handle);
    int num_thd, num_blk;
    cudaOccupancyMaxPotentialBlockSize(&num_blk, &num_thd, __kernel,
                                       mat.nnz_col * sizeof(uint32_t));
    // start
    __kernel<<<num_blk, num_thd, mat.nnz_col * sizeof(uint32_t)>>>(
        input, mat.col_id->get(), workspace, mat.nnz_col, seq_len, in_size);

    auto status = cuBLAS_mmul(handle, workspace, mat.cols->get(), output,
                              seq_len, out_size, mat.nnz_col, false, false);
    culib::cuda_map(output, seq_len * out_size, add_bias);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Algo %d is %d\n", algo, status);
        return 9999999;
    }
    cudaChk(cudaDeviceSynchronize());
    if (mat.nnz_col % 16 != 0)
        printf("%d\n", mat.nnz_col);
    // auto time = wtime(
    //     100,
    //     [&]() {
    //         // __kernel<<<num_blk, num_thd, mat.nnz_col *
    //         sizeof(uint32_t)>>>(
    //         //     input, mat.col_id->get(), workspace, mat.nnz_col, seq_len,
    //         //     in_size);
    //         cuBLAS_mmul(handle, workspace, mat.cols->get(), output, seq_len,
    //                     out_size, mat.nnz_col, false, false);
    //         culib::cuda_map(output, seq_len * out_size, add_bias);
    //     },
    //     [&]() {}, [&]() {  });

    double time = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i : culib::range(0, 100)) {
        __kernel<<<num_blk, num_thd, mat.nnz_col * sizeof(uint32_t)>>>(
            input, mat.col_id->get(), workspace, mat.nnz_col, seq_len, in_size);
        cuBLAS_mmul(handle, workspace, mat.cols->get(), output, seq_len,
                    out_size, mat.nnz_col, false, false);
        culib::cuda_map(output, seq_len * out_size, add_bias);
    }
    cudaChk(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
               .count() /
           100.0;
    cublasDestroy(handle);
    // printf("Algo %d time %lf\n", algo, time);
    return time;
}

int main(int ac, char **av) {
    auto in_size = std::atoi(av[1]);
    auto out_size = std::atoi(av[2]);
    auto seq_len = std::atoi(av[3]);
    float sparsity = std::atof(av[4]);
    CUDA_ptr<half> IN(seq_len * in_size, half_one); // input
    CUDA_ptr<half> OUT(out_size * seq_len);         // output
    // weight
    auto weight = col_mat::gen_sparse_mat(out_size, in_size, sparsity);
    // workspace
    CUDA_ptr<half> workspace(weight.nnz_col * seq_len);
    // algos
    std::vector<int> tensor_algos(17);
    std::iota(tensor_algos.begin(), tensor_algos.end(), 99);
    std::vector<int> algos(18);
    std::iota(algos.begin(), algos.end(), 0);
    // result
    std::vector<double> res;
    for (int i : algos) {
        auto algo = static_cast<cublasGemmAlgo_t>(i);
        auto time = test(in_size, out_size, seq_len, OUT.get(), IN.get(),
                         workspace.get(), weight, algo);
        res.push_back(time);
    }
    if (weight.nnz_col % 16 == 0) {
        for (int i : tensor_algos) {
            auto algo = static_cast<cublasGemmAlgo_t>(i);
            auto time = test(in_size, out_size, seq_len, OUT.get(), IN.get(),
                             workspace.get(), weight, algo);
            res.push_back(time);
        }
    }
    // printf("M = %d\tN = %d\tK = %d\tsparsity = %f\tTime: %lf(us)\n", seq_len,
    //        out_size, in_size, sparsity,
    //        *std::min_element(res.begin(), res.end()));
    printf("%lf,", *std::min_element(res.begin(), res.end()));
}