#include "Linear.h"
#include <cstdio>
#define CHECK_CUDA(func)                                                       \
    {                                                                          \
        cudaError_t status = (func);                                           \
        if (status != cudaSuccess) {                                           \
            printf("CUDA API failed at line %d with error: %s (%d)\n",         \
                   __LINE__, cudaGetErrorString(status), status);              \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    }

#define CHECK_CUSPARSE(func)                                                   \
    {                                                                          \
        cusparseStatus_t status = (func);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n",     \
                   __LINE__, cusparseGetErrorString(status), status);          \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    }

void Linear<csr_mat>::forward(half *output, half *input,
                              cudaStream_t stream) {
    cusparseSetStream(handle, stream);
    
    auto bias_temp = this->bias.get();
    auto stride = out_size;
    const auto add_bias = [bias_temp, stride] __device__(half * data,
                                                         int i) -> half {
        return data[i] + bias_temp[i % stride];
    }; // unpruned bias
    static const half alpha = half_one, beta = half_zero;
    cusparseCreateDnMat(&denmat, in_size, size, in_size, input, CUDA_R_16F,
                        CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&resmat, out_size, size, out_size, output, CUDA_R_16F,
                        CUSPARSE_ORDER_COL);
    size_t bufferSize = 0;
    cusparseSpMM_bufferSize(handle, opA, opB, &alpha, spWeight, denmat, &beta,
                            resmat, CUDA_R_16F, CUSPARSE_SPMM_CSR_ALG2,
                            &bufferSize);
    culib::CUDA_ptr<half> workspace(bufferSize);

    cusparseSpMM(handle, opA, opB, &alpha, spWeight, denmat, &beta, resmat,
                 CUDA_R_16F, CUSPARSE_SPMM_CSR_ALG2, workspace.get());

    culib::cuda_map(output, size * out_size, add_bias, stream);
    cusparseDestroyDnMat(denmat);
    cusparseDestroyDnMat(resmat);
}

Linear<csr_mat>::~Linear() {
    cusparseDestroy(handle);
    cusparseDestroySpMat(spWeight);
    delete indptr;
    delete indices;
    delete data;
}

Linear<csr_mat>::Linear(int _in_size, int _out_size, const csr_mat &w,
                        const half *b, int _size)
    : bias(b, _out_size), in_size(_in_size), out_size(_out_size), size(_size) {
    cusparseCreate(&handle);
    indptr = new culib::CUDA_ptr<int>(w.indptr);
    indices = new culib::CUDA_ptr<int>(w.indices);
    data = new culib::CUDA_ptr<half>(w.data.size());
    culib::CUDA_ptr<float> data_f(w.data);
    culib::util::to_half_devptr(data->get(), data_f.get(), w.data.size());
    cusparseCreateCsr(&spWeight, w.nrow, w.ncol, w.nnz, indptr->get(),
                      indices->get(), data->get(), CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
}

Linear<csr_mat>::Linear(Linear<csr_mat> &&_linear)
    : in_size(_linear.in_size), out_size(_linear.out_size), size(_linear.size),
      bias(std::move(_linear.bias)), indptr(_linear.indptr),
      indices(_linear.indices), data(_linear.data) {
    cusparseCreate(&handle);
    cusparseCreateCsr(&spWeight, out_size, in_size, data->size, indptr->get(),
                      indices->get(), data->get(), CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
}
