#pragma once
// #include <CUDA_ptr.hpp>
#include <IO.hpp>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
// #include <utils.h>
#include <vector>

struct csr_mat {
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<float> data;
    int nrow;
    int ncol;
    int nnz;
    csr_mat() = delete;
    csr_mat(const std::string &path) {
        std::ifstream fs(path);
        char buffer[12];
        fs.read(buffer, 12);
        auto header = reinterpret_cast<uint32_t *>(buffer);
        // printf("nrow: %u\tncol: %u\tnnz: %u\n", header[0], header[1],
        //        header[2]);
        nrow = header[0];
        ncol = header[1];
        nnz = header[2];
        indptr = load_vec<int>(fs, 1 + nrow);
        indices = load_vec<int>(fs, nnz);
        data = load_vec<float>(fs, nnz);
        indptr.shrink_to_fit();
        indices.shrink_to_fit();
        data.shrink_to_fit();
    };
    csr_mat(std::vector<int> _indptr, std::vector<int> _indices,
            std::vector<float> _data, int _nrow, int _ncol, int _nnz)
        : indptr(_indptr), indices(_indices), data(_data), nrow(_nrow),
          ncol(_ncol), nnz(_nnz) {}
    static csr_mat gen_dense_mat(int nrow, int ncol);
    static csr_mat gen_sparse_mat(int nrow, int ncol, float sparsity);
};