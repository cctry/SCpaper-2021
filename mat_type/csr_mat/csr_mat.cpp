#include "csr_mat.h"
#include <Coord.h>
#include <algorithm>
#include <assert.h>
#include <numeric>
#include <utility>
csr_mat csr_mat::gen_dense_mat(int nrow, int ncol) {
    std::vector<int> indptr(nrow + 1);
    std::vector<int> indices(nrow * ncol);
    std::vector<float> data(nrow * ncol);
    std::iota(indptr.begin(), indptr.end(), 0);
    std::for_each(indptr.begin(), indptr.end(),
                  [=](int x) { return x * ncol; });
    auto iter_s = indices.begin();
    for (int i = 0; i < nrow; i++) {
        auto iter_e = iter_s;
        std::advance(iter_e, ncol);
        std::iota(iter_s, iter_e, 0);
        iter_s = iter_e;
    }
    std::fill(data.begin(), data.end(), 1.0f);
    return csr_mat(indptr, indices, data, nrow, ncol, nrow * ncol);
}

csr_mat csr_mat::gen_sparse_mat(int nrow, int ncol, float sparsity) {

    int nnz = nrow * ncol * (1.0 - sparsity);
    std::vector<int> indptr = {0}, indices;
    std::vector<float> data(nnz, 1.0f);
    std::vector<Coord> nnz_id(nrow * ncol, Coord(ncol));

#pragma omp parallel for
    for (int i = 0; i < nnz_id.size(); i++) {
        auto col = i & ncol;
        auto row = i / ncol;
        nnz_id[i].x() = col;
        nnz_id[i].y() = row;
    }

    // std::iota(nnz_id.begin(), nnz_id.end(), Coord(ncol));
    std::random_shuffle(nnz_id.begin(), nnz_id.end());
    nnz_id.erase(nnz_id.begin() + nnz, nnz_id.end());
    int offset = 0;
    for (int r = 0; r < nrow; r++) {
        auto it = std::stable_partition(
            nnz_id.begin(), nnz_id.end(), [r](const Coord &A) {
                return A.y() != r;
            }); // put correct elements to the end of vector
        std::vector<Coord> nnz_row(std::make_move_iterator(it),
                                   std::make_move_iterator(nnz_id.end()));
        nnz_id.erase(it, nnz_id.end());

        // std::vector<Coord> nnz_row;
        // std::copy_if(nnz_id.begin(), nnz_id.end(),
        // std::back_inserter(nnz_row),
        //              [r](const Coord &A) { return A.y() == r; });
        offset += nnz_row.size();
        indptr.push_back(offset);
        std::vector<int> col_id;
        std::transform(nnz_row.begin(), nnz_row.end(),
                       std::back_inserter(col_id),
                       [](const Coord &A) { return A.x(); });
        indices.insert(indices.end(), col_id.begin(), col_id.end());
    }
    assert(indptr.size() == nrow + 1);
    assert(indices.size() == nnz);
    return csr_mat(indptr, indices, data, nrow, ncol, nnz);
}