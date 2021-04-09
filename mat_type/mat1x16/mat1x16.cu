#include "mat1x16.h"
#include <Coord.h>
#include <algorithm>
#include <numeric>
#include <utils.h>

constexpr int padTo16(int n) {
    int m = n / 16;
    if (m * 16 == n)
        return n;
    else
        return (m + 1) * 16;
}

mat1x16 mat1x16::gen_dense_mat(int nrow, int ncol) {
    const int tile_per_part = nrow;
    const int npart = ncol / 16;
    const int ntile = npart * nrow;
    std::vector<int> row_idx(nrow + 1); // nrow + 1
    std::iota(row_idx.begin(), row_idx.end(), 0);
    std::vector<int> part_idx; // ntile
    auto iter_s = part_idx.begin();
    for (int i = 0; i < nrow; i++) {
        auto iter_e = iter_s;
        std::advance(iter_e, npart);
        std::iota(iter_s, iter_e, 0);
        iter_s = iter_e;
    }
    int pad = padTo16(ntile);
    std::vector<half> data(pad * 16, half_one); // (ntile + pad) * 16
    assert(data.size() % 256 == 0);
    assert(row_idx.size() == nrow + 1);
    assert(row_idx.back() == ntile);
    assert(part_idx.size() == ntile);
    return mat1x16(nrow, ncol, row_idx, part_idx, data);
}

mat1x16 mat1x16::gen_sparse_mat(int nrow, int ncol, float sparsity) {
    const int npart = ncol / 16;
    const int ntile = (npart * nrow) * (1.0 - sparsity);
    std::vector<Coord> tile_pos(nrow * npart, Coord(npart));
    std::iota(tile_pos.begin(), tile_pos.end(), Coord(npart));
    std::random_shuffle(tile_pos.begin(), tile_pos.end());
    tile_pos.erase(tile_pos.begin() + ntile, tile_pos.end());
    std::vector<int> tile_idx;
    std::vector<int> row_idx;
    int pad = padTo16(ntile);
    std::vector<half> data(pad * 16, half_one); // (ntile + pad) * 16
    for (int blk_row = 0; blk_row < nrow / 16; blk_row++) {
        std::vector<int> tile_idx_temp = {0}; // idx of this row set
        for (int part_id = 0; part_id < npart; part_id++) {
            std::vector<Coord> tile_temp;
            std::copy_if(tile_pos.begin(), tile_pos.end(),
                         std::back_inserter(tile_temp), [=](Coord &c) {
                             return c.y() / 16 == blk_row &&
                                    c.x() / 16 == part_id;
                         });
            tile_idx_temp.push_back(tile_idx_temp.back() + tile_temp.size());
            std::vector<int> row_idx_temp;
            std::transform(tile_temp.begin(), tile_temp.end(),
                           std::back_inserter(row_idx_temp),
                           [&](Coord &c) { return c.x(); });
            row_idx.insert(row_idx.end(), row_idx_temp.begin(),
                           row_idx_temp.end());
        }
        tile_idx.insert(tile_idx.end(), tile_idx_temp.begin(),
                        tile_idx_temp.end());
    }
    std::partial_sum(tile_idx.begin(), tile_idx.end(), tile_idx.begin());

    assert(row_idx.size() == ntile);
    assert(tile_idx.back() == ntile);
    assert(tile_idx.size() == (npart + 1) * nrow / 16);
    return mat1x16(nrow, ncol, row_idx, tile_idx, data);
}
