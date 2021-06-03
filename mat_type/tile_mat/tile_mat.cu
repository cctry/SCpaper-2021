#include "tile_mat.h"
#include <Coord.h>
#include <algorithm>
#include <numeric>
#include <utility>

tile_mat tile_mat::gen_sparse_mat(int nrow, int ncol, float sparsity) {
    constexpr int tile_h = 16, tile_w = 16;
    const int blk_row_num = (nrow / tile_h);
    const int blk_col_num = (ncol / tile_w);
    const int total_tile = blk_col_num * blk_row_num;
    const int ntile = total_tile * (1.0 - sparsity);
    culib::CUDA_ptr<half> data(tile_h * tile_w * ntile, half_one);
    std::vector<Coord> tile_id(total_tile, Coord(blk_col_num));
    std::iota(tile_id.begin(), tile_id.end(), Coord(blk_col_num));
    std::random_shuffle(tile_id.begin(), tile_id.end());
    tile_id.erase(tile_id.begin() + ntile, tile_id.end());
    int offset = 0;
    std::vector<int> row_ptr = {offset};
    std::vector<int> row_offset;
    for (int r = 0; r < blk_row_num; r++) {
        std::vector<Coord> tiles;
        std::copy_if(tile_id.begin(), tile_id.end(), std::back_inserter(tiles),
                     [r](const Coord &A) { return A.y() == r; });
        offset += tiles.size();
        row_ptr.push_back(offset);
        std::vector<int> col_id;
        std::transform(tiles.begin(), tiles.end(), std::back_inserter(col_id),
                       [](const Coord &A) { return A.x(); });
        row_offset.insert(row_offset.end(), col_id.begin(), col_id.end());
    }
    assert(row_ptr.size() == blk_row_num + 1);
    assert(row_ptr.back() == ntile);
    assert(row_offset.size() == ntile);
    culib::CUDA_ptr<int> _row_ptr(row_ptr);
    culib::CUDA_ptr<int> _row_offset(row_offset);
    tile_mat W(std::move(_row_ptr), std::move(_row_offset), std::move(data),
               blk_col_num);
    return W;
}

tile_mat tile_mat::gen_dense_mat(int nrow, int ncol) {
    const auto num_blk = (nrow * ncol) / 256;
    const auto num_blk_col = ncol / 16;
    const auto num_blk_row = nrow / 16;
    std::vector<half> data(num_blk * 256, half_one);
    std::vector<int> row_ptr(num_blk_row + 1), row_offset(num_blk);
    std::iota(row_ptr.begin(), row_ptr.end(), 0);
    std::for_each(row_ptr.begin(), row_ptr.end(),
                  [=](int& x) { x *= num_blk_col; });
    auto iter_s = row_offset.begin();
    for (int i = 0; i < num_blk_row; i++) {
        auto iter_e = iter_s;
        std::advance(iter_e, num_blk_col);
        std::iota(iter_s, iter_e, 0);
        iter_s = iter_e;
    }
    assert(row_ptr.size() == num_blk_row + 1);
    assert(row_ptr.back() == num_blk);
    assert(row_offset.size() == num_blk);
    return tile_mat(culib::CUDA_ptr<int>(row_ptr),
                    culib::CUDA_ptr<int>(row_offset),
                    culib::CUDA_ptr<half>(data), num_blk_col);
}