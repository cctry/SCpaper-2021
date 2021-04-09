#include "../Linear.h"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
using namespace culib;

int main() {
    constexpr int num = 128;
    constexpr int in_size = 1024;
    constexpr int out_size = 1024 * 3;
    constexpr int num_col = out_size;

    CUDA_ptr<half> cols(out_size * num_col, half_one);
    std::vector<int> h_col_id(num_col);
    std::iota(h_col_id.begin(), h_col_id.end(), 0);
    CUDA_ptr<int> col_id(h_col_id);
    row_mat W(std::move(col_id), std::move(cols), out_size, in_size);
}