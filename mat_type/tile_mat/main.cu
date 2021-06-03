__global__ void __kernel_dense2blk(const half *__restrict__ dense, half *__restrict__ blk,
                                   const int ncol) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto warp_id = tid / 32;
    const auto blk_col_num = ncol / 16;
    const auto row_blk = warp_id / blk_col_num;
    const auto col_blk = warp_id % blk_col_num;
    cmat_t frag;
    const auto src = get_blk_start(dense, row_blk, col_blk, ncol);
    wmma::load_matrix_sync(frag, src, ncol, wmma::mem_row_major);
    const auto dst = &blk[(blk_col_num * row_blk + col_blk) * 256];
    wmma::store_matrix_sync(dst, frag, 16, wmma::mem_row_major);
}
