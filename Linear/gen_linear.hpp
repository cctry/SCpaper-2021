template <typename T>
std::unique_ptr<Linear<T>> gen_dense_linear(int nrow, int ncol, int size) {
    auto W = T::gen_dense_mat(nrow, ncol);
    std::vector<half> bias(nrow, __float2half_rn(0.5));
    auto res = std::make_unique<Linear<T>>(ncol, nrow, std::move(W),
                                           bias.data(), size);
    return std::move(res);
}

template <>
std::unique_ptr<Linear<base_mat>>
gen_dense_linear<base_mat>(int nrow, int ncol, int size);

template <typename T>
std::unique_ptr<Linear<T>> gen_sparse_linear(int nrow, int ncol, int size,
                                             float sparsity) {
    // puts("pass");
    auto W = T::gen_sparse_mat(nrow, ncol, sparsity);
    std::vector<half> bias(nrow, __float2half_rn(0.5));
    auto res = std::make_unique<Linear<T>>(ncol, nrow, std::move(W),
                                           bias.data(), size);
    return std::move(res);
}

template <>
std::unique_ptr<Linear<base_mat>>
gen_sparse_linear<base_mat>(int nrow, int ncol, int size,
                                         float sparsity);