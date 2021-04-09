#pragma once
#include <memory>
namespace culib {
template <typename T> using arr_t = std::unique_ptr<T[]>;

template <typename T> arr_t<T> new_arr(size_t num) {
    auto res = arr_t<T>(new T[num]);
    return std::move(res);
}
} // namespace culib