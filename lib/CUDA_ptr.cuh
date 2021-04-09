#pragma once
#include "range.hpp"
#include <cuda_fp16.h>
namespace _private {
template <typename V>
__global__ void __kernel_setVal(V *_ptr, const std::size_t _size, const V val) {
    for (auto i : culib::grid_stride_range(std::size_t(0), _size))
        _ptr[i] = val;
}

template <>
__global__ void __kernel_setVal<half>(half *_ptr, const std::size_t _size,
                                      const half val);

} // namespace _private