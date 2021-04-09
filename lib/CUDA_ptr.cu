#include "CUDA_ptr.cuh"
#include "CUDA_ptr.hpp"
namespace _private {
template <>
__global__ void __kernel_setVal<half>(half *_ptr, const std::size_t _size,
                                      const half val) {
    auto ptr = reinterpret_cast<half2 *>(_ptr);
    const half2 val2{val, val};
    for (auto i : culib::grid_stride_range(std::size_t(0), _size / 2))
        ptr[i] = val2;
}

}