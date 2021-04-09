#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <utils.h>
#include "Encoder.h"
namespace cg = cooperative_groups;

namespace GELU {
__device__ inline float tanh(const float &x) { return tanhf(x); }

__device__ inline half tanh(const half &x) {
    const float tmp = tanhf(__half2float(x));
    return __float2half(tmp);
}

__device__ inline half2 tanh(const half2 &x) {
    // at the moment, there is no half2 tanh builtin
    float2 tmp = (__half22float2(x));
    tmp.x = tanhf(tmp.x);
    tmp.y = tanhf(tmp.y);
    return __float22half2_rn(tmp);
}

// constants for approximating the normal cdf
constexpr float A = 0.5f;
constexpr float B = 0.7978845608028654f;   // sqrt(2.0/M_PI)
constexpr float C = 0.035677408136300125f; // 0.044715 * sqrt(2.0/M_PI)

template <typename T, unsigned TPB>
__global__ void geluKernel(const T a, const T b, const T c, int n,
                           const T *input, T *output) {
    const int idx = blockIdx.x * TPB + threadIdx.x;

    if (idx < n) {
        const T in = input[idx];
        const T cdf = a + a * tanh(in * (c * in * in + b));
        output[idx] = in * cdf;
    }
}

void computeGelu(cudaStream_t stream, int n, const float *input,
                 float *output) {
    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    geluKernel<float, blockSize>
        <<<gridSize, blockSize, 0, stream>>>(A, B, C, n, input, output);
}

void computeGelu(cudaStream_t stream, int n, const half *input, half *output) {
    constexpr int blockSize = 256;

    if (0 == (n & 1)) {
        const int n2 = n / 2;

        const int gridSize = (n2 + blockSize - 1) / blockSize;
        const half2 A2 = __floats2half2_rn(A, A);
        const half2 B2 = __floats2half2_rn(B, B);
        const half2 C2 = __floats2half2_rn(C, C);
        const half2 *input2 = reinterpret_cast<const half2 *>(input);
        half2 *output2 = reinterpret_cast<half2 *>(output);
        geluKernel<half2, blockSize><<<gridSize, blockSize, 0, stream>>>(
            A2, B2, C2, n2, input2, output2);
    } else {
        const int gridSize = (n + blockSize - 1) / blockSize;
        geluKernel<half, blockSize>
            <<<gridSize, blockSize, 0, stream>>>(A, B, C, n, input, output);
    }
}

} // namespace GELU
