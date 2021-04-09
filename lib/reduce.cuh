#pragma once
template <typename T> __device__ T warp_max(const T *data, const int len) {
    const auto lane_id = threadIdx.x % 32;
    T temp;
    if (len >= 32) {
        temp = data[lane_id];
        for (int i = lane_id + 32; i < len; i += 32)
            temp = temp > data[i] ? temp : data[i];
    } else {
        temp = lane_id < len ? data[lane_id] : data[0];
    }
    auto lmax = temp;
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        temp = __shfl_down_sync(0xFFFFFFFF, lmax, offset);
        lmax = lmax > temp ? lmax : temp;
    }
    return __shfl_sync(0xFFFFFFFF, lmax, 0);
}

template <typename T> __device__ T warp_sum(const T *data, const int len) {
    const auto lane_id = threadIdx.x % 32;
    const T zero = 0;
    T temp;
    if (len >= 32) {
        temp = data[lane_id];
        for (int i = lane_id + 32; i < len; i += 32)
            temp += data[i];
    } else {
        temp = lane_id < len ? data[lane_id] : zero;
    }
    for (int offset = 32 / 2; offset > 0; offset /= 2)
        temp += __shfl_down_sync(0xFFFFFFFF, temp, offset);
    return __shfl_sync(0xFFFFFFFF, temp, 0);
}

// size must be power of 2
template <typename T> __device__ T block_sum(T *data, const int size) {
    for (auto s = size / 2; s > 0; s >>= 1) {
        for (int i = threadIdx.x; i < s; i += blockDim.x)
            data[i] += data[i + s];
        __syncthreads();
    }
    return data[0];
}

// size must be power of 2
template <typename T> __device__ T block_max(T *data, const int size) {
    for (auto s = size / 2; s > 0; s >>= 1) {
        for (int i = threadIdx.x; i < s; i += blockDim.x)
            data[i] = data[i + s] > data[i] ? data[i + s] : data[i];
        __syncthreads();
    }
    return data[0];
}