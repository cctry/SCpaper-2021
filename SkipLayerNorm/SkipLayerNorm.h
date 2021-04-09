#pragma once
#include <CUDA_ptr.hpp>
#include <memory>
#include <utils.h>

class SkipLayerNorm {
  private:
    const half eps;
    const int size;
    culib::CUDA_ptr<half> weight;
    culib::CUDA_ptr<half> bias;

  public:
    SkipLayerNorm(const half *w, const half *b, int _size);
    void forward(half *out, const half *src, const half *skip, int num) const;
};

std::unique_ptr<SkipLayerNorm> gen_skLN(int size);