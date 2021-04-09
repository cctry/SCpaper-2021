#pragma once
#include <CUDA_ptr.hpp>
#include <memory>
#include <utils.h>

class LayerNorm {
  private:
    const half eps;
    const int size;
    culib::CUDA_ptr<half> weight;
    culib::CUDA_ptr<half> bias;

  public:
    LayerNorm(const half *w, const half *b, int _size);
    void forward(half *out, const half *src, int num) const;
};



std::unique_ptr<LayerNorm> gen_LN(int size);