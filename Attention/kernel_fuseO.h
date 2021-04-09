#pragma once
#include <utils.h>
#include <memory>
#include "../Model.h"

void Multihead_atttion_fuseO(half *Q, half *K, half *V, half *mask, half *mat_z,
                             int *O_row_ptr, int *O_row_offset, half *O_data,
                             half *bias, std::shared_ptr<Model_t> model,
                             half *out);
