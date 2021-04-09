#include "../../Linear/Linear.h"
#include "../../Model.h"
#include "../Fused_attention.h"
#include "../Linear_VO.h"
#include <CUDA_ptr.hpp>
#include <SkipLayerNorm/SkipLayerNorm.h>
#include <bits/stdc++.h>
#include <wtime.h>
using namespace culib;
#define TIME
double test_fused_NOTF(CUDA_ptr<half> &output, CUDA_ptr<half> &input,
                       std::shared_ptr<Model_t> para, float sparsity,
                       int nlayer);

class encoder {
    using attn_config = Attention_config<tile_mat, tile_mat, tile_mat, tile_mat,
                                         NOTF_fused_prune_attn>;
    using attn_t = FusedAttention<attn_config>;

  private:
    std::unique_ptr<attn_t> attn;
    std::unique_ptr<SkipLayerNorm> LN1;
    std::unique_ptr<SkipLayerNorm> LN2;
    std::unique_ptr<Linear<tile_mat>> linear1;
    std::unique_ptr<Linear<tile_mat>> linear2;
    std::shared_ptr<Model_t> model;

  public:
    culib::CUDA_ptr<half> temp_linear;
    culib::CUDA_ptr<half> temp_attn;
    culib::CUDA_ptr<half> temp_LN;

    encoder(std::shared_ptr<Model_t> _model, float sparsity)
        : model(_model), temp_attn(model->seq_len * model->emdim),
          temp_linear(model->seq_len * model->dimFF),
          temp_LN(model->seq_len * model->emdim) {
        CUDA_ptr<half> biasO(model->emdim);
        auto LQ = gen_sparse_linear<tile_mat>(model->kdim, model->emdim,
                                              model->seq_len, sparsity);
        auto LK = gen_sparse_linear<tile_mat>(model->kdim, model->emdim,
                                              model->seq_len, sparsity);
        // build LVO
        auto weight = col_mat::gen_sparse_mat(model->nhead * model->emdim,
                                              model->emdim, sparsity);
        CUDA_ptr<half> bias(model->nhead * weight.col_id->size,
                            __float2half_rn(0.2));
        auto LVO = std::make_unique<Linear_VO_prune>(weight, bias, model);
        attn = std::make_unique<attn_t>(std::move(LQ), std::move(LK),
                                        std::move(LVO), biasO, model);
        LN1 = gen_skLN(model->emdim);
        LN2 = gen_skLN(model->emdim);

        linear1 = gen_sparse_linear<tile_mat>(model->dimFF, model->emdim,
                                              model->seq_len, sparsity);
        linear2 = gen_sparse_linear<tile_mat>(model->emdim, model->dimFF,
                                              model->seq_len, sparsity);
    }

    void forward(half *out, half *src) {
        attn->forward(temp_attn.get(), src, src, src);
        LN1->forward(temp_LN.get(), temp_attn.get(), src, model->seq_len);
        linear1->forward(temp_linear.get(), temp_LN.get());
        linear2->forward(temp_attn.get(), temp_linear.get());
        LN2->forward(out, temp_attn.get(), temp_LN.get(), model->seq_len);
    }
};

int main(int ac, char **av) {
    std::srand(time(NULL));
    if (ac < 3) {
        puts("./a.out nlayer sparsity...");
    }
    int nlayer = std::atoi(av[1]);
    std::vector<float> sparsities;
    for (int i = 2; i < ac; i++) {
        sparsities.push_back(std::atof(av[i]));
    }
    auto para =
        std::make_shared<Model_t>(Model_t{768, 768, 128, 12, 768 * 4, 768});
    CUDA_ptr<half> input(para->seq_len * para->emdim, half_one);
    CUDA_ptr<half> output(para->seq_len * para->emdim);
    std::vector<double> res_time;
    for (auto sp : sparsities) {
        res_time.push_back(test_fused_NOTF(output, input, para, sp, nlayer));
    }
    puts("res_time: ");
    for (auto r : res_time) {
        std::cout << " & " << r << " ";
    }
    printf("& \\textbf{");
    std::cout << std::accumulate(res_time.begin(), res_time.end(), 0.0) /
                     res_time.size()
              << "}" << std::endl;

}

double test_fused_NOTF(CUDA_ptr<half> &output, CUDA_ptr<half> &input,
                       std::shared_ptr<Model_t> para, float sparsity,
                       int nlayer) {
    std::vector<std::unique_ptr<encoder>> model(nlayer);
    for (size_t i = 0; i < nlayer; i++) {
        model[i] = std::make_unique<encoder>(para, sparsity);
    }
    for (size_t i = 0; i < nlayer; i++) {
        model[i]->forward(output.get(), input.get());
    }
    std::vector<half> h_res(para->emdim * para->seq_len);
    cudaChk(cudaDeviceSynchronize());
    double time = wtime_new(
        10,
        [&]() {
            for (size_t i = 0; i < nlayer; i++) {
                model[i]->forward(output.get(), input.get());
            }
        },
        [&]() { cudaChk(cudaDeviceSynchronize()); });
    output.dump(h_res.data());
    output.clear();
    return time;
}
