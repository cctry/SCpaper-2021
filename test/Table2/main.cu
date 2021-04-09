#include "../Attention/Attention.h"
#include "../Encoder/Encoder.h"
#include "../Linear/Linear.h"
#include "../Model.h"
#include "template.hpp"
#include <CUDA_ptr.hpp>
#include <bits/stdc++.h>
#include <wtime.h>
using namespace culib;

void prt(std::string name, const std::vector<double> &vec) {
    std::cout << name << std::endl;
    for (auto r : vec) {
        std::cout << " & " << r << " ";
    }
    printf("& \\textbf{");
    std::cout << std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size()
              << "}" << std::endl;
}

int main(int ac, char **av) {
    std::srand(time(NULL));
    if (ac < 3) {
        puts("./a.out nlayer sparsity...");
    }
    int nlayer = std::atoi(av[1]);
    std::vector<float> sparsities;
    for (size_t i = 2; i < ac; i++) {
        sparsities.push_back(std::atof(av[i]));
    }
    auto para =
        std::make_shared<Model_t>(Model_t{768, 768, 128, 12, 768 * 4, 768});
    std::vector<double> column, tile, aware1, cusp;
    for (auto sp : sparsities) {
        // column.push_back(test_uniform<col_mat>(para, sp, nlayer) / 1000);
        // cusp.push_back(test_uniform<csr_mat>(para, sp, nlayer) / 1000);
        // tile.push_back(test_uniform<tile_mat>(para, sp, nlayer) / 1000);
        aware1.push_back(test_prun_attn1(para, sp, nlayer) / 1000);
    }
    // prt("cusp", cusp);
    // prt("column", column);
    // prt("tile", tile);
    prt("aware1", aware1);
}
