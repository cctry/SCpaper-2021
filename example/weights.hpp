#pragma once
#include <algorithm>
#include <cassert>
#include <cnpy.h>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
class Weights {
  public:
    template <typename T>
    std::vector<T> load_mat(const std::string &name,
                            const std::string &npz_name) {
        const cnpy::npz_t &npz = weight_files.at(npz_name);
        auto arr = npz.at(name);
        const auto size = std::accumulate(arr.shape.begin(), arr.shape.end(), 1,
                                          std::multiplies<size_t>());
        assert(arr.word_size == sizeof(T));
        auto ptr = arr.data<T>();
        std::vector<T> res(size);
        memcpy(res.data(), ptr, sizeof(T) * size);
        return std::move(res);
    }
    std::vector<std::string> layer_names;
    std::string path;
    std::unordered_map<std::string, cnpy::npz_t> weight_files;
    // cnpy::npz_t load_npz(const std::string &name) {
    //     return cnpy::npz_load(path + name);
    // }
    Weights(const std::string &filename, const std::string &_path)
        : path(_path) {
        std::ifstream fin;
        fin.open(filename);
        std::string str;
        while (!fin.eof()) {
            std::getline(fin, str);
            if (str.length() > 0)
                layer_names.push_back(str);
        }
        fin.close();
        for (auto &name : layer_names) {
            weight_files.emplace(
                std::make_pair(name, cnpy::npz_load(path + name)));
        }
    }
};