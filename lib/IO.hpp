#pragma once
#include <vector>
#include <fstream>
template <typename T> std::vector<T> load_vec(std::ifstream &f, size_t num) {
    std::vector<T> res(num);
    auto buffer = reinterpret_cast<char *>(res.data());
    f.read(buffer, num * sizeof(T));
    return res;
}
