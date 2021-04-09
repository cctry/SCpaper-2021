#pragma once
#include <utility>
struct Coord {
    int stride;
    std::pair<int, int> value;
    Coord operator++() {
        if (x() < stride - 1) {
            x()++;
        } else {
            x() = 0;
            y()++;
        }
        return *this;
    };
    int &x() { return value.first; };
    int &y() { return value.second; };
    const int &x() const { return value.first; };
    const int &y() const { return value.second; };
    Coord(int _stride) : stride(_stride), value{0, 0} {};
    Coord(int _stride, int _x, int _y) : stride(_stride), value{_x, _y} {};
};
