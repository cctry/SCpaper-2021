#pragma once
#include "range.hpp"
#include <chrono>
template <typename _Action, typename _Prep>
double wtime(const int times, _Action action, _Prep prep) {
    double time = 0.0;
    for (int i : culib::range(0, times)) {
        prep();
        auto start = std::chrono::high_resolution_clock::now();
        action();
        auto end = std::chrono::high_resolution_clock::now();
        time +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count() *
            1.0;
    }

    return time / times;
}

template <typename _Action, typename _End>
double wtime_new(const int times, _Action action, _End end) {
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i : culib::range(0, times)) {
        action();
    }
    end();
    auto end_time = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - start_time)
                      .count() *
                  1.0;

    return time / times;
}