#include <array>
#include <cstdio>
#include <iostream>
#include <type_traits>
#include <utility>

template <template <std::size_t, typename> class F, typename T, std::size_t I,
          std::size_t... Is>
struct LoopExecutor : public LoopExecutor<F, T, Is...> {
    using func_type = F<I, T>;
    func_type mfunc;
    template <typename Op> T run(const Op &op) {
        T temp = LoopExecutor<F, T, Is...>::template run<Op>(op);
        return op(temp, mfunc());
    }
};

template <template <std::size_t, typename> class F, typename T>
struct LoopExecutor<F, T, 0> {
    using func_type = F<0, T>;
    func_type mfunc;
    template <typename Op> T run(const Op &op) { return mfunc(); }
};

template <template <std::size_t, typename> class F, std::size_t I,
          std::size_t... Is>
struct LoopExecutor<F, void, I, Is...> : public LoopExecutor<F, void, Is...> {
    using func_type = F<I, void>;
    func_type mfunc;
    void run() {
        LoopExecutor<F, void, Is...>::run();
        mfunc();
    }
};

template <template <std::size_t, typename> class F>
struct LoopExecutor<F, void, 0> {
    using func_type = F<0, void>;
    func_type mfunc;
    void run() { mfunc(); }
};

// int work_impl(std::size_t N) { std::cout << N << std::endl; }

// template <std::size_t N> int work() { return work_impl(N); }

// template <std::size_t N, typename T> struct Func_call {
//     T operator()() { work<N>(); }
// };

template <std::size_t... Is>
constexpr auto indexSequenceReverse(std::index_sequence<Is...> const &)
    -> decltype(std::index_sequence<(sizeof...(Is) - 1U) - Is...>{});

template <std::size_t N>
using makeIndexSequenceReverse =
    decltype(indexSequenceReverse(std::make_index_sequence<N>{}));

template <template <std::size_t, typename> class F, typename T,
          std::size_t... I>
constexpr auto make_loop_impl(std::index_sequence<I...>) {
    return LoopExecutor<F, T, I...>();
}

template <template <std::size_t, typename> class F, typename T, int N>
constexpr auto make_loop() {
    static_assert(N > 0);
    using seq_t = makeIndexSequenceReverse<N>;
    return make_loop_impl<F, T>(seq_t{});
}

// int main() {
//     constexpr int n = 5;
//     // LoopExecutor<Func_call, void, 2, 1, 0> loop;
//     // loop.run();

//     auto loop = make_loop<Func_call, void, n>();
//     loop.run();
// }
