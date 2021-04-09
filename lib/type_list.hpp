#pragma once
#include <type_traits>
namespace type_list {
template <std::size_t N, typename T, typename... Ts> struct type_entry {
    using type = T;
    using next = type_entry<N - 1, Ts...>;
};

template <typename T, typename... Ts> struct type_entry<0, T, Ts...> {
    using type = T;
    using next = void;
};

template <typename T, typename... Ts> struct list {
    using next = type_entry<sizeof...(Ts), T, Ts...>;
    static const auto size = sizeof...(Ts) + 1;
};

template <int Index, typename Container> struct get_type {
    using type = typename get_type<Index - 1, typename Container::next>::type;
    // TODO: out of bound check
};

template <typename Container> struct get_type<0, Container> {
    using type = typename Container::next::type;
};
} // namespace type_list