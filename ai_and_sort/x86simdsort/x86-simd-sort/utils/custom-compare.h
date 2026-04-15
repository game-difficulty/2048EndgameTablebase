#ifndef UTILS_CUSTOM_COMPARE
#define UTILS_CUSTOM_COMPARE

#include <limits>
template <typename T, typename Comparator>
struct compare {
    static constexpr auto op = Comparator {};
    bool operator()(const T a, const T b)
    {
        return op(a, b);
    }
};

template <typename T, typename Comparator>
struct compare_arg {
    compare_arg(const T *arr)
    {
        this->arr = arr;
    }
    bool operator()(const int64_t a, const int64_t b)
    {
        return compare<T, Comparator>()(arr[a], arr[b]);
    }
    const T *arr;
};

#endif // UTILS_CUSTOM_COMPARE