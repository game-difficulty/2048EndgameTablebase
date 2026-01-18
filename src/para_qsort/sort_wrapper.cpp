#include <iostream>
#include <vector>
#include <cstdint>
#include "x86simdsort.h"

// C++ wrapper function for sorting uint64_t array
extern "C" void sort_uint64(uint64_t *arr, size_t arrsize, bool descending = false) {
    x86simdsort::qsort<uint64_t>(arr, arrsize, false, descending);
}
