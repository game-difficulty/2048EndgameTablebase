#pragma once

#include <cstddef>
#include <cstdint>

namespace CanonicalBatch {

void canonicalize_by_mode(
    const uint64_t *src,
    uint64_t *dst,
    size_t count,
    int symm_mode
);

void canonicalize_inplace(
    uint64_t *data,
    size_t count,
    int symm_mode
);

const char *backend_name();

} // namespace CanonicalBatch
