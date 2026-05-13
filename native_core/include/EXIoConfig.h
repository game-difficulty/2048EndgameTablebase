#pragma once

#include "FileIOUtils.h"

inline FileIOUtils::DirectIoConfig zmask_io_config_from_options(const RunOptions &options) {
    FileIOUtils::DirectIoConfig config = FileIOUtils::direct_io_config_from_options(options);
#ifdef _WIN32
    config.enabled = false;
#endif
    return config;
}
