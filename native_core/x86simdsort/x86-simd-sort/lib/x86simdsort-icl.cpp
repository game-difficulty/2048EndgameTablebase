// ICL specific routines:
// 由于目标仅保留 uint64_t 排序，此文件处理的 16-bit 类型代码已全部移除。

#include "x86simdsort-internal.h"

namespace xss {
namespace avx512 {
    // 为空，不进行 uint16_t 和 int16_t 的实例化
} 
namespace fp16_icl {
    // 为空，不进行 _Float16 的实例化
}
} // namespace xss