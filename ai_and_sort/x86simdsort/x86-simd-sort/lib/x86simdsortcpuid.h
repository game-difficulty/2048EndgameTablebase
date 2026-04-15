#ifndef X86SIMDSORT_CPUID_H
#define X86SIMDSORT_CPUID_H

// 如果你百分之百确定只使用 MinGW/GCC 编译，可以精简为：
#ifdef _MSC_VER
    // 如果以后需要用 MSVC 编译，可以在这里保留最简逻辑
    // 但目前你的环境是 MinGW，直接跳过
#else
    #include <cpuid.h> 
    #define xss_cpu_init() __builtin_cpu_init()
    #define xss_cpu_supports(feature) __builtin_cpu_supports(feature)
#endif 

#endif // X86SIMDSORT_CPUID_H