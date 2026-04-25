libtargets = []

# 1. 编译 AVX2 目标：9950X 的兜底高性能路径
libtargets += static_library('libavx',
  files('x86simdsort-avx2.cpp'),
  include_directories : [src],
  # 保持 haswell 架构足以开启 AVX2
  cpp_args : cpp.get_id() == 'msvc' ? ['/arch:AVX2'] : ['-march=haswell'],
  gnu_symbol_visibility : 'inlineshidden',
  dependencies: [omp_dep],
)

# 2. 编译 SKX 目标：9950X 的核心 AVX-512 路径
libtargets += static_library('libskx',
  files('x86simdsort-skx.cpp'),
  include_directories : [src],
  # skylake-avx512 是最稳妥的开启 AVX-512F/DQ/BW/VL 的方式
  cpp_args : cpp.get_id() == 'msvc' ? ['/arch:AVX512'] : ['-march=skylake-avx512'],
  gnu_symbol_visibility : 'inlineshidden',
  dependencies: [omp_dep],
)

# 3. 移除 libicl 和 libspr
# 虽然你清空了 .cpp 文件，但在这里直接移除编译目标可以彻底避免编译器
# 检查是否支持 -march=icelake-client 等参数，提高编译兼容性。

install_headers('x86simdsort.h')