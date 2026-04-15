project('x86-simd-sort', 'cpp',
        version : '7.0.x',
        license : 'BSD 3-clause',
        default_options : ['cpp_std=c++17'])

cpp = meson.get_compiler('cpp')

# 包含路径定义
src = include_directories('src')
lib = include_directories('lib')
utils = include_directories('utils')

# 简化编译器检查：
# 既然你有 9950X，且使用的是 MinGW，我们只检查基础的 AVX2 和 AVX-512 支持
if cpp.get_id() != 'msvc'
  if not cpp.has_argument('-march=haswell') or not cpp.has_argument('-march=skylake-avx512')
    error('Compiler does not support -march=haswell or -march=skylake-avx512.')
  endif
endif

# OpenMP 支持：保持现状，以便在 9950X 上进行并行排序
omp_dep = []
if get_option('use_openmp')
  omp = dependency('openmp', required : true)
  omp_dep = declare_dependency(dependencies: omp, compile_args: ['-DXSS_USE_OPENMP'])
endif

# 移除 fp16 编译检测逻辑，因为我们已经删除了相关的 _Float16 代码
cancompilefp16 = false

# 进入 lib 目录编译我们的精简版 libtargets
subdir('lib')

# 核心库构建：
# 根据之前的讨论，我们强制使用 static_library 以实现单 DLL 封装
libsimdsort = static_library('x86simdsortcpp',
                             'lib/x86simdsort.cpp',
                             include_directories : [src, utils, lib],
                             link_with : [libtargets],
                             dependencies: [omp_dep],
                             gnu_symbol_visibility : 'inlineshidden',
                             install : true,
                             pic: true,
                            )

# 定义依赖项，方便后续 Wrapper 链接
x86simdsortcpp_dep = declare_dependency(
  include_directories: include_directories('lib'),
  link_with: libsimdsort,
)

# 总结报告
summary({
  'Build uint64 only': true,
  'OpenMP support': get_option('use_openmp'),
  },
  section: 'Configuration',
  bool_yn: true
  )