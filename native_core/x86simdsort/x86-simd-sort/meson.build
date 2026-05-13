project('x86-simd-sort', 'cpp',
        version : '7.0.x',
        license : 'BSD 3-clause',
        default_options : ['cpp_std=c++17'])

cpp = meson.get_compiler('cpp')

src = include_directories('src')
lib = include_directories('lib')
utils = include_directories('utils')

if cpp.get_id() != 'msvc'
  if not cpp.has_argument('-march=haswell') or not cpp.has_argument('-march=skylake-avx512')
    error('Compiler does not support -march=haswell or -march=skylake-avx512.')
  endif
endif

omp_dep = []
if get_option('use_openmp')
  omp = dependency('openmp', required : true)
  omp_dep = declare_dependency(dependencies: omp, compile_args: ['-DXSS_USE_OPENMP'])
endif

subdir('lib')

libsimdsort = static_library('x86simdsortcpp',
                             'lib/x86simdsort.cpp',
                             include_directories : [src, utils, lib],
                             link_with : [libtargets],
                             dependencies: [omp_dep],
                             gnu_symbol_visibility : 'inlineshidden',
                             install : true,
                             pic: true,
                            )

x86simdsortcpp_dep = declare_dependency(
  include_directories: include_directories('lib'),
  link_with: libsimdsort,
)

summary({
  'Build uint32/uint64 qsort+argsort only': true,
  'OpenMP support': get_option('use_openmp'),
  },
  section: 'Configuration',
  bool_yn: true
  )
