libtargets = []

libtargets += static_library('libavx',
  files('x86simdsort-avx2.cpp'),
  include_directories : [src],
  cpp_args : cpp.get_id() == 'msvc' ? ['/arch:AVX2'] : ['-march=haswell'],
  gnu_symbol_visibility : 'inlineshidden',
  dependencies: [omp_dep],
)

libtargets += static_library('libskx',
  files('x86simdsort-skx.cpp'),
  include_directories : [src],
  cpp_args : cpp.get_id() == 'msvc' ? ['/arch:AVX512'] : ['-march=skylake-avx512'],
  gnu_symbol_visibility : 'inlineshidden',
  dependencies: [omp_dep],
)

install_headers('x86simdsort.h')
