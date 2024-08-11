project('x86-simd-sort-modified', 'cpp',
        version : '5.0.x',
        license : 'BSD 3-clause',
        default_options : ['cpp_std=c++17'])
fs = import('fs')
cpp = meson.get_compiler('cpp')
src = include_directories('src')
lib = include_directories('lib')
utils = include_directories('utils')


fp16code = '''#include<immintrin.h>
int main() {
  __m512h temp = _mm512_set1_ph(1.0f);
  __m512h var2 = _mm512_min_ph(temp, temp);
  return 0;
}
'''
cancompilefp16 = cpp.compiles(fp16code, args:'-march=sapphirerapids')

subdir('lib')
libsimdsort = shared_library('x86simdsortcpp',
                             'lib/x86simdsort.cpp',
                             include_directories : [src, utils, lib],
                             link_args : [openmpflags],
                             link_with : [libtargets],
                             gnu_symbol_visibility : 'inlineshidden',
                             install : true,
                             soversion : 1,
                            )

pkg_mod = import('pkgconfig')
pkg_mod.generate(libraries : libsimdsort,
                 version : '4.0',
                 name : 'libx86simdsortcpp',
                 filebase : 'x86simdsortcpp',
                 description : 'C++ template library for high performance SIMD based sorting routines.')

summary({
  'Can compile AVX-512 FP16 ISA': cancompilefp16,
  },
  section: 'Configuration',
  bool_yn: true
  )
