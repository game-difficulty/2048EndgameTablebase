libtargets = []
srcdir = include_directories('../src')
libdir = include_directories('.')
add_project_arguments('-mavx', '-mavx512f', '-mavx512dq', '-mavx512vl', language: 'cpp')

# Add compile flags for OpenMP if enabled
openmpflags = []
if get_option('use_openmp')
  openmpflags = ['-DXSS_USE_OPENMP=true', '-fopenmp']
endif

# Enable AVX and AVX512 support for relevant files
if cpp.has_argument('-mavx')
  libtargets += static_library('libavx',
    files(
      'x86simdsort-avx2.cpp',
      ),
    include_directories : [srcdir, libdir],
    cpp_args : ['-march=haswell', '-mavx', openmpflags],
    gnu_symbol_visibility : 'inlineshidden',
    )
endif

if cpp.has_argument('-mavx512f')
  libtargets += static_library('libskx',
    files(
      'x86simdsort-skx.cpp',
      ),
    include_directories : [srcdir, libdir],
    cpp_args : ['-march=skylake-avx512', '-mavx512f', openmpflags],
    gnu_symbol_visibility : 'inlineshidden',
    )
endif

if cpp.has_argument('-march=icelake-client')
  libtargets += static_library('libicl',
    files(
      'x86simdsort-icl.cpp',
      ),
    include_directories : [srcdir, libdir],
    cpp_args : ['-march=icelake-client', '-mavx', openmpflags],
    gnu_symbol_visibility : 'inlineshidden',
    )
endif

if cpp.has_argument('-march=sapphirerapids')
  libtargets += static_library('libspr',
    files(
      'x86simdsort-spr.cpp',
      ),
    include_directories : [srcdir, libdir],
    cpp_args : ['-march=sapphirerapids', '-mavx512f', openmpflags],
    gnu_symbol_visibility : 'inlineshidden',
    )
endif

install_headers('x86simdsort.h')