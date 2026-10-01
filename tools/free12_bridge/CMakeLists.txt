cmake_minimum_required(VERSION 3.18)
project(free12_bridge LANGUAGES CXX)
find_package(OpenMP REQUIRED)
get_filename_component(NATIVE "${CMAKE_CURRENT_SOURCE_DIR}/../../native_core" ABSOLUTE)
find_path(LIBLZMA_INCLUDE_DIR lzma.h HINTS "C:/Anaconda/Library/include")
if(NOT LIBLZMA_INCLUDE_DIR)
  message(FATAL_ERROR "Set LIBLZMA_INCLUDE_DIR to the directory containing lzma.h")
endif()
add_executable(free12_bridge bridge.cpp
  ${NATIVE}/src/BoardMaskerAD.cpp ${NATIVE}/src/BoardMoverAD.cpp
  ${NATIVE}/src/BookGeneratorUtils.cpp ${NATIVE}/src/CanonicalBatch.cpp
  ${NATIVE}/src/EXADBuilder.cpp ${NATIVE}/src/EXADIO.cpp
  ${NATIVE}/src/EXADLayer.cpp ${NATIVE}/src/EXFrozenLayer.cpp
  ${NATIVE}/src/BookGeneratorEXAD.cpp
  ${NATIVE}/src/FileIOUtils.cpp ${NATIVE}/src/NativeLzma.cpp
  ${NATIVE}/src/CompressionBridge.cpp ${NATIVE}/src/TrieCompression.cpp
  ${NATIVE}/src/HybridSearch.cpp ${NATIVE}/src/AdaptiveIndex.cpp)
target_compile_features(free12_bridge PRIVATE cxx_std_17)
target_include_directories(free12_bridge PRIVATE ${NATIVE}/include ${LIBLZMA_INCLUDE_DIR})
target_link_libraries(free12_bridge PRIVATE OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
if(MSVC)
  target_compile_options(free12_bridge PRIVATE /O2 /Gy /arch:AVX2)
  target_link_options(free12_bridge PRIVATE /OPT:REF)
else()
  target_compile_options(free12_bridge PRIVATE -O3 -march=x86-64-v3 -ffunction-sections -fdata-sections)
  target_link_options(free12_bridge PRIVATE -Wl,--gc-sections -static-libstdc++)
endif()
if(MINGW)
  get_filename_component(COMPILER_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
  foreach(DLL libgcc_s_seh-1.dll libgomp-1.dll libwinpthread-1.dll)
    add_custom_command(TARGET free12_bridge POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${COMPILER_BIN}/${DLL}" "$<TARGET_FILE_DIR:free12_bridge>/${DLL}")
  endforeach()
endif()
# The helper calls only the forward entrypoint from BookGeneratorEXAD.cpp.
# Backward solving runs through the installed formation_core Python binding.
