cmake_minimum_required(VERSION 3.27)

project(_ext LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----------------------------- Dependencies -----------------------------
find_package(
  Python 3.8
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# The MLX wheel bakes the framework paths of the machine that built it into its
# CMake package (MLXTargets.cmake), e.g.
#   /Applications/Xcode_26.5.app/.../MacOSX.sdk/System/Library/Frameworks/Metal.framework
# That prefix does not exist here, so clang hands the missing directory to the
# linker and every link reports:
#   ld: warning: search path '...' not found
# Re-point stale framework paths at the SDK this machine actually uses.
get_target_property(_mlx_link_libraries mlx INTERFACE_LINK_LIBRARIES)
set(_mlx_link_libraries_fixed "")
foreach(_mlx_library IN LISTS _mlx_link_libraries)
  if("${_mlx_library}" MATCHES "\\.framework$" AND NOT EXISTS "${_mlx_library}")
    if(NOT DEFINED _mlx_sdkroot)
      execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE _mlx_sdkroot
        ERROR_QUIET)
    endif()
    get_filename_component(_mlx_framework "${_mlx_library}" NAME_WE)
    set(_mlx_library "${_mlx_sdkroot}/System/Library/Frameworks/${_mlx_framework}.framework")
    if(NOT EXISTS "${_mlx_library}")
      message(STATUS "MLX: skipping framework missing from '${_mlx_sdkroot}': ${_mlx_framework}")
      continue()
    endif()
  endif()
  list(APPEND _mlx_link_libraries_fixed "${_mlx_library}")
endforeach()
set_target_properties(mlx PROPERTIES INTERFACE_LINK_LIBRARIES "${_mlx_link_libraries_fixed}")
unset(_mlx_link_libraries)
unset(_mlx_link_libraries_fixed)
unset(_mlx_sdkroot)

# ----------------------------- Extensions -----------------------------

# Add library
add_library(tiny_llm_ext_ref)

# Add sources
target_sources(
  tiny_llm_ext_ref
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/src/quantized_matmul.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/week2_kernels.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/paged_attention.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/utils.cpp
)

# Add include headers
target_include_directories(tiny_llm_ext_ref PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/src)

# Link to mlx
target_link_libraries(tiny_llm_ext_ref PUBLIC mlx)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # so that clangd can pick this up

# ----------------------------- Metal -----------------------------

# Build metallib
if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET
    tiny_llm_ext_ref_metallib
    TITLE
    tiny_llm_ext_ref
    SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/quantized_matmul.metal
    ${CMAKE_CURRENT_LIST_DIR}/src/week2_kernels.metal
    ${CMAKE_CURRENT_LIST_DIR}/src/paged_attention.metal
    INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(tiny_llm_ext_ref tiny_llm_ext_ref_metallib)
endif()

# ----------------------------- Python Bindings -----------------------------
nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE tiny_llm_ext_ref)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
