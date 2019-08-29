if(NOT USE_CUDA_LITE STREQUAL "OFF")

  if(NOT IS_DIRECTORY ${USE_CUDA_LITE})
    message(FATAL_ERROR "Please specify bsg_bladerunner root path in config.cmake")
  elseif(IS_DIRECTORY ${USE_CUDA_LITE})
    set(BSG_BLADERUNNER_ROOT ${USE_CUDA_LITE})
    configure_file(cmake/cuda_lite.makefile ../hb/cuda_lite_compile/Makefile)
    message(STATUS "Build with CUDA-Lite support...")
    list(APPEND COMPILER_SRCS src/codegen/opt/build_cuda_lite_on.cc)
    file(GLOB RUNTIME_HB_SRCS src/runtime/hbmc/*.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_HB_SRCS})

    find_cuda_lite(${CUDA_LITE_LIB_PATH})

    if(CUDA_LITE_FOUND)
      include_directories(${CUDA_LITE_INCLUDE_DIRS})
    endif(CUDA_LITE_FOUND)

    list(APPEND TVM_RUNTIME_LINKER_LIBS ${BSG_MANYCORE_RUNTIME_LIBRARY})
  endif()
endif()
