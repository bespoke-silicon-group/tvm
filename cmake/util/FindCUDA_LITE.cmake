macro(find_cuda_lite)
  find_library(BSG_MANYCORE_RUNTIME_LIBRARY bsg_manycore_runtime)

  if(NOT BSG_MANYCORE_RUNTIME_LIBRARY)
    if(NOT BSG_F1_DIR)
        message(FATAL_ERROR "Can not find bsg_manycore_runtime library, please set(BSG_F1_DIR /path/to/bsg_f1)")
    elseif(IS_DIRECTORY ${BSG_F1_DIR})
      string(CONCAT CUDA_LITE_LIB_PATH ${BSG_F1_DIR} "/cl_manycore/libraries/")
      find_library(BSG_MANYCORE_RUNTIME_LIBRARY bsg_manycore_runtime PATHS ${CUDA_LITE_LIB_PATH})
      if(BSG_MANYCORE_RUNTIME_LIBRARY)
        set(CUDA_LITE_FOUND TRUE)
        set(CUDA_LITE_INCLUDE_DIRS ${CUDA_LITE_LIB_PATH})
        message(STATUS "Found CUDA_LITE_INCLUDE_DIRS=" ${CUDA_LITE_INCLUDE_DIRS})
      else()
        message(FATAL_ERROR "Can not find bsg_manycore_runtime library, please check your bsg_f1 library")
      endif()
    endif()
  endif()

  message(STATUS "Found BSG_MANYCORE_RUNTIME_LIBRARY =" ${BSG_MANYCORE_RUNTIME_LIBRARY})
endmacro(find_cuda_lite)
