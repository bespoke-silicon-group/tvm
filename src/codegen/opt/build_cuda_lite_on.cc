/*!
 *  Copyright (c) 2017 by Contributors
 *  Build cuda modules from source.
 *  requires cuda to be available.
 *
 * \file build_cuda.cc
 */
#if defined(__linux__)
#include <sys/stat.h>
#endif
#include <tvm/base.h>
#include <cstdlib>

#include "../codegen_cuda_lite.h"
#include "../build_common.h"

namespace tvm {
namespace codegen {

runtime::Module BuildCUDALite(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenCUDALite cg;
  cg.Init(output_ssa);

  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();
  return codegen::DeviceSourceModuleCreate(code, "cuda-lite", ExtractFuncInfo(funcs), "cuda-lite");

  /*
  if (const auto* f = Registry::Get("tvm_callback_cuda_postproc")) {
    code = (*f)(code).operator std::string();
  }
  std::string fmt = "ptx";
  std::string ptx;
  if (const auto* f = Registry::Get("tvm_callback_cuda_compile")) {
    ptx = (*f)(code).operator std::string();
    // Dirty matching to check PTX vs cubin.
    // TODO(tqchen) more reliable checks
    if (ptx[0] != '/') fmt = "cubin";
  } else {
    ptx = NVRTCCompile(code, cg.need_include_path());
  }
  return CUDAModuleCreate(ptx, fmt, ExtractFuncInfo(funcs), code);
  */
}

TVM_REGISTER_API("codegen.build_cuda_lite")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildCUDALite(args[0]);
  });
}  // namespace codegen
}  // namespace tvm
