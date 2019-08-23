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
#include "../../runtime/file_util.h"
#include "../../runtime/meta_data.h"
#include "../../runtime/hbmc/hbmc_common.h"
#include "../../runtime/hbmc/hbmc_module.h"

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

  if (const auto* f = Registry::Get("tvm_callback_cuda_lite_postproc")) {
    code = (*f)(code).operator std::string();
  }

  const char* tvm_home = std::getenv("TVM_HOME");
  std::string file_name = "cuda_lite_kernel.c";
  std::string make_path = std::string(tvm_home) + "/hb/cuda_lite_compile/";
  
  runtime::SaveBinaryToFile(make_path + file_name, code.c_str());

  std::string make_clean = "make -C " + make_path + " clean";
  if (system(make_clean.c_str()) != 0)
    LOG(FATAL) << "Error while make clean CUDA-Lite code";

  std::string make_comm = "make -C " + make_path + " main.riscv";
  if (system(make_comm.c_str()) != 0)
    LOG(FATAL) << "Error while make CUDA-Lite code";

  // TODO Chage to return module with loaded binary
  //return codegen::DeviceSourceModuleCreate(code, "cuda-lite", ExtractFuncInfo(funcs), "cuda-lite");

  std::string data;
  //runtime::LoadBinaryFromFile(out_name, &data);
  runtime::LoadBinaryFromFile(make_path + "main.riscv", &data);

  return HBMCModuleCreate(data, "hbmc", ExtractFuncInfo(funcs), code.c_str());

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
