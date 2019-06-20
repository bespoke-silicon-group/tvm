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
  std::cout << code;

  std::string file_name_prefix = "cuda_lite_kernel";
  runtime::SaveBinaryToFile(file_name_prefix + ".c", code.c_str());

  std::string compiler = "/home/centos/bsg_manycore/software/riscv-tools/riscv-install/bin/riscv32-unknown-elf-gcc";
  std::string flags = "-march=rv32ima -static -std=gnu99 -ffast-math -fno-common -fno-builtin-printf";
  std::string includes = "-I/home/centos/bsg_manycore/software/spmd/common/";
  includes += " -I/home/centos/bsg_manycore/software/bsg_manycore_lib";
  std::string dynamics = "-Dbsg_tiles_X=2 -Dbsg_tiles_Y=2 -Dbsg_global_X=4 -Dbsg_global_Y=4 -Dbsg_group_size=4 -O2 -DPREALLOCATE=0 -DHOST_DEBUG=0";

  std::string cmd = compiler + " " + flags + " " + includes + " " + dynamics;
  cmd += " -c " + file_name_prefix + ".c -o " + file_name_prefix + ".o";

  LOG(INFO) << cmd;
  // compile the kernel object code
  if (system(cmd.c_str()) != 0)
    LOG(FATAL) << "Error while compiling CUDA-Lite code";

  // linke the objects 
  std::string main_o = "/home/centos/tvm-hb/tutorials/cuda_lite/main.o";
  std::string set_o = "/home/centos/tvm-hb/tutorials/cuda_lite/bsg_set_tile_x_y.o";

  std::string compiler_t = "-t -T /home/centos/bsg_manycore/software/spmd/common/link_dmem.ld";
  std::string compiler_w = "-Wl,--defsym,bsg_group_size=4 -Wl,--no-check-sections";
  std::string compiler_l = "-lc -lgcc -l:crt.o -L /home/centos/bsg_manycore/software/spmd/common";
  std::string compiler_misc = "-march=rv32ima -nostdlib -nostartfiles -ffast-math";
  std::string out_name = file_name_prefix + ".riscv";

  cmd = compiler + " " + compiler_t + " " + compiler_w + " " + main_o + " " + set_o + " " + file_name_prefix + ".o ";
  cmd = cmd + "-o " + out_name + " " + compiler_misc + " " + compiler_l;
  LOG(INFO) << cmd;
  // compile the kernel object code
  if (system(cmd.c_str()) != 0)
    LOG(FATAL) << "Error while linking CUDA-Lite code";

  // TODO Chage to return module with loaded binary
  //return codegen::DeviceSourceModuleCreate(code, "cuda-lite", ExtractFuncInfo(funcs), "cuda-lite");

  std::string data;
  runtime::LoadBinaryFromFile(out_name, &data);

  return HBMCModuleCreate(data, "riscv", ExtractFuncInfo(funcs), code.c_str(), out_name);

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
