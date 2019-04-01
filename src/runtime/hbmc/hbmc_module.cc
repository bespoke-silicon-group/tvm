/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_module.cc
 */
#include "hbmc_module.h"

#include <tvm/runtime/registry.h>
//#include <cuda.h>
//#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include <bsg_manycore_driver.h>
#include <bsg_manycore_mem.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_print.h>
#include "hbmc_common.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "../meta_data.h"
#include "../file_util.h"

namespace tvm {
namespace runtime {

// Module to support thread-safe multi-GPU execution.
// cuModule is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class HBMCModuleNode : public runtime::ModuleNode {
 public:
  explicit HBMCModuleNode(std::string data,
                          std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string hbmc_source)
      : data_(data), fmt_(fmt), fmap_(fmap), hbmc_source_(hbmc_source) {
    //std::fill(module_.begin(), module_.end(), nullptr);
  }
  // destructor
  ~HBMCModuleNode() {
    /*
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        CUDA_CALL(cudaSetDevice(static_cast<int>(i)));
        CUDA_DRIVER_CALL(cuModuleUnload(module_[i]));
      }
    }
    */
  }

  const char* type_key() const final {
    return "hbmc";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    /*
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "cu") {
      CHECK_NE(cuda_source_.length(), 0);
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, cuda_source_);
    } else {
      CHECK_EQ(fmt, fmt_)
          << "Can only save to format=" << fmt_;
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, data_);
    }
    */
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    /*
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
    */
  }

  std::string GetSource(const std::string& format) final {
    /*
    if (format == fmt_) return data_;
    if (cuda_source_.length() != 0) {
      return cuda_source_;
    } else {
      if (fmt_ == "ptx") return data_;
      return "";
    }
    */
    return hbmc_source_;
  }

  /*
  // get a CUfunction from primary context in device_id
  CUfunction GetFunc(int device_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), data_.c_str()));
    }
    CUfunction func;
    CUresult result = cuModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != CUDA_SUCCESS) {
      const char *msg;
      cuGetErrorName(result, &msg);
      LOG(FATAL)
          << "CUDAError: cuModuleGetFunction " << func_name
          << " failed with error: " << msg;
    }
    return func;
  }
  */
  /*
  // get a global var from primary context in device_id
  CUdeviceptr GetGlobal(int device_id,
                        const std::string& global_name,
                        size_t expect_nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), data_.c_str()));
    }
    CUdeviceptr global;
    size_t nbytes;

    CUresult result = cuModuleGetGlobal(&global, &nbytes,
                                        module_[device_id], global_name.c_str());
    CHECK_EQ(nbytes, expect_nbytes);
    if (result != CUDA_SUCCESS) {
      const char *msg;
      cuGetErrorName(result, &msg);
      LOG(FATAL)
          << "CUDAError: cuModuleGetGlobal " << global_name
          << " failed with error: " << msg;
    }
    return global;
  }
  */

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The cuda source.
  std::string hbmc_source_;
  // the internal modules per GPU, to be lazily initialized.
  // std::array<CUmodule, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  //std::mutex mutex_;
};

// a wrapped function class to get packed func.
class HBMCWrappedFunc {
 public:
  // initialize the CUDA function.
  void Init(HBMCModuleNode* m,
            std::shared_ptr<ModuleNode> sptr,
            const std::string& func_name,
            size_t num_void_args,
            const std::vector<std::string>& thread_axis_tags) {
    std::cout << "Call HBMCWrappedFunc Init()\n";
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    //std::fill(fcache_.begin(), fcache_.end(), nullptr);
    thread_axis_cfg_.Init(num_void_args, thread_axis_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  void** void_args) const {
    std::cout << "Call HBMCWrappedFunc operator()\n";

    uint8_t hbmc_device_id;
    if(!hb_mc_init_host("/dev/bsg_manycore_kernel_driver", &hbmc_device_id)) {
      printf("failed to initialize host\n");
    }
    else {
      printf("success to initialize host\n");
    }

    uint32_t DMEM_BASE = 0x1000;
    uint32_t DATA = 1234;

    bool write = hb_mc_copy_to_epa(hbmc_device_id, 0, 0, DMEM_BASE>>2, &DATA, 1);
    if(!write) {
      printf("writing data to tile (0, 0)'s DMEM failed\n");
    }
    else {
      printf("writing data to tile (0, 0)'s DMEM successed\n");
    }

    uint8_t x = 0, y = 0;

    hb_mc_freeze(hbmc_device_id, x, y);
    hb_mc_load_binary(hbmc_device_id, getenv("MAIN_LOOPBACK"), &x, &y, 1);
    hb_mc_unfreeze(hbmc_device_id, x, y);

    usleep(100);
    uint32_t *received_packet = hb_mc_read_fifo(hbmc_device_id, 1, NULL);
    hb_mc_print_hex((uint8_t *) received_packet);

    /*
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    }
    CUstream strm = static_cast<CUstream>(CUDAThreadEntry::ThreadLocal()->stream);
    ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
    CUresult result = cuLaunchKernel(
        fcache_[device_id],
        wl.grid_dim(0),
        wl.grid_dim(1),
        wl.grid_dim(2),
        wl.block_dim(0),
        wl.block_dim(1),
        wl.block_dim(2),
        0, strm, void_args, 0);
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
      const char *msg;
      cuGetErrorName(result, &msg);
      std::ostringstream os;
      os << "CUDALaunch Error: " << msg << "\n"
         << " grid=(" << wl.grid_dim(0) << ","
         << wl.grid_dim(1) << "," << wl.grid_dim(2) << "), "
         << " block=(" << wl.block_dim(0) << ","
         << wl.block_dim(1) << "," << wl.block_dim(2) << ")\n";
      std::string cuda = m_->GetSource("");
      if (cuda.length() != 0) {
        os << "// func_name=" << func_name_ << "\n"
           << "// CUDA Source\n"
           << "// -----------\n"
           << cuda;
      }
      LOG(FATAL) << os.str();
    }
    */
  }

 private:
  // internal module
  HBMCModuleNode* m_;
  // the resource holder
  std::shared_ptr<ModuleNode> sptr_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  //mutable std::array<CUfunction, kMaxNumGPUs> fcache_;
  // thread axis configuration
  ThreadAxisConfig thread_axis_cfg_;
};

/*
class CUDAPrepGlobalBarrier {
 public:
  CUDAPrepGlobalBarrier(CUDAModuleNode* m,
                        std::shared_ptr<ModuleNode> sptr)
      : m_(m), sptr_(sptr) {
    std::fill(pcache_.begin(), pcache_.end(), 0);
  }

  void operator()(const TVMArgs& args, TVMRetValue* rv) const {
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    if (pcache_[device_id] == 0) {
      pcache_[device_id] = m_->GetGlobal(
          device_id, runtime::symbol::tvm_global_barrier_state, sizeof(unsigned));
    }
    CUDA_DRIVER_CALL(cuMemsetD32(pcache_[device_id], 0, 1));
  }

 private:
  // internal module
  CUDAModuleNode* m_;
  // the resource holder
  std::shared_ptr<ModuleNode> sptr_;
  // mark as mutable, to enable lazy initialization
  mutable std::array<CUdeviceptr, kMaxNumGPUs> pcache_;
};
*/

PackedFunc HBMCModuleNode::GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) {
  std::cout << "Call HBMCModuleNode GetFunction():  " << name << std::endl;
  CHECK_EQ(sptr_to_self.get(), this);
  CHECK_NE(name, symbol::tvm_module_main)
      << "Device function do not have main";
  //if (name == symbol::tvm_prepare_global_barrier) {
    //return PackedFunc(HBPrepGlobalBarrier(this, sptr_to_self));
  //}
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  HBMCWrappedFunc f;
  f.Init(this, sptr_to_self, name, info.arg_types.size(), info.thread_axis_tags);
  return PackFuncVoidAddr(f, info.arg_types);
}

Module HBMCModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string hb_source) {
  std::shared_ptr<HBMCModuleNode> n =
      std::make_shared<HBMCModuleNode>(data, fmt, fmap, hb_source);
  std::cout << "Call HBMCModuleCreate()" << std::endl;
  return Module(n);
}

// Load module from module.
Module HBMCModuleLoadFile(const std::string& file_name,
                          const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  std::cout << "Call HBMCModuleLoadFile()" << std::endl;

  /*
  uint32_t hb_mc_data = 0xABCD;
  uint32_t DMEM_BASE = 0x1000;
  bool write = hb_mc_copy_to_epa(fd, 0, 0, DMEM_BASE >> 2, &hb_mc_data, 1);

  if (!write) {
    printf("writing data to tile (0, 0)'s DMEM failed.\n");
  }
  else {
    // read back data
    uint32_t **buf = (uint32_t **) calloc(1, sizeof(uint32_t *));
    bool read = hb_mc_copy_from_epa(fd, buf, 0, 0, DMEM_BASE >> 2, 1); 
    printf("completed read.\n");
    if (read == 1) {
      printf("read packet: ");
      hb_mc_print_hex((uint8_t *) buf[0]);
    }
    else {
      printf("read from tile failed.\n");
    }
  }
  */

  return HBMCModuleCreate(data, fmt, fmap, std::string());
}

/*
Module CUDAModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return CUDAModuleCreate(data, fmt, fmap, std::string());
}
*/

TVM_REGISTER_GLOBAL("module.loadfile_hbmc")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = HBMCModuleLoadFile(args[0], args[1]);
  });

/*
TVM_REGISTER_GLOBAL("module.loadfile_cubin")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = CUDAModuleLoadFile(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("module.loadfile_ptx")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = CUDAModuleLoadFile(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("module.loadbinary_cuda")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = CUDAModuleLoadBinary(args[0]);
  });
*/
}  // namespace runtime
}  // namespace tvm
