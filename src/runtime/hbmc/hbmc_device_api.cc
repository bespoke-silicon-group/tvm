/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */
#include <tvm/runtime/device_api.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>

#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_cuda.h>

#include "hbmc_common.h"

namespace tvm {
namespace runtime {

//extern device_t HBMC_DEVICE_;
class HBMCDeviceAPI final : public DeviceAPI {
 public:
  HBMCDeviceAPI() {
    init_flag = false;
  }
  ~HBMCDeviceAPI() {
    hb_mc_device_finish(&HBMC_DEVICE_); /* freeze the tiles and memory manager cleanup */
  }
  void SetDevice(TVMContext ctx) final {
    if (init_flag == false) {
      std::cout << "Call HBMCDeviceAPI::SetDevice(), Initializing HBMC device...\n";

      hb_mc_device_t device;
      char elf_path[] = "cuda_lite_kernel.riscv";
      if (hb_mc_device_init(&HBMC_DEVICE_, "tvm_hb", 0) != HB_MC_SUCCESS)
        LOG(FATAL) << "could not initialize device.";
      
      ctx.device_id = HBMC_DEVICE_.mc->id;

      if (hb_mc_device_program_init(&HBMC_DEVICE_, elf_path, "tvm_hb", 0) != HB_MC_SUCCESS)
        LOG(FATAL) << "could not initialize program.";

      init_flag = true;
    }
  }
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }
  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       TVMType type_hint) final {
    // TODO do this when TVM setup the context
    HBMCDeviceAPI::SetDevice(ctx);

    eva_t ptr;
    if (init_flag == true) {
      hb_mc_device_malloc(&HBMC_DEVICE_, nbytes, &ptr);
      printf("allocate FPGA memory at addr: 0x%x\n", ptr);
    }
    else
      LOG(FATAL) << "You should init hbmc device first";

    return (void*) ptr;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    if (init_flag == true) {
      printf("free FPGA memory at addr: 0x%x\n", reinterpret_cast<uint32_t*>(ptr));
      hb_mc_device_free(&HBMC_DEVICE_, static_cast<uint32_t>(
                                      reinterpret_cast<std::uintptr_t>(ptr)));
    }
    else
      LOG(FATAL) << "You should init hbmc device first";
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMType type_hint,
                      TVMStreamHandle stream) final {
    std::cout << "Call HBMCDeviceAPI::CopyDataFromTo()\n";

    if (init_flag == true) {
      if (ctx_from.device_type == kDLCPU) {
        printf("copy from host mem addr: 0x%x ", reinterpret_cast<uint64_t*>((void*) from));
        printf("to fpga mem addr: 0x%x, ", reinterpret_cast<uint32_t*>(to));
        printf("size %d\n", size);
        if (hb_mc_device_memcpy(&HBMC_DEVICE_, to, 
            from, size, HB_MC_MEMCPY_TO_DEVICE) != HB_MC_SUCCESS)
          LOG(FATAL) << "Unable to memcpy from host to hbmc device.";
      }
      else {
        printf("copy from fpga mem addr: 0x%x ", 
                reinterpret_cast<uint64_t*>((void*) from));
        printf("to host mem addr: 0x%x, ", reinterpret_cast<uint32_t*>(to));
        printf("size %d\n", size);

        if (hb_mc_device_memcpy(&HBMC_DEVICE_, to, 
            from, size, HB_MC_MEMCPY_TO_HOST) != HB_MC_SUCCESS)
          LOG(FATAL) << "Could not do memory copy from host to hbmc device.";
      }
    }
    else
      LOG(FATAL) << "You should init hbmc device first";
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final {
    return HBMCThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    HBMCThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
  }

  static const std::shared_ptr<HBMCDeviceAPI>& Global() {
    static std::shared_ptr<HBMCDeviceAPI> inst =
        std::make_shared<HBMCDeviceAPI>();
    return inst;
  }

  private:
    bool init_flag;
  /*
  static void GPUCopy(const void* from,
                      void* to,
                      size_t size,
                      cudaMemcpyKind kind,
                      cudaStream_t stream) {
    if (stream != 0) {
      CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
    } else {
      CUDA_CALL(cudaMemcpy(to, from, size, kind));
    }
  }
  */
};

typedef dmlc::ThreadLocalStore<HBMCThreadEntry> HBMCThreadStore;

HBMCThreadEntry::HBMCThreadEntry()
    : pool(static_cast<DLDeviceType>(kDLHBMC), HBMCDeviceAPI::Global()) {
}

HBMCThreadEntry* HBMCThreadEntry::ThreadLocal() {
  return HBMCThreadStore::Get();
}

TVM_REGISTER_GLOBAL("device_api.hbmc")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = HBMCDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace runtime
}  // namespace tvm
