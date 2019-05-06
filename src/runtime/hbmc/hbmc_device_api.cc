/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */
#include <tvm/runtime/device_api.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <bsg_manycore_driver.h>

#include "hbmc_common.h"

namespace tvm {
namespace runtime {

class HBMCDeviceAPI final : public DeviceAPI {
 public:
  HBMCDeviceAPI() {
    init_flag = false;
  }
  void SetDevice(TVMContext ctx) final {
    //std::cout << "Call HBMCDeviceAPI::SetDevice()";

    if (init_flag == false) {
      //std::cout << "Initializing HBMC host...\n";
      hb_mc_init_host((uint8_t*)&(ctx.device_id));
      //LOG(INFO) << "ctx.device_id: " << ctx.device_id;

      tile_t tiles[1];
      tiles[0].x = 0;
      tiles[0].y = 1;
      tiles[0].origin_x = 0;
      tiles[0].origin_y = 1;
      uint32_t num_tiles = 1;
      eva_id_t eva_id = 0;

      std::cout << "Initializing HBMC device...\n";
      if (hb_mc_init_device(ctx.device_id, eva_id, "/home/centos/cuda_add2.riscv", 
                            &tiles[0], num_tiles) != HB_MC_SUCCESS)
        LOG(FATAL) << "could not initialize device.";

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
    //std::cout << "Call HBMCDeviceAPI::AllocDataSpace()\n";
    HBMCDeviceAPI::SetDevice(ctx);

    /*
    std::cout << "HBMCDeviceAPI::AllocDataSpace(): ";
    std::cout << "ctx.(device_type, device_id) = (" << ctx.device_type;
    std::cout << ", " << ctx.device_id << ")" << std::endl;
    */

    eva_id_t eva_id = 0; // set eva_id to zero to make malloc func work

    eva_t ptr;
    hb_mc_device_malloc(eva_id, nbytes, &ptr);
    printf("allocate FPGA memory at addr: 0x%x\n", ptr);

    return (void*) ptr;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    //std::cout << "Call HBMCDeviceAPI::FreeDataSpace()\n";
    printf("free FPGA memory at addr: 0x%x\n", reinterpret_cast<uint32_t*>(ptr));

    eva_id_t eva_id = 0; // set eva_id to zero to make malloc func work
    hb_mc_device_free(eva_id, static_cast<uint32_t>(
                              reinterpret_cast<std::uintptr_t>(ptr)));
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
    //std::cout << "Call HBMCDeviceAPI::CopyDataFromTo()\n";

    /*
    std::cout << "size: " << size << std::endl;
    std::cout << "HBMCDeviceAPI::CopyFromBytes(): \n";
    std::cout << "ctx_from.(device_type, device_id) = (" << ctx_from.device_type;
    std::cout << ", " << ctx_from.device_id << ")" << std::endl;
    std::cout << "ctx_to.(device_type, device_id) = (" << ctx_to.device_type;
    std::cout << ", " << ctx_to.device_id << ")" << std::endl;
    */

    if (ctx_from.device_type == kDLCPU) {
      printf("copy from host mem addr: 0x%x ", 
              reinterpret_cast<uint64_t*>((void*) from));
      printf("to fpga mem addr: 0x%x, ", reinterpret_cast<uint32_t*>(to));
      printf("size %d\n", size);
      if (hb_mc_device_memcpy(ctx_to.device_id, (eva_id_t) 0, to, 
          from, size, hb_mc_memcpy_to_device) != HB_MC_SUCCESS)
        LOG(FATAL) << "Unable to memcpy from host to hbmc device.";

      int32_t *arr = static_cast<int32_t*>((void*)from);
      /*
      printf("from[0:8]: ");
      for (int i = 0; i < 8; i++)
        printf("%d ", arr[i]); 
      printf("\n");
      */
    }
    else {
      printf("copy from fpga mem addr: 0x%x ", 
              reinterpret_cast<uint64_t*>((void*) from));
      printf("to host mem addr: 0x%x, ", reinterpret_cast<uint32_t*>(to));
      printf("size %d\n", size);

      if (hb_mc_device_memcpy(ctx_to.device_id, (eva_id_t) 0, to, 
          from, size, hb_mc_memcpy_to_host) != HB_MC_SUCCESS)
        LOG(FATAL) << "Could not do memory copy from host to hbmc device.";
    }
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
