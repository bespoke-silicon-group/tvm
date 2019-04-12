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
  void SetDevice(TVMContext ctx) final {
    LOG(INFO) << "Call HBMCDeviceAPI::SetDevice()";

    hb_mc_init_host((uint8_t*)&(ctx.device_id));

    tile_t tiles[1];
    tiles[0].x = 0;
    tiles[0].y = 1;
    tiles[0].origin_x = 0;
    tiles[0].origin_y = 1;
    uint32_t num_tiles = 1;
    eva_id_t eva_id = 0;

    if (hb_mc_init_device(ctx.device_id, eva_id, "/home/centos/cuda_add.riscv", 
                          &tiles[0], num_tiles) != HB_MC_SUCCESS) {
      LOG(FATAL) << "could not initialize device.";
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
    std::cout << "Call HBMC AllocDataSpace()\n";
    HBMCDeviceAPI::SetDevice(ctx);

    eva_id_t eva_id = 0; // set eva_id to zero to make malloc func work

    eva_t ptr = hb_mc_device_malloc(eva_id, nbytes);

    return static_cast<void *>(&ptr);
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    std::cout << "Call HBMC FreeDataSpace()\n";

    eva_id_t eva_id = 0; // set eva_id to zero to make malloc func work
    hb_mc_device_free(eva_id, static_cast<uint32_t>(reinterpret_cast<std::uintptr_t>(ptr)));
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
    std::cout << "Call HBMC CopyDataFromTo()\n";

    std::cout << "ctx_from: " << ctx_from.device_id << std::endl;
    std::cout << "ctx_to: " << ctx_to.device_id << std::endl;

    //if (hb_mc_device_memcpy(hbmc_device_id, eva_id, dst, src, size_buffer * sizeof(uint32_t), hb_mc_memcpy_to_device) != HB_MC_SUCCESS) {
      //printf("Could not copy buffer A to device.\n");
    //}
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

 //private:
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
