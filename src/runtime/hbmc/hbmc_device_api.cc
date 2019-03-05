/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */
#include <tvm/runtime/device_api.h>

#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include "hbmc_common.h"

namespace tvm {
namespace runtime {

class HBMCDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final {
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
      return nullptr;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
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
