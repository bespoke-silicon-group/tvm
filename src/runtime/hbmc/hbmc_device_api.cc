/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */
#include <tvm/runtime/device_api.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>

#include <bsg_manycore_driver.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_mem.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_cuda.h>

#include "hbmc_common.h"

namespace tvm {
namespace runtime {

class HBMCDeviceAPI final : public DeviceAPI {
 public:
  HBMCDeviceAPI() {
    init_flag = false;
  }
  void SetDevice(TVMContext ctx) final {
    if (init_flag == false) {
      /*
      std::cout << "Call HBMCDeviceAPI::SetDevice(), Initializing HBMC host...\n";
      hb_mc_init_host((uint8_t*)&(ctx.device_id));
      //LOG(INFO) << "ctx.device_id: " << ctx.device_id;

      tile_t tiles[4];
      uint32_t num_tiles = 4, num_tiles_x = 2, num_tiles_y = 2, origin_x = 0, origin_y = 1;
      // 2 x 2 tile group at (0, 1) 
      create_tile_group(tiles, num_tiles_x, num_tiles_y, origin_x, origin_y); 

      eva_id_t eva_id = 0;

      std::cout << "Initializing HBMC device...\n";
      //char elf_path[] = "/home/centos/tvm-hb/tutorials/cuda_lite/cuda_lite_kernel.riscv";
      char elf_path[] = "cuda_lite_kernel.riscv";
      // TODO hb_mc_init_device(should not take binary as input)
      if (hb_mc_init_device(ctx.device_id, eva_id, elf_path, &tiles[0], num_tiles) != HB_MC_SUCCESS)
        LOG(FATAL) << "could not initialize device.";
      */

      uint8_t mesh_dim_x = 4;
      uint8_t mesh_dim_y = 4;
      uint8_t mesh_origin_x = 0;
      uint8_t mesh_origin_y = 1;
      eva_id_t eva_id = 0;
      char elf_path[] = "cuda_lite_kernel.riscv";

      std::cout << "Initializing HBMC device...\n";
      // TODO the device info should be passed to ctx
      if (hb_mc_device_init(&HBMC_DEVICE_, eva_id, elf_path, mesh_dim_x, mesh_dim_y, 
          mesh_origin_x, mesh_origin_y) != HB_MC_SUCCESS)
        LOG(FATAL) << "could not initialize device.";
      ctx.device_id = HBMC_DEVICE_.fd;
      LOG(INFO) << "ctx.device_id: " << ctx.device_id;

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
    // TODO do this when TVM setup the context
    HBMCDeviceAPI::SetDevice(ctx);

    /*
    std::cout << "HBMCDeviceAPI::AllocDataSpace(): ";
    std::cout << "ctx.(device_type, device_id) = (" << ctx.device_type;
    std::cout << ", " << ctx.device_id << ")" << std::endl;
    */

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
    //std::cout << "Call HBMCDeviceAPI::FreeDataSpace()\n";
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
    //std::cout << "Call HBMCDeviceAPI::CopyDataFromTo()\n";

    /*
    std::cout << "size: " << size << std::endl;
    std::cout << "HBMCDeviceAPI::CopyFromBytes(): \n";
    std::cout << "ctx_from.(device_type, device_id) = (" << ctx_from.device_type;
    std::cout << ", " << ctx_from.device_id << ")" << std::endl;
    std::cout << "ctx_to.(device_type, device_id) = (" << ctx_to.device_type;
    std::cout << ", " << ctx_to.device_id << ")" << std::endl;
    */

    if (init_flag == true) {
      if (ctx_from.device_type == kDLCPU) {
        printf("copy from host mem addr: 0x%x ", reinterpret_cast<uint64_t*>((void*) from));
        printf("to fpga mem addr: 0x%x, ", reinterpret_cast<uint32_t*>(to));
        printf("size %d\n", size);
        if (hb_mc_device_memcpy(&HBMC_DEVICE_, to, 
            from, size, hb_mc_memcpy_to_device) != HB_MC_SUCCESS)
          LOG(FATAL) << "Unable to memcpy from host to hbmc device.";

        /*
        int32_t *arr = static_cast<int32_t*>((void*)from);
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

        if (hb_mc_device_memcpy(&HBMC_DEVICE_, to, 
            from, size, hb_mc_memcpy_to_host) != HB_MC_SUCCESS)
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
    hb_mc_device_finish(&HBMC_DEVICE_); /* freeze the tiles and memory manager cleanup */
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
