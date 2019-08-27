main.riscv: 
	cp cuda_lite_kernel.c $(BSG_MANYCORE_DIR)/software/spmd/bsg_cuda_lite_runtime/tvm/
	$(MAKE) -C $(BSG_MANYCORE_DIR)/software/spmd/bsg_cuda_lite_runtime/tvm/ $@
	cp $(BSG_MANYCORE_DIR)/software/spmd/bsg_cuda_lite_runtime/tvm/$@ .

.Phony: clean

clean:
	rm -f *.riscv *.o *.log
	$(MAKE) -C $(BSG_MANYCORE_DIR)/software/spmd/bsg_cuda_lite_runtime/tvm/ clean

