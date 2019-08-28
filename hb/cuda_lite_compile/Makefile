bsg_global_X ?= $(bsg_tiles_X)
bsg_global_Y ?= $(bsg_tiles_Y)+1

bsg_tiles_org_X ?= 0
bsg_tiles_org_Y ?= 1

bsg_tiles_X ?= 2
bsg_tiles_Y ?= 2

# Chech if it's on AWS instance
ifneq (,$(findstring us-west-2.compute.internal, $(HOSTNAME)))
	IGNORE_CADENV=1
endif

all: main.run

OBJECT_FILES=main.o bsg_set_tile_x_y.o bsg_printf.o cuda_lite_kernel.o
CRT=$(BSG_MANYCORE_DIR)/software/spmd/common/crt.o

include $(BSG_MANYCORE_DIR)/software/spmd/Makefile.include

# remove -nostdlib flag
RISCV_LINK_OPTS:=$(filter-out -nostdlib,$(RISCV_LINK_OPTS))

main.riscv: $(OBJECT_FILES) $(SPMD_COMMON_OBJECTS) $(CRT)
	$(RISCV_LINK) $(OBJECT_FILES) $(SPMD_COMMON_OBJECTS) -o $@ $(RISCV_LINK_OPTS)

main.o: Makefile

include $(BSG_MANYCORE_DIR)/software/mk/Makefile.tail_rules

.Phony: clean

clean:
	rm -f *.riscv *.o *.log
