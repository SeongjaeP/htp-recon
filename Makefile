# htp-recon — build all reconstruction tools into ./build
#
# Everything here is self-contained C99 with no dependencies beyond libm.
# src/qnn_min_runtime.c is excluded from the default build: it needs the vendor SDK headers.
# See the `runtime` target below.

CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra
LDLIBS  := -lm
BUILD   := build

# every self-contained tool in src/, i.e. everything except the SDK-dependent runtime
TOOLS := $(filter-out qnn_min_runtime,$(notdir $(basename $(wildcard src/*.c))))

BINS := $(addprefix $(BUILD)/,$(TOOLS))

.PHONY: all clean demo runtime

all: $(BINS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# A quick tour of every result in the repository.
demo: all
	@echo "=== tiling rule: tiles = ceil(H/8) ==="
	@for h in 16 64 128 256; do ./$(BUILD)/lowering $$h 64 32; done
	@echo
	@echo "=== tile shape derived from an on-chip memory budget ==="
	./$(BUILD)/vtcm_tiling
	@echo
	@echo "=== the integrated compiler, small enough to read ==="
	./$(BUILD)/minicc 2097152 16 64 32
	@echo
	@echo "=== scheduling: the same graph under two tie-break policies ==="
	./$(BUILD)/scheduler graphs/branch_merge.graph 0 | tail -12
	./$(BUILD)/scheduler graphs/branch_merge.graph 1 | tail -12
	@echo
	@echo "=== cost-based vs depth-first, on deliberately asymmetric tensor sizes ==="
	./$(BUILD)/scheduler_cbs graphs/asym.graph graphs/asym.sizes | tail -12
	@echo
	@echo "=== resource-aware parallel scheduling ==="
	./$(BUILD)/scheduler_par graphs/branch_merge.graph
	@echo
	@echo "=== VTCM allocation: lifetimes, in-place reuse, offsets ==="
	./$(BUILD)/allocator graphs/branch_merge.graph graphs/branch_merge.sizes 2097152
	@echo
	@echo "=== dequantisation formula, and its check against a shipping model ==="
	./$(BUILD)/dequant_repro
	./$(BUILD)/verify_dequant_llama

# The minimal inference runtime needs the vendor SDK's headers (QnnInterface.h and the C++
# wrapper headers), so it is built separately and only when an SDK is present:
#
#   make runtime QNN_SDK=/path/to/sdk
#
# It is C++ because the SDK's wrapper headers are.
runtime: | $(BUILD)
ifndef QNN_SDK
	@echo "set QNN_SDK=/path/to/vendor/sdk to build the runtime" && false
endif
	$(CXX) -std=c++17 -O2 -I$(QNN_SDK)/include/QNN \
	    -o $(BUILD)/qnn_min_runtime src/qnn_min_runtime.c -ldl

clean:
	rm -rf $(BUILD)
