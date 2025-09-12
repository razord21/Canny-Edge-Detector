# CUDA + C build
NVCC=nvcc
CC=gcc
CUDA_ARCH?=-arch=sm_60
INCLUDE_DIR=include
SRC_DIR=src
BUILD_DIR=build
BIN_DIR=bin
CFLAGS=-O3 -fPIC
NVCCFLAGS=-O3 -Xcompiler -fPIC $(CUDA_ARCH) -rdc=true -I$(INCLUDE_DIR)
LDFLAGS=-lm
TARGET=$(BIN_DIR)/canny

CSRCS=$(SRC_DIR)/hysteresis.c $(SRC_DIR)/pgm_io.c $(SRC_DIR)/canny_edge.c
CUSRC=$(SRC_DIR)/cuda_functions.cu
COBJS=$(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(CSRCS))
CUOBJS=$(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUSRC))

all: $(TARGET)

$(TARGET): $(COBJS) $(CUOBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(COBJS) $(CUOBJS) $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

run: all
	$(TARGET) canny/pics/pic_large.pgm 2.5 0.25 0.5

clean:
	@rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all run clean
