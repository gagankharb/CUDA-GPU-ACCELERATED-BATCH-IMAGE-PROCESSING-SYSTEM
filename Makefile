# Makefile for CUDA GPU-Accelerated Batch Image Processing System
# Coursera: CUDA at Scale for the Enterprise - Independent Project

CXX := g++
NVCC := nvcc

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Iinclude
NVCCFLAGS := -O3 -std=c++17 -Iinclude \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_80,code=sm_80 \
    -gencode=arch=compute_86,code=sm_86 \
    -gencode=arch=compute_89,code=sm_89 \
    -gencode=arch=compute_90,code=sm_90 \
    -Xcompiler "-Wall -Wextra"

BIN_DIR := bin
OBJ_DIR := build
TARGET := $(BIN_DIR)/image_processor

CPP_SRCS := src/main.cpp src/image_processor.cpp src/image_io.cpp src/cpu_reference.cpp src/benchmark.cpp
CU_SRCS := src/cuda_kernels.cu

CPP_OBJS := $(patsubst src/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
CU_OBJS := $(patsubst src/%.cu, $(OBJ_DIR)/%.cu.o, $(CU_SRCS))

# Check if NVCC is available on system PATH
HAVE_NVCC := $(shell which $(NVCC) 2>/dev/null)

.PHONY: all clean run test help check-env

all: check-env $(TARGET)

check-env:
	@echo "Checking build environment..."
ifdef HAVE_NVCC
	@echo "Found NVCC: $(shell $(NVCC) --version | grep release)"
else
	@echo "Warning: NVCC not detected on PATH. Compiling in host CPU reference verification mode."
endif

$(TARGET): $(CPP_OBJS) $(CU_OBJS) | $(BIN_DIR)
ifdef HAVE_NVCC
	$(NVCC) $(NVCCFLAGS) -o $@ $^
else
	$(CXX) $(CXXFLAGS) -o $@ $^
endif
	@echo "Build successful: $(TARGET)"

$(OBJ_DIR)/%.o: src/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.cu.o: src/%.cu | $(OBJ_DIR)
ifdef HAVE_NVCC
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
else
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@
endif

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) results/grayscale results/blur results/sobel results/sharpen results/resized results/all

run: $(TARGET)
	./$(TARGET) --input data/input --output results --operation all --compare-cpu

test:
	python tests/test_image_processing.py

help:
	@echo "Targets:"
	@echo "  make           - Build $(TARGET)"
	@echo "  make clean     - Remove binaries and build directories"
	@echo "  make run       - Execute batch processing on data/input"
	@echo "  make test      - Run automated verification test suite"
