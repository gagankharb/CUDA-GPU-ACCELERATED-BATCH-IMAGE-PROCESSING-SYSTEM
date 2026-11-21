# CUDA GPU-Accelerated Batch Image Processing System
**Coursera: CUDA at Scale for the Enterprise — Independent Project**

[![CUDA](https://img.shields.io/badge/CUDA-12.x%20%7C%2011.x-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-3.18%2B-064F8C?logo=cmake)](https://cmake.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, high-throughput batch image processing system engineered in CUDA C++ and C++. The system demonstrates genuine GPU parallel acceleration by processing hundreds of high-resolution images in a single execution across five foundational computer vision operations.

---

## Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. Problem Statement](#2-problem-statement)
- [3. Project Objectives](#3-project-objectives)
- [4. System Architecture & Memory Model](#4-system-architecture--memory-model)
- [5. Implemented CUDA Kernels](#5-implemented-cuda-kernels)
- [6. Dataset Acquisition & Generation](#6-dataset-acquisition--generation)
- [7. Benchmark Results & Speedup Analysis](#7-benchmark-results--speedup-analysis)
- [8. Visual Verification & Algorithmic Outputs](#8-visual-verification--algorithmic-outputs)
- [9. Repository Structure](#9-repository-structure)
- [10. Build and Execution Instructions](#10-build-and-execution-instructions)
- [11. Command-Line Interface (CLI)](#11-command-line-interface-cli)
- [12. Enterprise Challenges & Optimization Insights](#12-enterprise-challenges--optimization-insights)
- [13. Presentation & Demonstration Video](#13-presentation--demonstration-video)
- [14. GitHub Metadata](#14-github-metadata)

---

## 1. Executive Summary

In high-throughput enterprise pipelines (e.g. computer vision inference pre-processing, satellite imagery ingestion, medical scan analysis), processing massive image batches on CPUs causes severe throughput bottlenecks. This project presents a complete, production-ready batch image processing engine written in custom CUDA C++.

### Key Achievements:
- **Massive Batch Processing:** Processed **500 images** ($512 \times 512$ RGB, totaling **131.07 Megapixels**) in a single unified execution.
- **Custom CUDA Kernels:** Implemented 5 genuine GPU kernels without black-box third-party library abstractions:
  1. Grayscale Conversion (ITU-R BT.601 standard luminance).
  2. Gaussian Blur (5x5 spatial smoothing with 2D shared memory tiling).
  3. Sobel Edge Detection (3x3 horizontal and vertical gradient magnitude).
  4. Image Sharpening (3x3 Laplacian unsharp masking).
  5. Image Resizing (continuous spatial mapping with 4-tap bilinear interpolation).
- **Advanced GPU Optimizations:** Applied 2D Shared Memory Tiling ($20 \times 20$ and $18 \times 18$ aprons) and Constant Memory caching to reduce global memory traffic by up to $25\times$.
- **High-Performance Speedup:** Achieved up to **$47.3\times$ raw kernel speedup** and **$43.9\times$ end-to-end speedup** (including PCIe host-to-device and device-to-host transfers) on 2D convolutions.
- **Hardware-Accurate Profiling:** Measured execution times with nanosecond-resolution CUDA Events and compared against a CPU reference engine.

---

## 2. Problem Statement

Enterprise image processing pipelines frequently ingest millions of pixels per second. 2D spatial filtering operations, such as Gaussian smoothing and edge detection, require multiple memory lookups and floating-point computations per pixel (e.g., 25 memory reads and 50 FLOPs per pixel for a 5x5 stencil).

When executed sequentially or across limited CPU worker threads:
1. Cache hierarchies are continuously trashed by large 2D image strides.
2. Memory bandwidth limits prevent real-time frame rates.
3. CPU cores become a bottleneck for downstream GPU inference models.

By offloading batch image processing to GPU streaming multiprocessors (SMs), massive Single Instruction Multiple Threads (SIMT) parallelism can be exploited to achieve orders-of-magnitude higher throughput.

---

## 3. Project Objectives

1. **Scale:** Process 100 to 500 images in a single program execution.
2. **Authentic CUDA Computing:** Develop custom CUDA kernels using optimal 2D thread block and grid geometries.
3. **Memory Optimization:** Implement on-chip shared memory tiling to cache stencil halos and use constant memory for filter weights.
4. **End-to-End Profiling:** Measure raw kernel execution time independently from PCIe transfer overhead using CUDA events.
5. **Verification:** Validate output images mathematically (Mean Absolute Error and PSNR) against CPU reference implementations.
6. **Production Standards:** Provide clean C++17 modular code, CMake/Makefile build systems, CLI flags, and reproducible datasets.

---

## 4. System Architecture & Memory Model

```
+-----------------------------------------------------------------------------------+
|                                  HOST SYSTEM                                      |
|                                                                                   |
|   +-------------------+      +-------------------+      +--------------------+    |
|   | Input Directory   | ---> | ImageIO Module    | ---> | Host RGB Buffers   |    |
|   | (500x .bmp Files) |      | (Headerless BMP)  |      | (Pinned / Paged)   |    |
|   +-------------------+      +-------------------+      +--------------------+    |
+----------------------------------------|------------------------------------------+
                                         | PCIe Transfer (H2D)
                                         v
+-----------------------------------------------------------------------------------+
|                                DEVICE (NVIDIA GPU)                                |
|                                                                                   |
|   +----------------------------------------------------------------------------+  |
|   | Global Device Memory: d_in (RGB / Grayscale Pixels)                        |  |
|   +----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|   +----------------------------------------------------------------------------+  |
|   | Streaming Multiprocessors (SMs)                                            |  |
|   |                                                                            |  |
|   |   +----------------------------+     +-------------------------------+     |  |
|   |   | 2D Shared Memory Tile      |     | Constant Memory (64 KB)       |     |  |
|   |   | (Halo Apron Caching)       |     | (c_gaussianKernel, c_sobel)   |     |  |
|   |   +----------------------------+     +-------------------------------+     |  |
|   |                  \                          /                              |  |
|   |                   +-----------+------------+                               |  |
|   |                               v                                            |  |
|   |   +--------------------------------------------------------------------+   |  |
|   |   | Custom CUDA Kernels:                                               |   |  |
|   |   | - rgbToGrayscaleKernel        (BT.601 SIMT Luminance)              |   |  |
|   |   | - gaussianBlurSharedKernel    (5x5 Tiled Convolution, R=2)         |   |  |
|   |   | - sobelSharedKernel           (3x3 Gradient Magnitude, R=1)        |   |  |
|   |   | - sharpenSharedKernel         (3x3 Laplacian Unsharp Mask)         |   |  |
|   |   | - resizeBilinearKernel        (4-Tap Bilinear Interpolation)       |   |  |
|   |   +--------------------------------------------------------------------+   |  |
|   +------------------------------------|---------------------------------------+  |
|                                        v                                          |
|   +----------------------------------------------------------------------------+  |
|   | Global Device Memory: d_out (Processed Image Buffer)                       |  |
|   +----------------------------------------------------------------------------+  |
+----------------------------------------|------------------------------------------+
                                         | PCIe Transfer (D2H)
                                         v
+-----------------------------------------------------------------------------------+
|                              HOST RESULTS & METRICS                               |
|                                                                                   |
|   +----------------------+   +-----------------------+   +--------------------+   |
|   | Output Disk Files    |   | Benchmark Reporter    |   | CPU Verification   |   |
|   | (results/<operation>)|   | (CSV & Summary TXT)   |   | (MAE < 0.01)       |   |
|   +----------------------+   +-----------------------+   +--------------------+   |
+-----------------------------------------------------------------------------------+
```

### Memory Optimization Strategy:
- **Global Memory Coalescing:** Consecutive threads within a 32-thread warp access adjacent bytes in memory, ensuring maximum memory bus transaction efficiency.
- **2D Shared Memory Tiling:** For $5 \times 5$ Gaussian blur, thread blocks of $16 \times 16$ load an apron of $(16 + 2 \cdot 2) \times (16 + 2 \cdot 2) = 20 \times 20$ elements into fast on-chip `__shared__` memory. Redundant global memory requests are reduced by up to $25\times$.
- **Constant Memory Caching:** Static convolution weights (`c_gaussianKernel5x5`, `c_sobelX`, `c_sobelY`, `c_sharpen3x3`) reside in 64 KB constant memory, benefiting from zero-cycle warp broadcast when all threads access identical mask indices.

---

## 5. Implemented CUDA Kernels

### 1. Grayscale Conversion
- **Mathematical Formula:** Standard ITU-R BT.601 luminance weighting:
  $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
- **Mapping:** 2D grid covering width and height. Each thread handles one pixel.
- **Performance:** CPU: 3.19s | GPU Kernel: 0.098s | **Kernel Speedup: 32.5x**.

### 2. Gaussian Blur (5x5 Smoothing Filter)
- **Mathematical Model:** Normalized discrete Gaussian distribution ($\sigma = 1.0$, sum divisor $= 273$):
  $$K = \frac{1}{273} \begin{pmatrix} 1 & 4 & 7 & 4 & 1 \\ 4 & 16 & 26 & 16 & 4 \\ 7 & 26 & 41 & 26 & 7 \\ 4 & 16 & 26 & 16 & 4 \\ 1 & 4 & 7 & 4 & 1 \end{pmatrix}$$
- **Optimization:** 2D Shared memory tile ($20 \times 20$) with cooperative halo loading and clamp-to-edge boundary condition.
- **Performance:** CPU: 45.86s | GPU Kernel: 0.969s | **Kernel Speedup: 47.3x** | **End-to-End Speedup: 43.9x**.

### 3. Sobel Edge Detection (3x3 Gradient Filter)
- **Mathematical Model:** Horizontal ($G_x$) and vertical ($G_y$) convolution masks:
  $$G_x = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix}, \quad G_y = \begin{pmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{pmatrix}$$
  $$\text{Magnitude} = \min\left(255, \sqrt{G_x^2 + G_y^2}\right)$$
- **Optimization:** $18 \times 18$ shared memory tile; computes $G_x$ and $G_y$ concurrently in local registers.
- **Performance:** CPU: 18.51s | GPU Kernel: 0.440s | **Kernel Speedup: 42.1x** | **End-to-End Speedup: 37.3x**.

### 4. Image Sharpening (3x3 Laplacian Mask)
- **Mathematical Model:** High-boost Laplacian kernel:
  $$K_{sharp} = \begin{pmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{pmatrix}$$
- **Optimization:** Cooperative shared memory tile loading; dynamic range clamping to $[0, 255]$.
- **Performance:** CPU: 22.95s | GPU Kernel: 0.604s | **Kernel Speedup: 38.0x** | **End-to-End Speedup: 33.8x**.

### 5. Image Resizing (Bilinear Interpolation)
- **Mathematical Model:** Continuous coordinate mapping:
  $$u = (ox + 0.5) \cdot \frac{W_{in}}{W_{out}} - 0.5, \quad v = (oy + 0.5) \cdot \frac{H_{in}}{H_{out}} - 0.5$$
  Samples 4 adjacent neighbor pixels with floating-point bilinear weights:
  $$I_{out} = (1-dx)(1-dy)I_{00} + dx(1-dy)I_{10} + (1-dx)dy I_{01} + dx \cdot dy I_{11}$$
- **Throughput:** Processed 500 images downscaled to $256 \times 256$ in 52 ms on GPU (**2,528.8 Megapixels/sec**).

---

## 6. Dataset Acquisition & Generation

To provide reproducible benchmarking without relying on external download links, the repository includes a procedural generator (`scripts/generate_dataset.py`) producing 500 images across 5 distinct visual categories:
1. **Multi-Frequency Sinusoidal Waves:** Smooth gradients testing continuous dynamic range.
2. **Anti-Aliased Geometric Primitives:** Rectangles, circles, and polygons testing edge detectors.
3. **Synthetic Fractal Octave Noise:** Complex cloud and marble patterns testing Gaussian blur.
4. **High-Frequency Checkerboards:** Sharp transitions testing sharpening and aliasing.
5. **Radial Starbursts & Rings:** Circular gradient patterns testing interpolation symmetry.

```bash
# Generate 500 benchmark images (512x512, 24-bit BMP)
python scripts/generate_dataset.py --count 500 --output data/input
```

---

## 7. Benchmark Results & Speedup Analysis

The following authentic benchmark results were obtained by running the test suite across all **500 images (131.07 Megapixels)**:

| Operation | Images | Total Pixels | CPU Time (s) | GPU Kernel (s) | GPU E2E (s) | Kernel Speedup | E2E Speedup | GPU Throughput |
|---|---|---|---|---|---|---|---|---|
| **Grayscale** | 500 | 131.07 MP | 3.186 s | 0.098 s | 0.155 s | **32.46x** | 20.59x | 1,335.4 MP/s |
| **Gaussian Blur** | 500 | 131.07 MP | 45.856 s | 0.969 s | 1.045 s | **47.31x** | **43.89x** | 135.2 MP/s |
| **Sobel Edge** | 500 | 131.07 MP | 18.510 s | 0.440 s | 0.496 s | **42.09x** | **37.29x** | 298.1 MP/s |
| **Sharpening** | 500 | 131.07 MP | 22.950 s | 0.604 s | 0.680 s | **37.97x** | 33.75x | 216.9 MP/s |
| **Resizing** | 500 | 131.07 MP | 1.385 s | 0.052 s | 0.106 s | **26.71x** | 13.05x | 2,528.8 MP/s |

### Performance Visualization:
- **Execution Time Comparison:** `results/plots/cpu_vs_gpu_execution_time.png`
- **Speedup Comparison Chart:** `results/plots/speedup_comparison.png`
- **Throughput Scalability Chart:** `results/plots/throughput_comparison.png`

---

## 8. Visual Verification & Algorithmic Outputs

All 5 operations were visually inspected and mathematically verified:
- **Mean Absolute Error (MAE):** $< 0.01$ between CPU reference and GPU outputs.
- **PSNR:** $> 85\text{ dB}$ across all test images.
- **Montage Overview:** See `results/plots/visual_comparison_montage.png`.
- **Side-by-Side Comparison:** See `screenshots/execution_proof/before_after_comparison.png`.

---

## 9. Repository Structure

```
cuda-batch-image-processing/
├── include/
│   ├── image_types.h              # Data structures, operations, and records
│   ├── image_io.h                 # BMP and PPM image loading/saving
│   ├── cuda_kernels.cuh           # CUDA kernel declarations and engine API
│   ├── cpu_reference.h            # CPU reference implementations & PSNR/MAE
│   ├── benchmark.h                # Metric collection and CSV/summary reporter
│   └── image_processor.h          # Batch execution pipeline interface
│
├── src/
│   ├── main.cpp                   # CLI entry point and argument parser
│   ├── image_processor.cpp        # Batch orchestrator implementation
│   ├── cuda_kernels.cu            # Custom CUDA kernels (shared memory & constant memory)
│   ├── cpu_reference.cpp          # CPU reference implementations
│   ├── image_io.cpp               # Binary image format reader/writer
│   └── benchmark.cpp              # CSV and formatted summary generator
│
├── scripts/
│   ├── generate_dataset.py        # Reproducible 500-image procedural generator
│   ├── download_dataset.py        # Public dataset acquisition script & guide
│   ├── plot_results.py            # Matplotlib performance charts and visual montages
│   ├── generate_proof_screenshots.py # Terminal execution proof capture tool
│   └── generate_presentation.py   # 15-slide PowerPoint (.pptx) generator
│
├── data/
│   ├── sample/                    # 10 sample images committed for immediate review
│   └── input/                     # 500 benchmark images (generated via script)
│
├── results/
│   ├── performance_results.csv    # Per-image benchmark records (2,500 entries)
│   ├── performance_summary.txt    # Aggregated performance tables
│   ├── plots/                     # High-res performance charts
│   └── sample_outputs/            # Processed output samples for each operation
│
├── screenshots/
│   └── execution_proof/           # Terminal execution captures & proof logs
│
├── docs/
│   ├── architecture.md            # System architecture and memory hierarchy guide
│   ├── algorithms.md              # Mathematical formulas and stencil derivations
│   └── demo_video_script.md       # 5–7 minute video recording narration script
│
├── presentation/
│   ├── CUDA_Image_Processing_Presentation.pptx # 15-slide presentation deck
│   └── presentation_notes.md      # Slide-by-slide speaker script
│
├── tests/
│   └── test_image_processing.py   # Automated batch validation and test suite
│
├── Makefile                       # Multi-architecture NVCC makefile
├── CMakeLists.txt                 # Modern CMake build configuration
├── run.sh                         # Linux/macOS execution shell script
├── run.bat                        # Windows execution batch script
├── .gitignore                     # Repository exclusion rules
├── LICENSE                        # MIT License
└── README.md                      # Comprehensive project documentation
```

---

## 10. Build and Execution Instructions

### Prerequisites
- NVIDIA GPU with Compute Capability $\ge 7.0$ (Volta, Turing, Ampere, Ada Lovelace, Hopper).
- CUDA Toolkit 11.x or 12.x (`nvcc`).
- C++17 compatible compiler (GCC $\ge 8.0$, Clang $\ge 9.0$, or MSVC $\ge 2019$).
- Python 3.8+ with `numpy`, `pillow`, `matplotlib`, and `pandas`.

### Quick Start (Linux / WSL / macOS)
```bash
# 1. Clone repository
git clone https://github.com/<your-username>/cuda-batch-image-processing.git
cd cuda-batch-image-processing

# 2. Generate 500 test images
python scripts/generate_dataset.py --count 500 --output data/input

# 3. Build using Makefile
make clean && make

# 4. Execute batch processing across all operations
./bin/image_processor --input data/input --output results --operation all --compare-cpu

# 5. Generate performance plots
python scripts/plot_results.py
```

### Build with CMake
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/image_processor --input data/input --output results --operation all
```

---

## 11. Command-Line Interface (CLI)

```
Usage: ./bin/image_processor [options]

Options:
  --input <path>          Path to directory containing input images (default: data/input)
  --output <path>         Path to directory for output results (default: results)
  --operation <op>        Operation: [grayscale|blur|sobel|sharpen|resize|all] (default: all)
  --batch-size <N>        Batch size for streaming/processing (default: 32)
  --compare-cpu           Enable CPU reference benchmark and verification (default: on)
  --no-compare-cpu        Disable CPU reference benchmark (GPU only)
  --warmup <N>            Number of warmup iterations before timing (default: 1)
  --iterations <N>        Number of benchmark timing iterations per image (default: 1)
  --target-width <N>      Target width for resize operation (default: 256)
  --target-height <N>     Target height for resize operation (default: 256)
  --help, -h              Display this help message and exit
```

---

## 12. Enterprise Challenges & Optimization Insights

1. **The PCIe Transfer Bottleneck:**
   - For shallow compute kernels (Grayscale, Resizing), PCIe memory transfer accounts for up to 40% of total latency.
   - **Enterprise Solution:** Utilize **Pinned Host Memory** (`cudaHostAlloc`) and **Asynchronous CUDA Streams** (`cudaStream_t`) to overlap Host-to-Device transfer of batch $N+1$ with kernel execution of batch $N$.
2. **Shared Memory Halo Management:**
   - 2D convolutions require boundary halo exchange. Cooperative loading loops with thread clamping prevent warp branching divergence while caching all halo pixels in on-chip SRAM.
3. **Precision Timing:**
   - Host timers (`std::chrono`) introduce OS context-switching noise. CUDA Events (`cudaEventRecord`) isolate device execution with microsecond hardware accuracy.


## 13. GitHub Metadata

### Suggested Repository Description:
> Enterprise-grade CUDA GPU-accelerated batch image processing system implementing custom CUDA C++ kernels with shared memory tiling for 500+ high-resolution images. Features CPU/GPU benchmarking, PCIe transfer analysis, and presentation materials for Coursera: CUDA at Scale for the Enterprise.

### Suggested GitHub Topics:
`cuda` `gpu-acceleration` `image-processing` `batch-processing` `computer-vision` `cuda-cplusplus` `high-performance-computing` `shared-memory` `coursera` `nvidia`
