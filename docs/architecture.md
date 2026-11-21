# System Architecture & Technical Specifications
## CUDA GPU-Accelerated Batch Image Processing System
**Course:** Coursera: CUDA at Scale for the Enterprise — Independent Project

---

## 1. High-Level Architecture Overview

The **CUDA GPU-Accelerated Batch Image Processing System** is designed for high-throughput, latency-optimized batch image processing in enterprise environments. The system processes hundreds of high-resolution images in a single program execution using custom CUDA C++ kernels.

```
+-----------------------------------------------------------------------------------+
|                                HOST (CPU & RAM)                                   |
|                                                                                   |
|  +--------------------+     +---------------------+     +----------------------+  |
|  | Input Image Files  | --> | ImageIO / Decoder   | --> | Host Memory Buffers  |  |
|  | (500x .bmp Images) |     | Directory Discovery |     | (Pinned / Raw RGB)   |  |
|  +--------------------+     +---------------------+     +----------------------+  |
+----------------------------------------|------------------------------------------+
                                         |  PCIe Host-to-Device Transfer (cudaMemcpy)
                                         v
+-----------------------------------------------------------------------------------+
|                                DEVICE (NVIDIA GPU)                                |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | Global Device Memory (d_in: RGB24 / Gray8 Buffers)                          |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | Streaming Multiprocessors (SMs)                                             |  |
|  |                                                                             |  |
|  |  +-----------------------+     +--------------------+                       |  |
|  |  | On-Chip Shared Memory |     | Constant Memory    |                       |  |
|  |  | (2D Tiled Halo Apron) |     | (64 KB Filter Mask)|                       |  |
|  |  +-----------------------+     +--------------------+                       |  |
|  |             |                            |                                  |  |
|  |             +------------+---------------+                                  |  |
|  |                          v                                                  |  |
|  |  +-----------------------------------------------------------------------+  |  |
|  |  | Custom CUDA Compute Kernels:                                          |  |  |
|  |  | 1. rgbToGrayscaleKernel        (BT.601 Luminance SIMT)                |  |  |
|  |  | 2. gaussianBlurSharedKernel    (5x5 Tiled Convolution, R=2)           |  |  |
|  |  | 3. sobelSharedKernel           (3x3 Gradient Vector & Mag)            |  |  |
|  |  | 4. sharpenSharedKernel         (3x3 Laplacian Unsharp Mask)           |  |  |
|  |  | 5. resizeBilinearKernel        (4-Tap Bilinear Interpolation)         |  |  |
|  |  +-----------------------------------------------------------------------+  |  |
|  +-------------------------------------|---------------------------------------+  |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | Global Device Memory (d_out: Processed Output Buffers)                      |  |
|  +-----------------------------------------------------------------------------+  |
+----------------------------------------|------------------------------------------+
                                         |  PCIe Device-to-Host Transfer (cudaMemcpy)
                                         v
+-----------------------------------------------------------------------------------+
|                                HOST OUTPUT STAGE                                  |
|                                                                                   |
|  +----------------------+     +---------------------+     +--------------------+  |
|  | Performance Analyzer | <-> | CPU Reference Engine| --> | Disk Output / CSV  |  |
|  | (CUDA Events Timing) |     | (Verification/PSNR) |     | & Summary Reports  |  |
|  +----------------------+     +---------------------+     +--------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 2. Memory Hierarchy & Optimization Strategy

Enterprise GPU applications must optimize across the distinct tiers of the NVIDIA memory hierarchy:

| Memory Space | Location | Scope | Latency | Bandwidth | System Role in this Project |
|---|---|---|---|---|---|
| **Global Memory** | Off-chip DRAM (VRAM) | Grid-wide (all threads) | 200–400 cycles | 500–1000+ GB/s | Holds raw input images and final output buffers. |
| **Constant Memory** | On-chip Cache backed by VRAM | Grid-wide (read-only) | ~1 cycle (on hit) | Multi-TB/s (warp broadcast) | Stores fixed stencil masks (Gaussian 5x5, Sobel 3x3, Sharpen 3x3). |
| **Shared Memory** | On-chip SRAM per SM | Block-wide (threads in block) | ~1-5 cycles | >10 TB/s aggregate | 2D Tiled Aprons for stencil filters, caching halo pixels. |
| **Registers** | On-chip Register File | Single Thread | 0 cycles | Tens of TB/s | Holds pixel coordinates, interpolation weights, and intermediate sums. |

### 2.1 2D Shared Memory Tiling
In stencil operations (Gaussian Blur, Sobel, Sharpening), neighboring threads access overlapping pixels. For a 5x5 filter, unoptimized global memory reads result in **25 reads per pixel**.

By allocating a shared memory tile with a border apron of radius $R$:
$$\text{Shared Tile Dimensions} = (TILE\_DIM + 2R) \times (TILE\_DIM + 2R)$$

For $TILE\_DIM = 16$ and $R = 2$:
$$\text{Tile Size} = 20 \times 20 = 400 \text{ elements (1.6 KB per block)}$$

Threads in the block cooperatively load the interior and halo pixels into `__shared__` memory, execute `__syncthreads()`, and compute the entire 2D convolution from on-chip SRAM. This yields a **25x reduction in global memory traffic**.

---

## 3. Thread Block & Grid Sizing

Image processing maps naturally to 2D thread blocks and 2D execution grids:
- **Block Dimensions:** $16 \times 16 = 256$ threads per block.
  - Aligns cleanly with NVIDIA warp size (32 threads: 8 warps per block).
  - Maintains high SM occupancy and leaves sufficient registers/shared memory per block.
- **Grid Dimensions:**
  $$\text{Grid.x} = \left\lceil \frac{\text{Width}}{16} \right\rceil, \quad \text{Grid.y} = \left\lceil \frac{\text{Height}}{16} \right\rceil$$
  For a $512 \times 512$ image:
  $$\text{Grid} = 32 \times 32 = 1,024 \text{ blocks (262,144 total active threads)}$$

Every thread checks boundary guards:
```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < width && y < height) { ... }
```

---

## 4. Hardware Timing Methodology (CUDA Events)

Host-side timers (such as `std::chrono`) measure CPU-to-GPU asynchronous launch overhead and OS context switches rather than true GPU execution time.

To ensure scientific accuracy, our system uses hardware **CUDA Events**:
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
// ... Kernel Launch ...
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float kernelMs = 0.0f;
cudaEventElapsedTime(&kernelMs, start, stop);
```
Host-to-Device (H2D) and Device-to-Host (D2H) memory transfers are timed independently to isolate raw kernel throughput from PCIe bus latency.
