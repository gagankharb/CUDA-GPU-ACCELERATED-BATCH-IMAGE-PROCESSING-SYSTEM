#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "image_types.h"
#include <string>

namespace cuda_image {

// Structure holding performance metrics for a single GPU operation
struct GpuMetrics {
    double kernelTimeMs = 0.0;
    double h2dTimeMs = 0.0;
    double d2hTimeMs = 0.0;
    double totalTimeMs = 0.0;
};

class CudaKernelEngine {
public:
    static bool init();
    static void cleanup();
    static void printDeviceInfo();
    static bool isGpuAvailable();

    // 1. Grayscale Conversion
    static bool runGrayscale(const RawImage& input, RawImage& output, GpuMetrics& metrics);

    // 2. Gaussian Blur (5x5 filter with Shared Memory Tiling)
    static bool runGaussianBlur(const RawImage& input, RawImage& output, GpuMetrics& metrics);

    // 3. Sobel Edge Detection (3x3 filter with Shared Memory Tiling)
    static bool runSobelEdge(const RawImage& input, RawImage& output, GpuMetrics& metrics);

    // 4. Image Sharpening (3x3 Laplacian / Unsharp Filter with Shared Memory Tiling)
    static bool runSharpen(const RawImage& input, RawImage& output, GpuMetrics& metrics);

    // 5. Image Resizing (Bilinear Interpolation)
    static bool runResize(const RawImage& input, RawImage& output, int targetWidth, int targetHeight, GpuMetrics& metrics);
};

} // namespace cuda_image

#endif // CUDA_KERNELS_CUH
