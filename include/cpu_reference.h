#ifndef CPU_REFERENCE_H
#define CPU_REFERENCE_H

#include "image_types.h"

namespace cuda_image {

class CpuReferenceEngine {
public:
    // 1. Grayscale Conversion (ITU-R BT.601)
    static bool runGrayscale(const RawImage& input, RawImage& output, double& elapsedMs);

    // 2. Gaussian Blur (5x5 filter, sigma = 1.0)
    static bool runGaussianBlur(const RawImage& input, RawImage& output, double& elapsedMs);

    // 3. Sobel Edge Detection (3x3 operator, gradient magnitude)
    static bool runSobelEdge(const RawImage& input, RawImage& output, double& elapsedMs);

    // 4. Image Sharpening (3x3 Laplacian mask)
    static bool runSharpen(const RawImage& input, RawImage& output, double& elapsedMs);

    // 5. Image Resizing (Bilinear Interpolation)
    static bool runResize(const RawImage& input, RawImage& output, int targetWidth, int targetHeight, double& elapsedMs);

    // Compare two images (CPU vs GPU) for correctness verification (returns Mean Absolute Error)
    static double computeMAE(const RawImage& imgA, const RawImage& imgB);
    static double computePSNR(const RawImage& imgA, const RawImage& imgB);
};

} // namespace cuda_image

#endif // CPU_REFERENCE_H
