#ifndef IMAGE_TYPES_H
#define IMAGE_TYPES_H

#include <vector>
#include <string>
#include <cstdint>

namespace cuda_image {

enum class Operation {
    GRAYSCALE,
    GAUSSIAN_BLUR,
    SOBEL_EDGE,
    SHARPEN,
    RESIZE,
    ALL
};

inline std::string operationToString(Operation op) {
    switch (op) {
        case Operation::GRAYSCALE: return "grayscale";
        case Operation::GAUSSIAN_BLUR: return "blur";
        case Operation::SOBEL_EDGE: return "sobel";
        case Operation::SHARPEN: return "sharpen";
        case Operation::RESIZE: return "resize";
        case Operation::ALL: return "all";
        default: return "unknown";
    }
}

inline Operation stringToOperation(const std::string& str) {
    if (str == "grayscale" || str == "gray") return Operation::GRAYSCALE;
    if (str == "blur" || str == "gaussian") return Operation::GAUSSIAN_BLUR;
    if (str == "sobel" || str == "edges" || str == "edge") return Operation::SOBEL_EDGE;
    if (str == "sharpen") return Operation::SHARPEN;
    if (str == "resize") return Operation::RESIZE;
    if (str == "all") return Operation::ALL;
    return Operation::ALL;
}

struct RawImage {
    int width = 0;
    int height = 0;
    int channels = 0; // 1 = Grayscale, 3 = RGB, 4 = RGBA
    std::vector<uint8_t> data;
    std::string filename;

    size_t sizeBytes() const {
        return static_cast<size_t>(width) * height * channels;
    }

    bool isValid() const {
        return width > 0 && height > 0 && channels > 0 && data.size() == sizeBytes();
    }
};

struct BenchmarkRecord {
    std::string operationName;
    std::string filename;
    int width;
    int height;
    int channels;
    int outputWidth;
    int outputHeight;
    int outputChannels;
    double cpuTimeMs;
    double gpuKernelTimeMs;
    double gpuH2DTimeMs;
    double gpuD2HTimeMs;
    double gpuEndToEndTimeMs;
    double speedupKernel;
    double speedupEndToEnd;
};

} // namespace cuda_image

#endif // IMAGE_TYPES_H
