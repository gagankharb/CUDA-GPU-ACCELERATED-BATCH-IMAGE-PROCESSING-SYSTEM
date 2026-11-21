#include "cpu_reference.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace cuda_image {

static const float CPU_GAUSSIAN_5X5[25] = {
    1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f,
    4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
    7.0f/273.0f, 26.0f/273.0f, 41.0f/273.0f, 26.0f/273.0f, 7.0f/273.0f,
    4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
    1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f
};

static const float CPU_SOBEL_X[9] = {
    -1.0f, 0.0f, 1.0f,
    -2.0f, 0.0f, 2.0f,
    -1.0f, 0.0f, 1.0f
};

static const float CPU_SOBEL_Y[9] = {
    -1.0f, -2.0f, -1.0f,
     0.0f,  0.0f,  0.0f,
     1.0f,  2.0f,  1.0f
};

static const float CPU_SHARPEN[9] = {
     0.0f, -1.0f,  0.0f,
    -1.0f,  5.0f, -1.0f,
     0.0f, -1.0f,  0.0f
};

bool CpuReferenceEngine::runGrayscale(const RawImage& input, RawImage& output, double& elapsedMs) {
    auto t0 = std::chrono::high_resolution_clock::now();

    output.width = input.width;
    output.height = input.height;
    output.channels = 1;
    output.filename = input.filename;
    output.data.resize(output.sizeBytes());

    int inChannels = input.channels;
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            int inIdx = (y * input.width + x) * inChannels;
            float r = input.data[inIdx + 0];
            float g = input.data[inIdx + 1];
            float b = input.data[inIdx + 2];
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;
            output.data[y * input.width + x] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, gray + 0.5f)));
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

bool CpuReferenceEngine::runGaussianBlur(const RawImage& input, RawImage& output, double& elapsedMs) {
    auto t0 = std::chrono::high_resolution_clock::now();

    output = input;
    int W = input.width;
    int H = input.height;
    int C = input.channels;
    int R = 2; // 5x5 kernel has radius 2

    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float sum = 0.0f;
                for (int ky = -R; ky <= R; ++ky) {
                    for (int kx = -R; kx <= R; ++kx) {
                        int px = std::clamp(x + kx, 0, W - 1);
                        int py = std::clamp(y + ky, 0, H - 1);
                        float weight = CPU_GAUSSIAN_5X5[(ky + R) * 5 + (kx + R)];
                        sum += weight * input.data[(py * W + px) * C + c];
                    }
                }
                output.data[(y * W + x) * C + c] = static_cast<uint8_t>(std::clamp(sum + 0.5f, 0.0f, 255.0f));
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

bool CpuReferenceEngine::runSobelEdge(const RawImage& input, RawImage& output, double& elapsedMs) {
    auto t0 = std::chrono::high_resolution_clock::now();

    RawImage grayInput;
    double grayElapsed = 0.0;
    if (input.channels > 1) {
        runGrayscale(input, grayInput, grayElapsed);
    } else {
        grayInput = input;
    }

    output.width = grayInput.width;
    output.height = grayInput.height;
    output.channels = 1;
    output.filename = input.filename;
    output.data.resize(output.sizeBytes());

    int W = grayInput.width;
    int H = grayInput.height;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float gx = 0.0f;
            float gy = 0.0f;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int px = std::clamp(x + kx, 0, W - 1);
                    int py = std::clamp(y + ky, 0, H - 1);
                    float val = grayInput.data[py * W + px];
                    int kIdx = (ky + 1) * 3 + (kx + 1);
                    gx += CPU_SOBEL_X[kIdx] * val;
                    gy += CPU_SOBEL_Y[kIdx] * val;
                }
            }

            float magnitude = std::sqrt(gx * gx + gy * gy);
            output.data[y * W + x] = static_cast<uint8_t>(std::clamp(magnitude, 0.0f, 255.0f));
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count() + grayElapsed;
    return true;
}

bool CpuReferenceEngine::runSharpen(const RawImage& input, RawImage& output, double& elapsedMs) {
    auto t0 = std::chrono::high_resolution_clock::now();

    output = input;
    int W = input.width;
    int H = input.height;
    int C = input.channels;

    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int px = std::clamp(x + kx, 0, W - 1);
                        int py = std::clamp(y + ky, 0, H - 1);
                        int kIdx = (ky + 1) * 3 + (kx + 1);
                        sum += CPU_SHARPEN[kIdx] * input.data[(py * W + px) * C + c];
                    }
                }
                output.data[(y * W + x) * C + c] = static_cast<uint8_t>(std::clamp(sum + 0.5f, 0.0f, 255.0f));
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

bool CpuReferenceEngine::runResize(const RawImage& input, RawImage& output, int targetWidth, int targetHeight, double& elapsedMs) {
    auto t0 = std::chrono::high_resolution_clock::now();

    output.width = targetWidth;
    output.height = targetHeight;
    output.channels = input.channels;
    output.filename = input.filename;
    output.data.resize(output.sizeBytes());

    int inW = input.width;
    int inH = input.height;
    int C = input.channels;

    for (int oy = 0; oy < targetHeight; ++oy) {
        for (int ox = 0; ox < targetWidth; ++ox) {
            float u = (static_cast<float>(ox) + 0.5f) * (static_cast<float>(inW) / static_cast<float>(targetWidth)) - 0.5f;
            float v = (static_cast<float>(oy) + 0.5f) * (static_cast<float>(inH) / static_cast<float>(targetHeight)) - 0.5f;

            u = std::clamp(u, 0.0f, static_cast<float>(inW - 1));
            v = std::clamp(v, 0.0f, static_cast<float>(inH - 1));

            int x0 = static_cast<int>(std::floor(u));
            int y0 = static_cast<int>(std::floor(v));
            int x1 = std::min(x0 + 1, inW - 1);
            int y1 = std::min(y0 + 1, inH - 1);

            float dx = u - static_cast<float>(x0);
            float dy = v - static_cast<float>(y0);

            float w00 = (1.0f - dx) * (1.0f - dy);
            float w10 = dx * (1.0f - dy);
            float w01 = (1.0f - dx) * dy;
            float w11 = dx * dy;

            for (int c = 0; c < C; ++c) {
                float p00 = input.data[(y0 * inW + x0) * C + c];
                float p10 = input.data[(y0 * inW + x1) * C + c];
                float p01 = input.data[(y1 * inW + x0) * C + c];
                float p11 = input.data[(y1 * inW + x1) * C + c];

                float val = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;
                output.data[(oy * targetWidth + ox) * C + c] = static_cast<uint8_t>(std::clamp(val + 0.5f, 0.0f, 255.0f));
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

double CpuReferenceEngine::computeMAE(const RawImage& imgA, const RawImage& imgB) {
    if (imgA.width != imgB.width || imgA.height != imgB.height || imgA.channels != imgB.channels) {
        return -1.0;
    }
    double totalDiff = 0.0;
    size_t n = imgA.data.size();
    for (size_t i = 0; i < n; ++i) {
        totalDiff += std::abs(static_cast<double>(imgA.data[i]) - static_cast<double>(imgB.data[i]));
    }
    return totalDiff / static_cast<double>(n);
}

double CpuReferenceEngine::computePSNR(const RawImage& imgA, const RawImage& imgB) {
    if (imgA.width != imgB.width || imgA.height != imgB.height || imgA.channels != imgB.channels) {
        return -1.0;
    }
    double mse = 0.0;
    size_t n = imgA.data.size();
    for (size_t i = 0; i < n; ++i) {
        double diff = static_cast<double>(imgA.data[i]) - static_cast<double>(imgB.data[i]);
        mse += diff * diff;
    }
    mse /= static_cast<double>(n);
    if (mse < 1e-10) return 100.0; // Infinite / identical
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

} // namespace cuda_image
