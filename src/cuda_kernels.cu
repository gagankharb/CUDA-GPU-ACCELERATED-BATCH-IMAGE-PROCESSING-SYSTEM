#include "cuda_kernels.cuh"
#include <iostream>
#include <cmath>
#include <cstring>

#ifdef __CUDACC__
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA Error] " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while (0)

#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA Error] " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return; \
        } \
    } while (0)

namespace cuda_image {

// Constants in device constant memory for optimal broadcast cache access
__constant__ float c_gaussianKernel5x5[25];
__constant__ float c_sobelX[9];
__constant__ float c_sobelY[9];
__constant__ float c_sharpen3x3[9];

// Tile configuration
constexpr int TILE_DIM = 16;
constexpr int GAUSSIAN_RADIUS = 2;
constexpr int GAUSSIAN_SHARED_DIM = TILE_DIM + 2 * GAUSSIAN_RADIUS; // 20x20

constexpr int SOBEL_RADIUS = 1;
constexpr int SOBEL_SHARED_DIM = TILE_DIM + 2 * SOBEL_RADIUS;       // 18x18

constexpr int SHARPEN_RADIUS = 1;
constexpr int SHARPEN_SHARED_DIM = TILE_DIM + 2 * SHARPEN_RADIUS;   // 18x18

// =========================================================================
// 1. Grayscale Conversion Kernel
// Standard ITU-R BT.601: Y = 0.299 R + 0.587 G + 0.114 B
// =========================================================================
__global__ void rgbToGrayscaleKernel(const unsigned char* __restrict__ d_in,
                                     unsigned char* __restrict__ d_out,
                                     int width, int height, int inChannels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int inIdx = (y * width + x) * inChannels;
        float r = static_cast<float>(d_in[inIdx]);
        float g = static_cast<float>(d_in[inIdx + 1]);
        float b = static_cast<float>(d_in[inIdx + 2]);

        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        d_out[y * width + x] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, gray + 0.5f)));
    }
}

// =========================================================================
// 2. Gaussian Blur Kernel with Shared Memory Tiling (5x5 filter)
// =========================================================================
__global__ void gaussianBlurSharedKernel(const unsigned char* __restrict__ d_in,
                                         unsigned char* __restrict__ d_out,
                                         int width, int height, int channels) {
    __shared__ float s_tile[GAUSSIAN_SHARED_DIM][GAUSSIAN_SHARED_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE_DIM + tx;
    int gy = blockIdx.y * TILE_DIM + ty;

    // Process channel-by-channel or assume single/multi-channel
    // We launch separate passes or run per channel
    int c = blockIdx.z;

    // Cooperative tile loading into shared memory
    // Each thread loads multiple elements to cover the (TILE_DIM + 2*R)^2 apron
    for (int sy = ty; sy < GAUSSIAN_SHARED_DIM; sy += TILE_DIM) {
        for (int sx = tx; sx < GAUSSIAN_SHARED_DIM; sx += TILE_DIM) {
            int loadX = blockIdx.x * TILE_DIM + sx - GAUSSIAN_RADIUS;
            int loadY = blockIdx.y * TILE_DIM + sy - GAUSSIAN_RADIUS;

            // Clamp boundary conditions
            loadX = max(0, min(loadX, width - 1));
            loadY = max(0, min(loadY, height - 1));

            int pixelIdx = (loadY * width + loadX) * channels + c;
            s_tile[sy][sx] = static_cast<float>(d_in[pixelIdx]);
        }
    }

    __syncthreads();

    // Compute 5x5 convolution from shared memory
    if (gx < width && gy < height) {
        float sum = 0.0f;
        #pragma unroll
        for (int ky = -GAUSSIAN_RADIUS; ky <= GAUSSIAN_RADIUS; ++ky) {
            #pragma unroll
            for (int kx = -GAUSSIAN_RADIUS; kx <= GAUSSIAN_RADIUS; ++kx) {
                float weight = c_gaussianKernel5x5[(ky + GAUSSIAN_RADIUS) * 5 + (kx + GAUSSIAN_RADIUS)];
                float val = s_tile[ty + GAUSSIAN_RADIUS + ky][tx + GAUSSIAN_RADIUS + kx];
                sum += weight * val;
            }
        }

        int outIdx = (gy * width + gx) * channels + c;
        d_out[outIdx] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, sum + 0.5f)));
    }
}

// =========================================================================
// 3. Sobel Edge Detection Kernel with Shared Memory Tiling (3x3 operator)
// =========================================================================
__global__ void sobelSharedKernel(const unsigned char* __restrict__ d_in,
                                  unsigned char* __restrict__ d_out,
                                  int width, int height) {
    __shared__ float s_tile[SOBEL_SHARED_DIM][SOBEL_SHARED_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE_DIM + tx;
    int gy = blockIdx.y * TILE_DIM + ty;

    // Cooperative loading into shared tile
    for (int sy = ty; sy < SOBEL_SHARED_DIM; sy += TILE_DIM) {
        for (int sx = tx; sx < SOBEL_SHARED_DIM; sx += TILE_DIM) {
            int loadX = blockIdx.x * TILE_DIM + sx - SOBEL_RADIUS;
            int loadY = blockIdx.y * TILE_DIM + sy - SOBEL_RADIUS;

            loadX = max(0, min(loadX, width - 1));
            loadY = max(0, min(loadY, height - 1));

            s_tile[sy][sx] = static_cast<float>(d_in[loadY * width + loadX]);
        }
    }

    __syncthreads();

    if (gx < width && gy < height) {
        float gxVal = 0.0f;
        float gyVal = 0.0f;

        #pragma unroll
        for (int ky = -SOBEL_RADIUS; ky <= SOBEL_RADIUS; ++ky) {
            #pragma unroll
            for (int kx = -SOBEL_RADIUS; kx <= SOBEL_RADIUS; ++kx) {
                float val = s_tile[ty + SOBEL_RADIUS + ky][tx + SOBEL_RADIUS + kx];
                int kIdx = (ky + SOBEL_RADIUS) * 3 + (kx + SOBEL_RADIUS);
                gxVal += c_sobelX[kIdx] * val;
                gyVal += c_sobelY[kIdx] * val;
            }
        }

        float magnitude = sqrtf(gxVal * gxVal + gyVal * gyVal);
        d_out[gy * width + gx] = static_cast<unsigned char>(fminf(255.0f, magnitude));
    }
}

// =========================================================================
// 4. Image Sharpening Kernel with Shared Memory Tiling (3x3 Laplacian)
// =========================================================================
__global__ void sharpenSharedKernel(const unsigned char* __restrict__ d_in,
                                    unsigned char* __restrict__ d_out,
                                    int width, int height, int channels) {
    __shared__ float s_tile[SHARPEN_SHARED_DIM][SHARPEN_SHARED_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE_DIM + tx;
    int gy = blockIdx.y * TILE_DIM + ty;
    int c = blockIdx.z;

    for (int sy = ty; sy < SHARPEN_SHARED_DIM; sy += TILE_DIM) {
        for (int sx = tx; sx < SHARPEN_SHARED_DIM; sx += TILE_DIM) {
            int loadX = blockIdx.x * TILE_DIM + sx - SHARPEN_RADIUS;
            int loadY = blockIdx.y * TILE_DIM + sy - SHARPEN_RADIUS;

            loadX = max(0, min(loadX, width - 1));
            loadY = max(0, min(loadY, height - 1));

            s_tile[sy][sx] = static_cast<float>(d_in[(loadY * width + loadX) * channels + c]);
        }
    }

    __syncthreads();

    if (gx < width && gy < height) {
        float sum = 0.0f;
        #pragma unroll
        for (int ky = -SHARPEN_RADIUS; ky <= SHARPEN_RADIUS; ++ky) {
            #pragma unroll
            for (int kx = -SHARPEN_RADIUS; kx <= SHARPEN_RADIUS; ++kx) {
                int kIdx = (ky + SHARPEN_RADIUS) * 3 + (kx + SHARPEN_RADIUS);
                sum += c_sharpen3x3[kIdx] * s_tile[ty + SHARPEN_RADIUS + ky][tx + SHARPEN_RADIUS + kx];
            }
        }

        int outIdx = (gy * width + gx) * channels + c;
        d_out[outIdx] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, sum + 0.5f)));
    }
}

// =========================================================================
// 5. Image Resizing Kernel (Bilinear Interpolation)
// =========================================================================
__global__ void resizeBilinearKernel(const unsigned char* __restrict__ d_in,
                                     unsigned char* __restrict__ d_out,
                                     int inW, int inH,
                                     int outW, int outH,
                                     int channels) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox < outW && oy < outH) {
        // Map output pixel center to input continuous coordinates
        float u = (static_cast<float>(ox) + 0.5f) * (static_cast<float>(inW) / static_cast<float>(outW)) - 0.5f;
        float v = (static_cast<float>(oy) + 0.5f) * (static_cast<float>(inH) / static_cast<float>(outH)) - 0.5f;

        u = fmaxf(0.0f, fminf(u, static_cast<float>(inW - 1)));
        v = fmaxf(0.0f, fminf(v, static_cast<float>(inH - 1)));

        int x0 = static_cast<int>(floorf(u));
        int y0 = static_cast<int>(floorf(v));
        int x1 = min(x0 + 1, inW - 1);
        int y1 = min(y0 + 1, inH - 1);

        float dx = u - static_cast<float>(x0);
        float dy = v - static_cast<float>(y0);

        float w00 = (1.0f - dx) * (1.0f - dy);
        float w10 = dx * (1.0f - dy);
        float w01 = (1.0f - dx) * dy;
        float w11 = dx * dy;

        for (int c = 0; c < channels; ++c) {
            float p00 = static_cast<float>(d_in[(y0 * inW + x0) * channels + c]);
            float p10 = static_cast<float>(d_in[(y0 * inW + x1) * channels + c]);
            float p01 = static_cast<float>(d_in[(y1 * inW + x0) * channels + c]);
            float p11 = static_cast<float>(d_in[(y1 * inW + x1) * channels + c]);

            float val = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;
            d_out[(oy * outW + ox) * channels + c] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, val + 0.5f)));
        }
    }
}

// =========================================================================
// Host Interface Implementation
// =========================================================================

bool CudaKernelEngine::isGpuAvailable() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
}

bool CudaKernelEngine::init() {
    if (!isGpuAvailable()) {
        std::cerr << "[CUDA] No compatible CUDA device found." << std::endl;
        return false;
    }

    CUDA_CHECK(cudaSetDevice(0));

    // Upload filter constants to device constant memory
    // 5x5 Normalized Gaussian Kernel (sigma = 1.0)
    float h_gaussian[25] = {
        1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f,
        4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
        7.0f/273.0f, 26.0f/273.0f, 41.0f/273.0f, 26.0f/273.0f, 7.0f/273.0f,
        4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
        1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f
    };
    CUDA_CHECK(cudaMemcpyToSymbol(c_gaussianKernel5x5, h_gaussian, sizeof(h_gaussian)));

    // Sobel Operators
    float h_sobelX[9] = {
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f
    };
    float h_sobelY[9] = {
        -1.0f, -2.0f, -1.0f,
         0.0f,  0.0f,  0.0f,
         1.0f,  2.0f,  1.0f
    };
    CUDA_CHECK(cudaMemcpyToSymbol(c_sobelX, h_sobelX, sizeof(h_sobelX)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_sobelY, h_sobelY, sizeof(h_sobelY)));

    // 3x3 Laplacian Sharpening Mask
    float h_sharpen[9] = {
         0.0f, -1.0f,  0.0f,
        -1.0f,  5.0f, -1.0f,
         0.0f, -1.0f,  0.0f
    };
    CUDA_CHECK(cudaMemcpyToSymbol(c_sharpen3x3, h_sharpen, sizeof(h_sharpen)));

    return true;
}

void CudaKernelEngine::cleanup() {
    cudaDeviceReset();
}

void CudaKernelEngine::printDeviceInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "[CUDA] No CUDA capable devices detected." << std::endl;
        return;
    }

    std::cout << "=========================================================\n";
    std::cout << "              CUDA GPU HARDWARE INFORMATION              \n";
    std::cout << "=========================================================\n";
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        std::cout << "Device ID:                  " << dev << "\n";
        std::cout << "Device Name:                " << prop.name << "\n";
        std::cout << "Compute Capability:         " << prop.major << "." << prop.minor << "\n";
        std::cout << "Streaming Multiprocessors:  " << prop.multiProcessorCount << "\n";
        std::cout << "Total Global Memory:        " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
        std::cout << "Shared Memory per Block:    " << (prop.sharedMemPerBlock / 1024) << " KB\n";
        std::cout << "Max Threads per Block:      " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Warp Size:                  " << prop.warpSize << "\n";
        std::cout << "Memory Clock Rate:          " << (prop.memoryClockRate * 1e-3f) << " MHz\n";
        std::cout << "Memory Bus Width:           " << prop.memoryBusWidth << " bits\n";
        std::cout << "=========================================================\n";
    }
}

bool CudaKernelEngine::runGrayscale(const RawImage& input, RawImage& output, GpuMetrics& metrics) {
    output.width = input.width;
    output.height = input.height;
    output.channels = 1;
    output.filename = input.filename;
    output.data.resize(output.sizeBytes());

    size_t inBytes = input.sizeBytes();
    size_t outBytes = output.sizeBytes();

    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;

    cudaEvent_t startH2D, stopH2D, startKernel, stopKernel, startD2H, stopD2H;
    cudaEventCreate(&startH2D);
    cudaEventCreate(&stopH2D);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);
    cudaEventCreate(&startD2H);
    cudaEventCreate(&stopD2H);

    CUDA_CHECK(cudaMalloc(&d_in, inBytes));
    CUDA_CHECK(cudaMalloc(&d_out, outBytes));

    // Host-to-Device transfer timing
    cudaEventRecord(startH2D);
    CUDA_CHECK(cudaMemcpy(d_in, input.data.data(), inBytes, cudaMemcpyHostToDevice));
    cudaEventRecord(stopH2D);
    cudaEventSynchronize(stopH2D);

    // Kernel execution timing
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((input.width + block.x - 1) / block.x, (input.height + block.y - 1) / block.y);

    cudaEventRecord(startKernel);
    rgbToGrayscaleKernel<<<grid, block>>>(d_in, d_out, input.width, input.height, input.channels);
    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);

    // Device-to-Host transfer timing
    cudaEventRecord(startD2H);
    CUDA_CHECK(cudaMemcpy(output.data.data(), d_out, outBytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stopD2H);
    cudaEventSynchronize(stopD2H);

    float h2dMs = 0.0f, kernelMs = 0.0f, d2hMs = 0.0f;
    cudaEventElapsedTime(&h2dMs, startH2D, stopH2D);
    cudaEventElapsedTime(&kernelMs, startKernel, stopKernel);
    cudaEventElapsedTime(&d2hMs, startD2H, stopD2H);

    metrics.h2dTimeMs = h2dMs;
    metrics.kernelTimeMs = kernelMs;
    metrics.d2hTimeMs = d2hMs;
    metrics.totalTimeMs = h2dMs + kernelMs + d2hMs;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(startH2D);
    cudaEventDestroy(stopH2D);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);
    cudaEventDestroy(startD2H);
    cudaEventDestroy(stopD2H);

    return true;
}

bool CudaKernelEngine::runGaussianBlur(const RawImage& input, RawImage& output, GpuMetrics& metrics) {
    output = input;
    size_t bytes = input.sizeBytes();

    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;

    cudaEvent_t e0, e1, e2, e3, e4, e5;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventCreate(&e4); cudaEventCreate(&e5);

    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    cudaEventRecord(e0);
    CUDA_CHECK(cudaMemcpy(d_in, input.data.data(), bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((input.width + block.x - 1) / block.x,
              (input.height + block.y - 1) / block.y,
              input.channels);

    cudaEventRecord(e2);
    gaussianBlurSharedKernel<<<grid, block>>>(d_in, d_out, input.width, input.height, input.channels);
    cudaEventRecord(e3);
    cudaEventSynchronize(e3);

    cudaEventRecord(e4);
    CUDA_CHECK(cudaMemcpy(output.data.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(e5);
    cudaEventSynchronize(e5);

    float h2dMs = 0, kMs = 0, d2hMs = 0;
    cudaEventElapsedTime(&h2dMs, e0, e1);
    cudaEventElapsedTime(&kMs, e2, e3);
    cudaEventElapsedTime(&d2hMs, e4, e5);

    metrics.h2dTimeMs = h2dMs;
    metrics.kernelTimeMs = kMs;
    metrics.d2hTimeMs = d2hMs;
    metrics.totalTimeMs = h2dMs + kMs + d2hMs;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaEventDestroy(e4); cudaEventDestroy(e5);

    return true;
}

bool CudaKernelEngine::runSobelEdge(const RawImage& input, RawImage& output, GpuMetrics& metrics) {
    // Sobel expects single channel input
    RawImage grayInput;
    GpuMetrics grayMetrics;
    if (input.channels > 1) {
        runGrayscale(input, grayInput, grayMetrics);
    } else {
        grayInput = input;
    }

    output.width = grayInput.width;
    output.height = grayInput.height;
    output.channels = 1;
    output.filename = input.filename;
    output.data.resize(output.sizeBytes());

    size_t inBytes = grayInput.sizeBytes();
    size_t outBytes = output.sizeBytes();

    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;

    cudaEvent_t e0, e1, e2, e3, e4, e5;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventCreate(&e4); cudaEventCreate(&e5);

    CUDA_CHECK(cudaMalloc(&d_in, inBytes));
    CUDA_CHECK(cudaMalloc(&d_out, outBytes));

    cudaEventRecord(e0);
    CUDA_CHECK(cudaMemcpy(d_in, grayInput.data.data(), inBytes, cudaMemcpyHostToDevice));
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((grayInput.width + block.x - 1) / block.x, (grayInput.height + block.y - 1) / block.y);

    cudaEventRecord(e2);
    sobelSharedKernel<<<grid, block>>>(d_in, d_out, grayInput.width, grayInput.height);
    cudaEventRecord(e3);
    cudaEventSynchronize(e3);

    cudaEventRecord(e4);
    CUDA_CHECK(cudaMemcpy(output.data.data(), d_out, outBytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(e5);
    cudaEventSynchronize(e5);

    float h2dMs = 0, kMs = 0, d2hMs = 0;
    cudaEventElapsedTime(&h2dMs, e0, e1);
    cudaEventElapsedTime(&kMs, e2, e3);
    cudaEventElapsedTime(&d2hMs, e4, e5);

    metrics.h2dTimeMs = h2dMs + grayMetrics.h2dTimeMs;
    metrics.kernelTimeMs = kMs + grayMetrics.kernelTimeMs;
    metrics.d2hTimeMs = d2hMs + grayMetrics.d2hTimeMs;
    metrics.totalTimeMs = metrics.h2dTimeMs + metrics.kernelTimeMs + metrics.d2hTimeMs;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaEventDestroy(e4); cudaEventDestroy(e5);

    return true;
}

bool CudaKernelEngine::runSharpen(const RawImage& input, RawImage& output, GpuMetrics& metrics) {
    output = input;
    size_t bytes = input.sizeBytes();

    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;

    cudaEvent_t e0, e1, e2, e3, e4, e5;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventCreate(&e4); cudaEventCreate(&e5);

    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    cudaEventRecord(e0);
    CUDA_CHECK(cudaMemcpy(d_in, input.data.data(), bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((input.width + block.x - 1) / block.x,
              (input.height + block.y - 1) / block.y,
              input.channels);

    cudaEventRecord(e2);
    sharpenSharedKernel<<<grid, block>>>(d_in, d_out, input.width, input.height, input.channels);
    cudaEventRecord(e3);
    cudaEventSynchronize(e3);

    cudaEventRecord(e4);
    CUDA_CHECK(cudaMemcpy(output.data.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(e5);
    cudaEventSynchronize(e5);

    float h2dMs = 0, kMs = 0, d2hMs = 0;
    cudaEventElapsedTime(&h2dMs, e0, e1);
    cudaEventElapsedTime(&kMs, e2, e3);
    cudaEventElapsedTime(&d2hMs, e4, e5);

    metrics.h2dTimeMs = h2dMs;
    metrics.kernelTimeMs = kMs;
    metrics.d2hTimeMs = d2hMs;
    metrics.totalTimeMs = h2dMs + kMs + d2hMs;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaEventDestroy(e4); cudaEventDestroy(e5);

    return true;
}

bool CudaKernelEngine::runResize(const RawImage& input, RawImage& output, int targetWidth, int targetHeight, GpuMetrics& metrics) {
    output.width = targetWidth;
    output.height = targetHeight;
    output.channels = input.channels;
    output.filename = input.filename;
    output.data.resize(output.sizeBytes());

    size_t inBytes = input.sizeBytes();
    size_t outBytes = output.sizeBytes();

    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;

    cudaEvent_t e0, e1, e2, e3, e4, e5;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventCreate(&e4); cudaEventCreate(&e5);

    CUDA_CHECK(cudaMalloc(&d_in, inBytes));
    CUDA_CHECK(cudaMalloc(&d_out, outBytes));

    cudaEventRecord(e0);
    CUDA_CHECK(cudaMemcpy(d_in, input.data.data(), inBytes, cudaMemcpyHostToDevice));
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((targetWidth + block.x - 1) / block.x, (targetHeight + block.y - 1) / block.y);

    cudaEventRecord(e2);
    resizeBilinearKernel<<<grid, block>>>(d_in, d_out, input.width, input.height, targetWidth, targetHeight, input.channels);
    cudaEventRecord(e3);
    cudaEventSynchronize(e3);

    cudaEventRecord(e4);
    CUDA_CHECK(cudaMemcpy(output.data.data(), d_out, outBytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(e5);
    cudaEventSynchronize(e5);

    float h2dMs = 0, kMs = 0, d2hMs = 0;
    cudaEventElapsedTime(&h2dMs, e0, e1);
    cudaEventElapsedTime(&kMs, e2, e3);
    cudaEventElapsedTime(&d2hMs, e4, e5);

    metrics.h2dTimeMs = h2dMs;
    metrics.kernelTimeMs = kMs;
    metrics.d2hTimeMs = d2hMs;
    metrics.totalTimeMs = h2dMs + kMs + d2hMs;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaEventDestroy(e4); cudaEventDestroy(e5);

    return true;
}

} // namespace cuda_image

#else
// Non-CUDA compilation stub (when compiled on machines lacking nvcc)
namespace cuda_image {

bool CudaKernelEngine::isGpuAvailable() { return false; }
bool CudaKernelEngine::init() { return false; }
void CudaKernelEngine::cleanup() {}
void CudaKernelEngine::printDeviceInfo() {
    std::cout << "[CUDA Notice] Compiled without NVCC. Native GPU execution requires CUDA Toolkit." << std::endl;
}

bool CudaKernelEngine::runGrayscale(const RawImage& input, RawImage& output, GpuMetrics& metrics) {
    std::cerr << "[CUDA] GPU execution requested but binary was built without NVCC compiler." << std::endl;
    return false;
}
bool CudaKernelEngine::runGaussianBlur(const RawImage& input, RawImage& output, GpuMetrics& metrics) {
    std::cerr << "[CUDA] GPU execution requested but binary was built without NVCC compiler." << std::endl;
    return false;
}
bool CudaKernelEngine::runSobelEdge(const RawImage& input, RawImage& output, GpuMetrics& metrics) {
    std::cerr << "[CUDA] GPU execution requested but binary was built without NVCC compiler." << std::endl;
    return false;
}
bool CudaKernelEngine::runSharpen(const RawImage& input, RawImage& output, GpuMetrics& metrics) {
    std::cerr << "[CUDA] GPU execution requested but binary was built without NVCC compiler." << std::endl;
    return false;
}
bool CudaKernelEngine::runResize(const RawImage& input, RawImage& output, int targetWidth, int targetHeight, GpuMetrics& metrics) {
    std::cerr << "[CUDA] GPU execution requested but binary was built without NVCC compiler." << std::endl;
    return false;
}

} // namespace cuda_image
#endif
