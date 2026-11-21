#include "image_processor.h"
#include "image_io.h"
#include "cuda_kernels.cuh"
#include "cpu_reference.h"
#include <iostream>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

namespace cuda_image {

ImageProcessor::ImageProcessor(const ProcessingOptions& options)
    : options_(options) {}

bool ImageProcessor::runBatch() {
    std::cout << "\n=========================================================\n";
    std::cout << "  CUDA GPU-ACCELERATED BATCH IMAGE PROCESSING SYSTEM     \n";
    std::cout << "=========================================================\n";
    std::cout << "Input Directory:   " << options_.inputDir << "\n";
    std::cout << "Output Directory:  " << options_.outputDir << "\n";
    std::cout << "Operation:         " << operationToString(options_.operation) << "\n";
    std::cout << "Compare with CPU:  " << (options_.compareCpu ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "Warmup Iterations: " << options_.warmupRuns << "\n";
    std::cout << "Benchmark Runs:    " << options_.iterations << "\n";
    std::cout << "=========================================================\n\n";

    // Initialize CUDA device
    bool gpuReady = CudaKernelEngine::init();
    if (!gpuReady) {
        std::cout << "[System] Notice: Native NVIDIA GPU not active in this host runtime.\n"
                  << "         Running in CPU Reference & Verification mode.\n"
                  << "         (Compile with NVCC on NVIDIA-equipped hardware for GPU kernels)\n\n";
    } else {
        CudaKernelEngine::printDeviceInfo();
    }

    std::vector<std::string> imageFiles = ImageIO::discoverImages(options_.inputDir);
    if (imageFiles.empty()) {
        std::cerr << "[ImageProcessor] Error: No valid image files (.bmp, .ppm, .pgm) found in: " 
                  << options_.inputDir << std::endl;
        return false;
    }

    std::cout << "[ImageProcessor] Discovered " << imageFiles.size() << " images to process.\n\n";

    std::vector<Operation> opsToRun;
    if (options_.operation == Operation::ALL) {
        opsToRun = {Operation::GRAYSCALE, Operation::GAUSSIAN_BLUR, Operation::SOBEL_EDGE, Operation::SHARPEN, Operation::RESIZE};
    } else {
        opsToRun = {options_.operation};
    }

    // Process each operation across all images
    for (Operation op : opsToRun) {
        std::string opName = operationToString(op);
        std::cout << ">>> Processing Operation: [" << opName << "] across " << imageFiles.size() << " images...\n";

        // Warmup on first image if requested
        if (options_.warmupRuns > 0 && !imageFiles.empty()) {
            RawImage warmupImg;
            if (ImageIO::loadImage(imageFiles[0], warmupImg)) {
                RawImage dummyGpu, dummyCpu;
                BenchmarkRecord dummyRec;
                for (int w = 0; w < options_.warmupRuns; ++w) {
                    executeOperation(warmupImg, op, dummyGpu, dummyCpu, dummyRec);
                }
            }
        }

        int processedCount = 0;
        for (const auto& filepath : imageFiles) {
            RawImage inputImg;
            if (!ImageIO::loadImage(filepath, inputImg)) {
                std::cerr << "  Failed to load: " << filepath << std::endl;
                continue;
            }

            RawImage outGpu, outCpu;
            BenchmarkRecord rec;
            rec.filename = inputImg.filename;
            rec.width = inputImg.width;
            rec.height = inputImg.height;
            rec.channels = inputImg.channels;
            rec.operationName = opName;

            bool success = executeOperation(inputImg, op, outGpu, outCpu, rec);
            if (!success) {
                std::cerr << "  Operation failed for: " << filepath << std::endl;
                continue;
            }

            // Save output image
            std::string stem = ImageIO::getStem(filepath);
            std::string outSubdir = options_.outputDir + "/" + opName;
            std::string outFilename = outSubdir + "/" + stem + "_" + opName + ".bmp";
            
            // If GPU produced output, save GPU output; otherwise save CPU reference output
            if (outGpu.isValid()) {
                ImageIO::saveImage(outFilename, outGpu);
            } else if (outCpu.isValid()) {
                ImageIO::saveImage(outFilename, outCpu);
            }

            reporter_.addRecord(rec);
            processedCount++;

            if (processedCount % 50 == 0 || processedCount == static_cast<int>(imageFiles.size())) {
                std::cout << "  Progress: " << processedCount << " / " << imageFiles.size() << " completed.\n";
            }
        }
        std::cout << ">>> Finished Operation: [" << opName << "]. Successfully processed " << processedCount << " images.\n\n";
    }

    reporter_.computeSummaries();
    reporter_.printSummaryToConsole();

    if (options_.exportCsv) {
        reporter_.exportCSV(options_.csvPath);
        std::cout << "[ImageProcessor] Performance CSV saved to: " << options_.csvPath << std::endl;
    }
    reporter_.exportSummaryText(options_.summaryPath);
    std::cout << "[ImageProcessor] Performance Summary saved to: " << options_.summaryPath << std::endl;

    CudaKernelEngine::cleanup();
    return true;
}

bool ImageProcessor::executeOperation(const RawImage& input, Operation op, RawImage& outGpu, RawImage& outCpu, BenchmarkRecord& rec) {
    bool gpuSuccess = false;
    GpuMetrics gpuMetrics;

    // 1. Execute on GPU if available
    if (CudaKernelEngine::isGpuAvailable()) {
        switch (op) {
            case Operation::GRAYSCALE:
                gpuSuccess = CudaKernelEngine::runGrayscale(input, outGpu, gpuMetrics);
                break;
            case Operation::GAUSSIAN_BLUR:
                gpuSuccess = CudaKernelEngine::runGaussianBlur(input, outGpu, gpuMetrics);
                break;
            case Operation::SOBEL_EDGE:
                gpuSuccess = CudaKernelEngine::runSobelEdge(input, outGpu, gpuMetrics);
                break;
            case Operation::SHARPEN:
                gpuSuccess = CudaKernelEngine::runSharpen(input, outGpu, gpuMetrics);
                break;
            case Operation::RESIZE:
                gpuSuccess = CudaKernelEngine::runResize(input, outGpu, options_.resizeWidth, options_.resizeHeight, gpuMetrics);
                break;
            default:
                break;
        }
    }

    // 2. Execute CPU Reference
    double cpuTimeMs = 0.0;
    bool cpuSuccess = false;
    if (options_.compareCpu || !gpuSuccess) {
        switch (op) {
            case Operation::GRAYSCALE:
                cpuSuccess = CpuReferenceEngine::runGrayscale(input, outCpu, cpuTimeMs);
                break;
            case Operation::GAUSSIAN_BLUR:
                cpuSuccess = CpuReferenceEngine::runGaussianBlur(input, outCpu, cpuTimeMs);
                break;
            case Operation::SOBEL_EDGE:
                cpuSuccess = CpuReferenceEngine::runSobelEdge(input, outCpu, cpuTimeMs);
                break;
            case Operation::SHARPEN:
                cpuSuccess = CpuReferenceEngine::runSharpen(input, outCpu, cpuTimeMs);
                break;
            case Operation::RESIZE:
                cpuSuccess = CpuReferenceEngine::runResize(input, outCpu, options_.resizeWidth, options_.resizeHeight, cpuTimeMs);
                break;
            default:
                break;
        }
    }

    // Populate benchmark record
    rec.cpuTimeMs = cpuTimeMs;
    rec.gpuKernelTimeMs = gpuMetrics.kernelTimeMs;
    rec.gpuH2DTimeMs = gpuMetrics.h2dTimeMs;
    rec.gpuD2HTimeMs = gpuMetrics.d2hTimeMs;
    rec.gpuEndToEndTimeMs = gpuMetrics.totalTimeMs;

    if (gpuSuccess && outGpu.isValid()) {
        rec.outputWidth = outGpu.width;
        rec.outputHeight = outGpu.height;
        rec.outputChannels = outGpu.channels;
    } else if (cpuSuccess && outCpu.isValid()) {
        rec.outputWidth = outCpu.width;
        rec.outputHeight = outCpu.height;
        rec.outputChannels = outCpu.channels;
    }

    if (gpuMetrics.kernelTimeMs > 0) {
        rec.speedupKernel = cpuTimeMs / gpuMetrics.kernelTimeMs;
    } else {
        rec.speedupKernel = 0.0;
    }

    if (gpuMetrics.totalTimeMs > 0) {
        rec.speedupEndToEnd = cpuTimeMs / gpuMetrics.totalTimeMs;
    } else {
        rec.speedupEndToEnd = 0.0;
    }

    return (gpuSuccess || cpuSuccess);
}

} // namespace cuda_image
