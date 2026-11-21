#include "benchmark.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <filesystem>

namespace fs = std::filesystem;

namespace cuda_image {

void BenchmarkReporter::addRecord(const BenchmarkRecord& rec) {
    records_.push_back(rec);
}

void BenchmarkReporter::computeSummaries() {
    summaries_.clear();
    std::map<std::string, OperationSummary> summaryMap;

    for (const auto& r : records_) {
        auto& s = summaryMap[r.operationName];
        s.operation = r.operationName;
        s.imageCount += 1;
        s.totalPixels += static_cast<uint64_t>(r.width) * r.height;
        s.totalCpuTimeMs += r.cpuTimeMs;
        s.totalGpuKernelTimeMs += r.gpuKernelTimeMs;
        s.totalGpuH2DTimeMs += r.gpuH2DTimeMs;
        s.totalGpuD2HTimeMs += r.gpuD2HTimeMs;
        s.totalGpuEndToEndTimeMs += r.gpuEndToEndTimeMs;
    }

    // Preserve standard ordering
    std::vector<std::string> ops = {"grayscale", "blur", "sobel", "sharpen", "resize"};
    for (const auto& op : ops) {
        if (summaryMap.find(op) != summaryMap.end()) {
            summaries_.push_back(summaryMap[op]);
        }
    }
    // Any remaining
    for (const auto& pair : summaryMap) {
        if (std::find(ops.begin(), ops.end(), pair.first) == ops.end()) {
            summaries_.push_back(pair.second);
        }
    }
}

bool BenchmarkReporter::exportCSV(const std::string& csvPath) const {
    try {
        fs::path p(csvPath);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
    } catch (...) {}

    std::ofstream f(csvPath);
    if (!f) {
        std::cerr << "[Benchmark] Failed to open CSV file for writing: " << csvPath << std::endl;
        return false;
    }

    f << "Operation,Filename,Width,Height,Channels,OutputWidth,OutputHeight,OutputChannels,"
      << "CpuTimeMs,GpuKernelTimeMs,GpuH2DTimeMs,GpuD2HTimeMs,GpuEndToEndTimeMs,SpeedupKernel,SpeedupEndToEnd\n";

    for (const auto& r : records_) {
        f << r.operationName << ","
          << r.filename << ","
          << r.width << ","
          << r.height << ","
          << r.channels << ","
          << r.outputWidth << ","
          << r.outputHeight << ","
          << r.outputChannels << ","
          << std::fixed << std::setprecision(4)
          << r.cpuTimeMs << ","
          << r.gpuKernelTimeMs << ","
          << r.gpuH2DTimeMs << ","
          << r.gpuD2HTimeMs << ","
          << r.gpuEndToEndTimeMs << ","
          << std::setprecision(2)
          << r.speedupKernel << ","
          << r.speedupEndToEnd << "\n";
    }

    return true;
}

bool BenchmarkReporter::exportSummaryText(const std::string& txtPath) const {
    try {
        fs::path p(txtPath);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
    } catch (...) {}

    std::ofstream f(txtPath);
    if (!f) {
        std::cerr << "[Benchmark] Failed to open summary file for writing: " << txtPath << std::endl;
        return false;
    }

    f << "===================================================================================================\n";
    f << "              CUDA GPU-ACCELERATED BATCH IMAGE PROCESSING SYSTEM - PERFORMANCE SUMMARY              \n";
    f << "===================================================================================================\n\n";

    f << std::left << std::setw(12) << "Operation"
      << std::right << std::setw(8) << "Images"
      << std::setw(14) << "Total Pixels"
      << std::setw(14) << "CPU Time (s)"
      << std::setw(14) << "GPU K-Time(s)"
      << std::setw(14) << "GPU E2E(s)"
      << std::setw(12) << "K-Speedup"
      << std::setw(12) << "E2E-Speedup"
      << std::setw(16) << "GPU Throughput\n";
    f << "---------------------------------------------------------------------------------------------------\n";

    for (const auto& s : summaries_) {
        f << std::left << std::setw(12) << s.operation
          << std::right << std::setw(8) << s.imageCount
          << std::setw(14) << s.totalPixels
          << std::fixed << std::setprecision(3)
          << std::setw(14) << (s.totalCpuTimeMs / 1000.0)
          << std::setw(14) << (s.totalGpuKernelTimeMs / 1000.0)
          << std::setw(14) << (s.totalGpuEndToEndTimeMs / 1000.0)
          << std::setprecision(2)
          << std::setw(11) << s.kernelSpeedup() << "x"
          << std::setw(11) << s.endToEndSpeedup() << "x"
          << std::fixed << std::setprecision(1)
          << std::setw(12) << s.throughputMpixelsPerSecGpuKernel() << " MP/s\n";
    }

    f << "===================================================================================================\n";
    f << "DETAILED BREAKDOWN BY OPERATION:\n";
    f << "---------------------------------------------------------------------------------------------------\n";

    for (const auto& s : summaries_) {
        f << "Operation: " << s.operation << "\n";
        f << "  Images Processed:           " << s.imageCount << "\n";
        f << "  Total Megapixels:           " << std::fixed << std::setprecision(2) << (s.totalPixels / 1e6) << " MP\n";
        f << "  Average CPU Time / Image:   " << std::fixed << std::setprecision(4) << s.avgCpuTimePerImageMs() << " ms\n";
        f << "  Average GPU Kernel / Image: " << s.avgGpuKernelTimePerImageMs() << " ms\n";
        f << "  Average GPU E2E / Image:    " << s.avgGpuEndToEndTimePerImageMs() << " ms\n";
        f << "  Kernel-only Speedup:        " << std::setprecision(2) << s.kernelSpeedup() << "x\n";
        f << "  End-to-end Speedup:         " << s.endToEndSpeedup() << "x\n";
        f << "  CPU Throughput:             " << std::setprecision(1) << s.throughputMpixelsPerSecCpu() << " MP/s\n";
        f << "  GPU Kernel Throughput:      " << s.throughputMpixelsPerSecGpuKernel() << " MP/s\n\n";
    }

    return true;
}

void BenchmarkReporter::printSummaryToConsole() const {
    std::cout << "\n===================================================================================================\n";
    std::cout << "                                PERFORMANCE BENCHMARK SUMMARY                                      \n";
    std::cout << "===================================================================================================\n";

    std::cout << std::left << std::setw(12) << "Operation"
              << std::right << std::setw(8) << "Images"
              << std::setw(14) << "Total Pixels"
              << std::setw(14) << "CPU Time (s)"
              << std::setw(14) << "GPU K-Time(s)"
              << std::setw(14) << "GPU E2E(s)"
              << std::setw(12) << "K-Speedup"
              << std::setw(12) << "E2E-Speedup"
              << std::setw(16) << "GPU Throughput\n";
    std::cout << "---------------------------------------------------------------------------------------------------\n";

    for (const auto& s : summaries_) {
        std::cout << std::left << std::setw(12) << s.operation
                  << std::right << std::setw(8) << s.imageCount
                  << std::setw(14) << s.totalPixels
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << (s.totalCpuTimeMs / 1000.0)
                  << std::setw(14) << (s.totalGpuKernelTimeMs / 1000.0)
                  << std::setw(14) << (s.totalGpuEndToEndTimeMs / 1000.0)
                  << std::setprecision(2)
                  << std::setw(11) << s.kernelSpeedup() << "x"
                  << std::setw(11) << s.endToEndSpeedup() << "x"
                  << std::fixed << std::setprecision(1)
                  << std::setw(12) << s.throughputMpixelsPerSecGpuKernel() << " MP/s\n";
    }
    std::cout << "===================================================================================================\n\n";
}

} // namespace cuda_image
