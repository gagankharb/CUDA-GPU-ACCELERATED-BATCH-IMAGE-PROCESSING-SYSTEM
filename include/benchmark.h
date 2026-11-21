#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "image_types.h"
#include <string>
#include <vector>

namespace cuda_image {

struct OperationSummary {
    std::string operation;
    int imageCount = 0;
    uint64_t totalPixels = 0;
    double totalCpuTimeMs = 0.0;
    double totalGpuKernelTimeMs = 0.0;
    double totalGpuH2DTimeMs = 0.0;
    double totalGpuD2HTimeMs = 0.0;
    double totalGpuEndToEndTimeMs = 0.0;

    double avgCpuTimePerImageMs() const {
        return imageCount > 0 ? totalCpuTimeMs / imageCount : 0.0;
    }
    double avgGpuKernelTimePerImageMs() const {
        return imageCount > 0 ? totalGpuKernelTimeMs / imageCount : 0.0;
    }
    double avgGpuEndToEndTimePerImageMs() const {
        return imageCount > 0 ? totalGpuEndToEndTimeMs / imageCount : 0.0;
    }
    double kernelSpeedup() const {
        return totalGpuKernelTimeMs > 0 ? totalCpuTimeMs / totalGpuKernelTimeMs : 0.0;
    }
    double endToEndSpeedup() const {
        return totalGpuEndToEndTimeMs > 0 ? totalCpuTimeMs / totalGpuEndToEndTimeMs : 0.0;
    }
    double throughputMpixelsPerSecGpuKernel() const {
        return totalGpuKernelTimeMs > 0 ? (totalPixels / 1e6) / (totalGpuKernelTimeMs / 1e3) : 0.0;
    }
    double throughputMpixelsPerSecCpu() const {
        return totalCpuTimeMs > 0 ? (totalPixels / 1e6) / (totalCpuTimeMs / 1e3) : 0.0;
    }
};

class BenchmarkReporter {
public:
    void addRecord(const BenchmarkRecord& rec);
    void computeSummaries();

    bool exportCSV(const std::string& csvPath) const;
    bool exportSummaryText(const std::string& txtPath) const;
    void printSummaryToConsole() const;

    const std::vector<BenchmarkRecord>& getRecords() const { return records_; }
    const std::vector<OperationSummary>& getSummaries() const { return summaries_; }

private:
    std::vector<BenchmarkRecord> records_;
    std::vector<OperationSummary> summaries_;
};

} // namespace cuda_image

#endif // BENCHMARK_H
