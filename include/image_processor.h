#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include "image_types.h"
#include "benchmark.h"
#include <string>
#include <vector>

namespace cuda_image {

struct ProcessingOptions {
    std::string inputDir = "data/input";
    std::string outputDir = "results";
    Operation operation = Operation::ALL;
    int batchSize = 32;
    bool compareCpu = true;
    int warmupRuns = 1;
    int iterations = 1;
    int resizeWidth = 256;
    int resizeHeight = 256;
    bool exportCsv = true;
    std::string csvPath = "results/performance_results.csv";
    std::string summaryPath = "results/performance_summary.txt";
};

class ImageProcessor {
public:
    explicit ImageProcessor(const ProcessingOptions& options);

    // Run batch processing across all discovered images
    bool runBatch();

    const BenchmarkReporter& getReporter() const { return reporter_; }

private:
    ProcessingOptions options_;
    BenchmarkReporter reporter_;

    bool processSingleImage(const std::string& filepath, Operation op);
    bool executeOperation(const RawImage& input, Operation op, RawImage& outGpu, RawImage& outCpu, BenchmarkRecord& rec);
};

} // namespace cuda_image

#endif // IMAGE_PROCESSOR_H
