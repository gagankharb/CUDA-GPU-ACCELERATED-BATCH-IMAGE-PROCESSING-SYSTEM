#include "image_processor.h"
#include <iostream>
#include <string>

void printHelp(const char* progName) {
    std::cout << "CUDA GPU-Accelerated Batch Image Processing System\n";
    std::cout << "Coursera: CUDA at Scale for the Enterprise - Independent Project\n\n";
    std::cout << "Usage: " << progName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --input <path>          Path to directory containing input images (default: data/input)\n";
    std::cout << "  --output <path>         Path to directory for output results (default: results)\n";
    std::cout << "  --operation <op>        Processing operation to execute: [grayscale|blur|sobel|sharpen|resize|all] (default: all)\n";
    std::cout << "  --batch-size <N>        Batch size for streaming/processing (default: 32)\n";
    std::cout << "  --compare-cpu           Enable CPU reference benchmark and verification (default: on)\n";
    std::cout << "  --no-compare-cpu        Disable CPU reference benchmark (GPU only)\n";
    std::cout << "  --warmup <N>            Number of warmup iterations before timing (default: 1)\n";
    std::cout << "  --iterations <N>        Number of benchmark timing iterations per image (default: 1)\n";
    std::cout << "  --target-width <N>      Target width for resize operation (default: 256)\n";
    std::cout << "  --target-height <N>     Target height for resize operation (default: 256)\n";
    std::cout << "  --help, -h              Display this help message and exit\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << progName << " --input data/input --output results/grayscale --operation grayscale\n";
    std::cout << "  " << progName << " --input data/input --output results/edges --operation sobel\n";
    std::cout << "  " << progName << " --input data/input --output results/blur --operation blur\n";
    std::cout << "  " << progName << " --input data/input --output results/sharpen --operation sharpen\n";
    std::cout << "  " << progName << " --input data/input --output results/resized --operation resize\n";
    std::cout << "  " << progName << " --input data/input --output results/all --operation all --compare-cpu\n";
}

int main(int argc, char* argv[]) {
    cuda_image::ProcessingOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printHelp(argv[0]);
            return 0;
        } else if (arg == "--input" && i + 1 < argc) {
            options.inputDir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            options.outputDir = argv[++i];
        } else if (arg == "--operation" && i + 1 < argc) {
            options.operation = cuda_image::stringToOperation(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            options.batchSize = std::stoi(argv[++i]);
        } else if (arg == "--compare-cpu") {
            options.compareCpu = true;
        } else if (arg == "--no-compare-cpu") {
            options.compareCpu = false;
        } else if (arg == "--warmup" && i + 1 < argc) {
            options.warmupRuns = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            options.iterations = std::stoi(argv[++i]);
        } else if (arg == "--target-width" && i + 1 < argc) {
            options.resizeWidth = std::stoi(argv[++i]);
        } else if (arg == "--target-height" && i + 1 < argc) {
            options.resizeHeight = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            std::cerr << "Run with --help to see available options.\n";
            return 1;
        }
    }

    cuda_image::ImageProcessor processor(options);
    bool success = processor.runBatch();

    return success ? 0 : 1;
}
