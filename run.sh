#!/usr/bin/env bash
# Execution script for CUDA GPU-Accelerated Batch Image Processing System
set -e

echo "=========================================================="
echo " Starting CUDA Batch Image Processing Workflow Pipeline  "
echo "=========================================================="

# 1. Generate 500 synthetic benchmark images if not already present
if [ ! -d "data/input" ] || [ $(ls -1 data/input | wc -l) -lt 100 ]; then
    echo "[Step 1/4] Generating 500 benchmark test images..."
    python scripts/generate_dataset.py --count 500 --output data/input
else
    echo "[Step 1/4] Dataset already present in data/input ($(ls -1 data/input | wc -l) images)."
fi

# 2. Build the project
echo "[Step 2/4] Building project..."
if command -v make >/dev/null 2>&1; then
    make clean
    make
else
    mkdir -p build && cd build
    cmake ..
    cmake --build .
    cd ..
fi

# 3. Execute batch processing across all operations
echo "[Step 3/4] Executing batch image processing..."
if [ -f "bin/image_processor" ]; then
    ./bin/image_processor --input data/input --output results --operation all --compare-cpu
elif [ -f "build/image_processor" ]; then
    ./build/image_processor --input data/input --output results --operation all --compare-cpu
else
    echo "Running Python test verification pipeline..."
    python tests/test_image_processing.py
fi

# 4. Generate performance plots
echo "[Step 4/4] Generating performance analysis graphs..."
python scripts/plot_results.py

echo "=========================================================="
echo " Batch processing pipeline completed successfully!      "
echo " Results saved in results/ and results/plots/           "
echo "=========================================================="
