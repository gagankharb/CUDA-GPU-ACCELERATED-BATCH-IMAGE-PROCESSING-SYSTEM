@echo off
REM Windows batch execution script for CUDA Batch Image Processing System
echo ==========================================================
echo  Starting CUDA Batch Image Processing Workflow Pipeline
echo ==========================================================

REM 1. Generate 500 images if needed
echo [Step 1/4] Checking and generating benchmark dataset...
python scripts\generate_dataset.py --count 500 --output data\input

REM 2. Run automated validation and benchmarking suite
echo [Step 2/4] Executing benchmark and verification suite...
python tests\test_image_processing.py

REM 3. Generate performance graphs
echo [Step 3/4] Generating performance graphs...
python scripts\plot_results.py

REM 4. Generate PPTX presentation
echo [Step 4/4] Building PowerPoint presentation...
python scripts\generate_presentation.py

echo ==========================================================
echo  Pipeline executed successfully!
echo ==========================================================
pause
