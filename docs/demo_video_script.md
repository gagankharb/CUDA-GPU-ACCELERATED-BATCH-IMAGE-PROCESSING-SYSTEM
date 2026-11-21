# Video Demonstration Script & Recording Guide
## CUDA GPU-Accelerated Batch Image Processing System
**Course:** Coursera: CUDA at Scale for the Enterprise — Independent Project  
**Target Duration:** 5:00 – 7:00 Minutes  
**Format:** Screen Recording with Voiceover (1080p, 60fps recommended)

---

## 1. Video Structure & Suggested Timestamps

| Timestamp | Section | Visual Content on Screen | Key Talking Point |
|---|---|---|---|
| **0:00 – 0:35** | **Introduction & Title** | Slide 1 (Title Slide) | Course name, project title, student intro, problem summary. |
| **0:35 – 1:15** | **Architecture & Objectives** | Slides 2, 3, 4 | Computational bottleneck of CPU convolutions; 5 core objectives; end-to-end dataflow. |
| **1:15 – 1:50** | **Dataset & Kernel Implementation** | Slides 5, 7, 8, 9, 10 | 500-image dataset (131 MP); custom CUDA kernels; 2D shared memory tiling & constant memory. |
| **1:50 – 3:30** | **Live Terminal Demonstration** | Terminal Window / IDE | Inspect GPU, compile with make/cmake, run batch execution on 500 images, display summary table. |
| **3:30 – 4:20** | **Visual Output Verification** | Image Viewer / Slide 12 | Show original image alongside Grayscale, Gaussian Blur, Sobel Edge, Sharpen, and Resized images. |
| **4:20 – 5:30** | **Performance & Speedup Analysis** | Slides 11, 13 (Plots) | Walk through CPU vs GPU Kernel vs E2E bar charts; explain 47x speedup and PCIe transfer overhead. |
| **5:30 – 6:15** | **Lessons Learned & Conclusion** | Slides 14, 15 | Shared memory impact; future multi-stream/multi-GPU extensions; project wrap-up. |

---

## 2. Exact Terminal Commands to Execute During Demo

Open your terminal or IDE in the root project directory `cuda-batch-image-processing`:

### Command 1: Inspect GPU Environment
```bash
# Check NVIDIA GPU hardware and driver version
nvidia-smi

# Check CUDA compiler version
nvcc --version
```
*Speaking Point:*  
> *"First, let's verify our GPU computing environment using `nvidia-smi` and `nvcc --version`. Here we see our NVIDIA GPU with compute capability and active CUDA runtime."*

---

### Command 2: Clean and Compile the Project
```bash
# Clean previous build artifacts and compile
make clean
make
```
*(On Windows or with CMake:)*
```cmd
cmake -B build
cmake --build build --config Release
```
*Speaking Point:*  
> *"Next, we clean and build our project using `make`. The compiler invokes `nvcc` to compile our custom CUDA kernels targeting NVIDIA SM architectures, generating the optimized executable `bin/image_processor`."*

---

### Command 3: Verify the 500-Image Benchmark Dataset
```bash
# Check image file count in the input directory
ls -1 data/input | wc -l
```
*(Or in Windows PowerShell:)*
```powershell
(Get-ChildItem data/input).Count
```
*Speaking Point:*  
> *"Here we verify that our input directory contains exactly 500 high-resolution 512x512 images, representing over 131 Megapixels of raw image data."*

---

### Command 4: Execute the Batch Image Processing Pipeline
```bash
# Run batch processing across all 5 operations with CPU comparison
./bin/image_processor --input data/input --output results --operation all --compare-cpu
```
*(Or run the automated validation suite:)*
```bash
python tests/test_image_processing.py
```
*Speaking Point:*  
> *"Now we launch the batch processing pipeline across all 500 images for all five operations: Grayscale, Gaussian Blur, Sobel Edge Detection, Image Sharpening, and Bilinear Resizing. Notice the progress indicators as each operation completes. In under two minutes, over 2,500 operations and 130 Megapixels have been processed, verified, and saved."*

---

### Command 5: Display the Performance Benchmark Summary Table
```bash
cat results/performance_summary.txt
```
*(Or in Windows PowerShell:)*
```powershell
Get-Content results/performance_summary.txt
```
*Speaking Point:*  
> *"The application outputs this detailed performance table. For Gaussian Blur, CPU compute took 45.8 seconds, while our custom CUDA kernel executed in just 0.969 seconds—delivering an outstanding 47.3x kernel speedup. Even with Host-to-Device and Device-to-Host PCIe transfers included, the end-to-end speedup is 43.9x."*

---

### Command 6: Inspect Visual Outputs
```bash
# View sample output images
ls -l results/sample_outputs/*
```
*Open `results/plots/visual_comparison_montage.png` in an image viewer.*  
*Speaking Point:*  
> *"Opening our visual comparison montage, we can confirm the algorithmic correctness of each operation: clean luminance gradients in Grayscale, noise-free smoothing in Gaussian Blur, high-contrast gradient contours in Sobel, crisp contrast in Sharpening, and alias-free downscaling in Resizing."*

---

## 3. Screen Recording Walkthrough & Setup Tips

1. **Resolution & Layout:** Record your screen at 1920x1080. Arrange your desktop with PowerPoint on one half or virtual desktop, and a dark terminal on the other.
2. **Terminal Theme:** Use a clean dark background with green/cyan prompt syntax for high legibility. Set font size to 14pt–16pt so viewers can clearly read kernel launches and summary tables.
3. **Cursor Highlighting:** Enable mouse cursor highlighting so viewers can follow along when pointing to chart bars and table columns.
4. **Audio:** Use a clean external microphone or headset. Speak at a measured, confident pace.

---

## 4. Explaining CPU vs. GPU Performance (Key Video Talking Points)

When presenting the performance graphs on Slide 11:
1. **Explain the Three Metrics:**
   - **CPU Time (Red):** Baseline multi-core/single-core CPU execution.
   - **GPU Kernel Only (Green):** Raw execution time inside the CUDA Streaming Multiprocessors.
   - **GPU End-to-End (Blue):** Total elapsed time including PCIe Host-to-Device memory copy, kernel execution, and Device-to-Host transfer.
2. **Explain Why Stencils Achieve Highest Speedup:**
   - Gaussian Blur and Sobel Edge Detection perform 25 and 18 operations per pixel. By caching data in **2D Shared Memory Tiles**, threads eliminate redundant memory traffic, resulting in **42x to 47x speedups**.
3. **Explain the PCIe Overhead for Lightweight Kernels:**
   - For Grayscale and Resizing, the kernel executes so quickly (<0.1s) that PCIe memory transfer accounts for a significant portion of total time. Explain that in enterprise production, this is solved via **CUDA Streams and Pinned Memory (cudaHostAlloc)** to pipeline compute and transfer concurrently.

---

## 5. Wrap-Up & Submission Checklist

- [x] PowerPoint presentation open (`presentation/CUDA_Image_Processing_Presentation.pptx`).
- [x] Terminal prepared in project root with build tools active.
- [x] Input dataset generated (500 images in `data/input`).
- [x] Results and plots generated in `results/plots/`.
- [x] Video recording software (OBS Studio / Zoom / PowerPoint Record) configured for 1080p.
