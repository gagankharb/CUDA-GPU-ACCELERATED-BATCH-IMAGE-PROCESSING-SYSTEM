#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include "image_types.h"
#include <string>
#include <vector>

namespace cuda_image {

class ImageIO {
public:
    // Load image from file (supports BMP, PPM, PGM)
    static bool loadImage(const std::string& filepath, RawImage& image);

    // Save image to file (determines format from file extension: .bmp, .ppm, .pgm)
    static bool saveImage(const std::string& filepath, const RawImage& image);

    // Discover all supported images in a directory
    static std::vector<std::string> discoverImages(const std::string& directoryPath);

    // Helper: ensure parent directories exist
    static bool createDirectories(const std::string& dirPath);

    // Helper: extract file stem and extension
    static std::string getFilename(const std::string& filepath);
    static std::string getStem(const std::string& filepath);
    static std::string getExtension(const std::string& filepath);

private:
    static bool loadBMP(const std::string& filepath, RawImage& image);
    static bool saveBMP(const std::string& filepath, const RawImage& image);

    static bool loadPPM(const std::string& filepath, RawImage& image);
    static bool savePPM(const std::string& filepath, const RawImage& image);
    static bool savePGM(const std::string& filepath, const RawImage& image);
};

} // namespace cuda_image

#endif // IMAGE_IO_H
