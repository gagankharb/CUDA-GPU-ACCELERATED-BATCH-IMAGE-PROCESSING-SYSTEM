#include "image_io.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <cstring>

namespace fs = std::filesystem;

namespace cuda_image {

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t file_type{0x4D42}; // 'BM'
    uint32_t file_size{0};
    uint16_t reserved1{0};
    uint16_t reserved2{0};
    uint32_t offset_data{54}; // Standard 54-byte offset for 24-bit
};

struct BMPInfoHeader {
    uint32_t size{40};
    int32_t width{0};
    int32_t height{0};
    uint16_t planes{1};
    uint16_t bit_count{24};
    uint32_t compression{0};
    uint32_t size_image{0};
    int32_t x_pixels_per_meter{0};
    int32_t y_pixels_per_meter{0};
    uint32_t colors_used{0};
    uint32_t colors_important{0};
};
#pragma pack(pop)

bool ImageIO::loadImage(const std::string& filepath, RawImage& image) {
    std::string ext = getExtension(filepath);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".bmp") {
        return loadBMP(filepath, image);
    } else if (ext == ".ppm" || ext == ".pgm") {
        return loadPPM(filepath, image);
    }

    std::cerr << "[ImageIO] Unsupported file extension: " << ext << " for file: " << filepath << std::endl;
    return false;
}

bool ImageIO::saveImage(const std::string& filepath, const RawImage& image) {
    if (!image.isValid()) {
        std::cerr << "[ImageIO] Error: Invalid image data cannot be saved to " << filepath << std::endl;
        return false;
    }

    std::string ext = getExtension(filepath);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // Ensure directory exists
    try {
        fs::path p(filepath);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }
    } catch (...) {}

    if (ext == ".bmp") {
        return saveBMP(filepath, image);
    } else if (ext == ".ppm") {
        return savePPM(filepath, image);
    } else if (ext == ".pgm") {
        return savePGM(filepath, image);
    }

    // Default to BMP if unknown
    return saveBMP(filepath + ".bmp", image);
}

std::vector<std::string> ImageIO::discoverImages(const std::string& directoryPath) {
    std::vector<std::string> imagePaths;

    try {
        if (!fs::exists(directoryPath) || !fs::is_directory(directoryPath)) {
            std::cerr << "[ImageIO] Warning: Directory does not exist: " << directoryPath << std::endl;
            return imagePaths;
        }

        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".bmp" || ext == ".ppm" || ext == ".pgm" || ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                    imagePaths.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ImageIO] Exception during directory scan: " << e.what() << std::endl;
    }

    std::sort(imagePaths.begin(), imagePaths.end());
    return imagePaths;
}

bool ImageIO::createDirectories(const std::string& dirPath) {
    try {
        return fs::create_directories(dirPath);
    } catch (...) {
        return false;
    }
}

std::string ImageIO::getFilename(const std::string& filepath) {
    return fs::path(filepath).filename().string();
}

std::string ImageIO::getStem(const std::string& filepath) {
    return fs::path(filepath).stem().string();
}

std::string ImageIO::getExtension(const std::string& filepath) {
    return fs::path(filepath).extension().string();
}

bool ImageIO::loadBMP(const std::string& filepath, RawImage& image) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "[ImageIO] Cannot open BMP file: " << filepath << std::endl;
        return false;
    }

    BMPHeader header;
    BMPInfoHeader infoHeader;

    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (header.file_type != 0x4D42) {
        std::cerr << "[ImageIO] Not a valid BMP file (magic mismatch): " << filepath << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));
    if (infoHeader.bit_count != 24 && infoHeader.bit_count != 8) {
        std::cerr << "[ImageIO] BMP loading only supports 24-bit RGB or 8-bit Grayscale. Bit depth: " 
                  << infoHeader.bit_count << std::endl;
        return false;
    }

    image.width = infoHeader.width;
    image.height = std::abs(infoHeader.height);
    image.channels = (infoHeader.bit_count == 24) ? 3 : 1;
    image.filename = getFilename(filepath);

    bool topDown = (infoHeader.height < 0);

    // If 8-bit, skip palette (1024 bytes)
    if (infoHeader.bit_count == 8) {
        file.seekg(header.offset_data, std::ios::beg);
    } else {
        file.seekg(header.offset_data, std::ios::beg);
    }

    image.data.resize(image.sizeBytes());

    // BMP rows are padded to 4-byte boundaries
    int rowPadding = (4 - (image.width * (infoHeader.bit_count / 8)) % 4) % 4;
    std::vector<uint8_t> rowBuffer(image.width * (infoHeader.bit_count / 8));

    for (int y = 0; y < image.height; ++y) {
        int targetY = topDown ? y : (image.height - 1 - y);
        file.read(reinterpret_cast<char*>(rowBuffer.data()), rowBuffer.size());
        if (rowPadding > 0) {
            file.ignore(rowPadding);
        }

        uint8_t* destRow = &image.data[targetY * image.width * image.channels];
        if (image.channels == 3) {
            // BMP stores BGR; convert to RGB
            for (int x = 0; x < image.width; ++x) {
                destRow[x * 3 + 0] = rowBuffer[x * 3 + 2]; // R
                destRow[x * 3 + 1] = rowBuffer[x * 3 + 1]; // G
                destRow[x * 3 + 2] = rowBuffer[x * 3 + 0]; // B
            }
        } else {
            std::memcpy(destRow, rowBuffer.data(), image.width);
        }
    }

    return true;
}

bool ImageIO::saveBMP(const std::string& filepath, const RawImage& image) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "[ImageIO] Cannot write to BMP file: " << filepath << std::endl;
        return false;
    }

    BMPHeader header;
    BMPInfoHeader infoHeader;

    int bytesPerPixel = (image.channels == 1) ? 1 : 3;
    int rowPadding = (4 - (image.width * bytesPerPixel) % 4) % 4;
    uint32_t rowSizeWithPadding = image.width * bytesPerPixel + rowPadding;
    uint32_t paletteSize = (image.channels == 1) ? 1024 : 0;

    header.offset_data = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + paletteSize;
    header.file_size = header.offset_data + rowSizeWithPadding * image.height;

    infoHeader.width = image.width;
    infoHeader.height = image.height; // Positive = bottom-up
    infoHeader.planes = 1;
    infoHeader.bit_count = (image.channels == 1) ? 8 : 24;
    infoHeader.size_image = rowSizeWithPadding * image.height;
    infoHeader.colors_used = (image.channels == 1) ? 256 : 0;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));

    // Write grayscale palette if 8-bit
    if (image.channels == 1) {
        for (int i = 0; i < 256; ++i) {
            uint8_t entry[4] = {
                static_cast<uint8_t>(i),
                static_cast<uint8_t>(i),
                static_cast<uint8_t>(i),
                0
            };
            file.write(reinterpret_cast<const char*>(entry), 4);
        }
    }

    uint8_t paddingBytes[3] = {0, 0, 0};
    std::vector<uint8_t> rowBuffer(image.width * bytesPerPixel);

    // BMP standard bottom-up encoding
    for (int y = image.height - 1; y >= 0; --y) {
        const uint8_t* srcRow = &image.data[y * image.width * image.channels];
        if (image.channels == 3) {
            for (int x = 0; x < image.width; ++x) {
                rowBuffer[x * 3 + 0] = srcRow[x * 3 + 2]; // B
                rowBuffer[x * 3 + 1] = srcRow[x * 3 + 1]; // G
                rowBuffer[x * 3 + 2] = srcRow[x * 3 + 0]; // R
            }
            file.write(reinterpret_cast<const char*>(rowBuffer.data()), rowBuffer.size());
        } else {
            file.write(reinterpret_cast<const char*>(srcRow), image.width);
        }

        if (rowPadding > 0) {
            file.write(reinterpret_cast<const char*>(paddingBytes), rowPadding);
        }
    }

    return true;
}

bool ImageIO::loadPPM(const std::string& filepath, RawImage& image) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return false;

    std::string magic;
    file >> magic;
    if (magic != "P6" && magic != "P5") {
        std::cerr << "[ImageIO] Unsupported Netpbm format (must be P6 or P5): " << magic << std::endl;
        return false;
    }

    int width = 0, height = 0, maxVal = 0;
    // Skip comments
    while (file >> std::ws && file.peek() == '#') {
        std::string comment;
        std::getline(file, comment);
    }

    file >> width >> height >> maxVal;
    file.get(); // Consume newline after maxVal

    if (maxVal != 255 || width <= 0 || height <= 0) {
        std::cerr << "[ImageIO] Invalid PPM/PGM header values" << std::endl;
        return false;
    }

    image.width = width;
    image.height = height;
    image.channels = (magic == "P6") ? 3 : 1;
    image.filename = getFilename(filepath);
    image.data.resize(image.sizeBytes());

    file.read(reinterpret_cast<char*>(image.data.data()), image.data.size());
    return true;
}

bool ImageIO::savePPM(const std::string& filepath, const RawImage& image) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;

    file << "P6\n" << image.width << " " << image.height << "\n255\n";
    if (image.channels == 3) {
        file.write(reinterpret_cast<const char*>(image.data.data()), image.data.size());
    } else if (image.channels == 1) {
        // Expand grayscale to RGB for PPM
        std::vector<uint8_t> rgb(image.width * image.height * 3);
        for (size_t i = 0; i < image.data.size(); ++i) {
            rgb[i * 3 + 0] = image.data[i];
            rgb[i * 3 + 1] = image.data[i];
            rgb[i * 3 + 2] = image.data[i];
        }
        file.write(reinterpret_cast<const char*>(rgb.data()), rgb.size());
    }
    return true;
}

bool ImageIO::savePGM(const std::string& filepath, const RawImage& image) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;

    file << "P5\n" << image.width << " " << image.height << "\n255\n";
    if (image.channels == 1) {
        file.write(reinterpret_cast<const char*>(image.data.data()), image.data.size());
    } else if (image.channels == 3) {
        // Convert to grayscale on the fly
        std::vector<uint8_t> gray(image.width * image.height);
        for (int i = 0; i < image.width * image.height; ++i) {
            float y = 0.299f * image.data[i * 3 + 0] + 0.587f * image.data[i * 3 + 1] + 0.114f * image.data[i * 3 + 2];
            gray[i] = static_cast<uint8_t>(y + 0.5f);
        }
        file.write(reinterpret_cast<const char*>(gray.data()), gray.size());
    }
    return true;
}

} // namespace cuda_image
