#include "images_io.h"

#include <iostream>

#include <filesystem>
namespace fs = std::filesystem;

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_writer.h>


void generate_example_image(uint8_t* volume, const int size, const int depth) {
    // draw a gradient in the image, varying in the z direction
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            for (int k = 0; k < depth; k++)
            {
                volume[i * size * depth + j * depth + k] = 255 * (k / (float)depth);
            }
        }
    }

    // draw cross in the middle of the image of max intensity
    for (int i = size / 4; i < 3 * size / 4; i++)
    {
        for (int j = size / 4; j < 3 * size / 4; j++)
        {
            for (int k = 0; k < depth; k++)
            {
                volume[i * size * depth + j * depth + k] = (i > size / 2 ? 200 : 255);
            }
        }
    }
}


void read_volume_from_folder(uint8_t* volume, const int size, const int depth, const std::string& folder_name) {
    uint8_t *slice = new uint8_t[size * size];

    for (int k = 0; k < depth; k++)
    {
        std::string filename = folder_name + "/IM" + std::to_string(k) + ".png";
        int w, h, n;
        uint8_t *data = stbi_load(filename.c_str(), &w, &h, &n, 1);
        if (data == nullptr)
        {
            std::cerr << "Error loading image: " << filename << std::endl;
            exit(1);
        }

        if (w != size || h != size || n != 1)
        {
            std::cerr << "Image dimensions are not correct: " << filename << std::endl;
            exit(1);
        }

        // copy slice
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                volume[i * size * depth + j * depth + k] = data[i * size + j];
            }
        }

        stbi_image_free(data);
    }

    delete[] slice;
}


void save_volume_into_folder(const uint8_t* volume, const int size, const int depth, const std::string& folder_name) {
    fs::remove_all(folder_name);
    std::filesystem::create_directories(folder_name);
    
    uint8_t *slice = new uint8_t[size * size];

    for (int k = 0; k < depth; k++)
    {
        std::string filename = folder_name + "/IM" + std::to_string(k) + ".png";
        // extract slice
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                slice[i * size + j] = volume[i * size * depth + j * depth + k];
            }
        }
        stbi_write_png(filename.c_str(), size, size, 1, slice, size);
    }

    delete[] slice;
}
