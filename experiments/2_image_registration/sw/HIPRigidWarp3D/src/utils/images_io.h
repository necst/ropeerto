#pragma once

#include <string>

/**
 * @brief Generate an example image with a gradient in the z direction and a square in the middle of the image.
 * 
 * @param volume pointer to the volume to be filled with the example image
 * @param size size of the volume in the x and y dimensions
 * @param depth size of the volume in the z dimension
*/
void generate_example_image(uint8_t* volume, const int size, const int depth);

/**
 * @brief Read a volume from a folder containing a sequence of images.
 * 
 * @param volume pointer to the volume to be filled with the images
 * @param size size of the volume in the x and y dimensions
 * @param depth size of the volume in the z dimension
 * @param folder_name name of the folder containing the images (can be nested)
*/
void read_volume_from_folder(uint8_t* volume, const int size, const int depth, const std::string& folder_name);

/**
 * @brief Save a volume into a folder containing a sequence of images.
 * 
 * @param volume pointer to the volume to be saved
 * @param size size of the volume in the x and y dimensions
 * @param depth size of the volume in the z dimension
 * @param folder_name name of the folder where the images will be saved (will be created if it does not exist; can be nested)
*/
void save_volume_into_folder(const uint8_t* volume, const int size, const int depth, const std::string& folder_name);


/**
 * @brief Generate an example image with a gradient in the z direction and a square in the middle of the image.
 * 
 * @param volume pointer to the volume to be filled with the example image
 * @param size size of the volume in the x and y dimensions
 * @param depth size of the volume in the z dimension
*/
void generate_example_image(uint8_t* volume, const int size, const int depth);

/**
 * @brief Read a volume from a folder containing a sequence of images.
 * 
 * @param volume pointer to the volume to be filled with the images
 * @param size size of the volume in the x and y dimensions
 * @param depth size of the volume in the z dimension
 * @param folder_name name of the folder containing the images (can be nested)
*/
void read_volume_from_folder(uint8_t* volume, const int size, const int depth, const std::string& folder_name);

/**
 * @brief Save a volume into a folder containing a sequence of images.
 * 
 * @param volume pointer to the volume to be saved
 * @param size size of the volume in the x and y dimensions
 * @param depth size of the volume in the z dimension
 * @param folder_name name of the folder where the images will be saved (will be created if it does not exist; can be nested)
*/
void save_volume_into_folder(const uint8_t* volume, const int size, const int depth, const std::string& folder_name);
