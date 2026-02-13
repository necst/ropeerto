/*
MIT License

Copyright (c) 2025 Giuseppe Sorrentino, Paolo Salvatore Galfano, Davide Conficconi, Eleonora D'Arnese

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once
#include "opencv2/opencv.hpp"
#include <iostream>
#include <unordered_map>

#define TRACK_READS                 
#define COMPILE_WITHOUT_DCMTK       

#ifndef COMPILE_WITHOUT_DCMTK
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmimage/diregist.h>
#endif

#define MODE_NEAREST false
#define MODE_BILINEAR true

enum ImageFormat { DICOM,PNG };

int cast_mats_to_vector (uint8_t* volume, std::vector<cv::Mat> images, const int SIZE, int N_COUPLES, const int BORDER_PADDING, const int DEPTH_PADDING);
int vector_to_mats(uint8_t* volume, std::vector<cv::Mat> images, int SIZE, int N_COUPLES,int BORDER_PADDING,int DEPTH_PADDING);
int read_volume_from_file(uint8_t *volume, const int SIZE, const int N_COUPLES, const int BORDER_PADDING, const int DEPTH_PADDING, const std::string &path, const ImageFormat imageFormat = ImageFormat::PNG);
void write_volume_to_file(uint8_t *volume, const int SIZE, const int N_COUPLES, const int BORDER_PADDING, const int DEPTH_PADDING, const std::string &path);


uint8_t track_reads(uint8_t *mem, const int index, float *ratio = NULL);
void get_cache_stats(int &hits, int &misses, int &size);
void reset_cache_stats();


void transform_volume(
    uint8_t *volume_src,
    uint8_t *volume_dest,
    const float TX,
    const float TY,
    const float ANG,
    const int SIZE,
    const int LAYERS,
    const bool bilinear_interpolation = MODE_NEAREST
);


int cache_hits = 0;
int cache_misses = 0;
int cache_size = 0;

void get_cache_stats(int &hits, int &misses, int &size) {
    hits = cache_hits;
    misses = cache_misses;
    size = cache_size;
}

void reset_cache_stats() {
    cache_hits = 0;
    cache_misses = 0;
    cache_size = 0;
}

#define CACHE_STRATEGY_SIGNATURE uint8_t *source, int Pi, int Pj, int i, int j, int k, int index, int SIZE, int LAYERS, bool can_overwrite
typedef uint8_t (*CacheStrategy)(CACHE_STRATEGY_SIGNATURE);
#define NO_CACHING NULL

uint8_t read_from_cache(CACHE_STRATEGY_SIGNATURE, CacheStrategy cache_strategy = NO_CACHING) {
    if (index == -1) return 0;
    if (k > 0 || cache_strategy == NO_CACHING) return source[index];

    return cache_strategy(source, Pi, Pj, i, j, k, index, SIZE, LAYERS, can_overwrite);
}

// macro for the function signature of a caching strategy with name X
#define CACHE_STRATEGY_DEF(X) uint8_t cache_strategy_##X(CACHE_STRATEGY_SIGNATURE)
#define CACHE_STRATEGY(X) &cache_strategy_##X


// ---------- cahing strategies ----------

// just for testing
// uint8_t cache_strategy_dummy(uint8_t *source, int i, int j, int k, int index, int SIZE, int LAYERS, bool can_overwrite) {
CACHE_STRATEGY_DEF(dummy) {
    return source[index];
}

// like that paper (https://sfat.massey.ac.nz/research/centres/crisp/pdfs/2004_DELTA_126.pdf)
CACHE_STRATEGY_DEF(GribbonBailey) {
    struct CacheElem {
        int y;
        uint8_t value;
    };
    static std::unordered_map<int, CacheElem> cache;

    bool hit = (cache.find(Pi) != cache.end() && cache[Pi].y == Pj);

    if (hit) {
        cache_hits++;
        return cache[Pi].value;
    }

    cache_misses++;
    uint8_t new_value = source[index];
    if (can_overwrite)
        cache[Pi] = {Pj, new_value};
    return new_value;
}

template <class T>
int compute_buffer_offset(
    const int SIZE, const int LAYERS, 
    const T i, const T j, const T k
) {
    #ifndef USE_OLD_FORMAT
        return std::round(j * (T)(SIZE * LAYERS) + i * (T)LAYERS + k); // new formula
    #else
        return std::round(k * (T)(SIZE*SIZE) + j * (T)SIZE + i); // old formula
    #endif
}

int transform_coords(const int SIZE, const int LAYERS, const int TX,const int TY,const float ANG, const int i, const int j, const int k);

inline uint8_t transform_nearest_neighbour(
    uint8_t *volume_src,
    const int TX, const int TY, const float ANG,
    const int SIZE, const int LAYERS,
    const int i, const int j, const int k
) {
    // compute source index (transform [i,j] coordinates)
    int old_index = transform_coords(SIZE, LAYERS, TX, TY, ANG, i, j, k);

    // read pixel from input volume
    #ifndef TRACK_READS
    uint8_t pixel = (old_index != -1 ? volume_src[old_index] : 0);
    #else
    uint8_t pixel = (old_index != -1 ? track_reads(volume_src, old_index) : 0); // count sequential reads for stas
    #endif

    return pixel;
}

// NOTE: assuming that k is always in bounds
inline bool is_out_of_bounds(const int SIZE, const int LAYERS, const float i, const float j) {
    return (i < 0 || i >= SIZE ||
            j < 0 || j >= SIZE);
}

inline uint8_t transform_bilinear(
    uint8_t *volume_src,
    const float TX, const float TY, const float ANG,
    const int SIZE, const int LAYERS,
    const int i, const int j, const int k
) {
    // compute source position (transform [i,j] coordinates)
    const float P_i = (i-SIZE/2.f - TX)*std::cos(ANG) - (j-SIZE/2.f - TY)*std::sin(ANG) + (SIZE/2.f);
    const float P_j = (i-SIZE/2.f - TX)*std::sin(ANG) + (j-SIZE/2.f - TY)*std::cos(ANG) + (SIZE/2.f);

    // computing the positions of the 4 pixels surrounding the transformed coordinates
    const float P_left = std::floor(P_i);
    const float P_right = std::ceil(P_i);
    const float P_top = std::floor(P_j);
    const float P_bottom = std::ceil(P_j);

    // compute indexes of the 4 pixels surrounding the transformed coordinates
    const int Q11_index = (!is_out_of_bounds(SIZE, LAYERS, P_left, P_top)     ? compute_buffer_offset<int>(SIZE, LAYERS, P_left,  P_top,    k) : -1); // top-left
    const int Q12_index = (!is_out_of_bounds(SIZE, LAYERS, P_right, P_top)    ? compute_buffer_offset<int>(SIZE, LAYERS, P_right, P_top,    k) : -1); // top-right
    const int Q21_index = (!is_out_of_bounds(SIZE, LAYERS, P_left, P_bottom)  ? compute_buffer_offset<int>(SIZE, LAYERS, P_left,  P_bottom, k) : -1); // bottom-left
    const int Q22_index = (!is_out_of_bounds(SIZE, LAYERS, P_right, P_bottom) ? compute_buffer_offset<int>(SIZE, LAYERS, P_right, P_bottom, k) : -1); // bottom-right

    // retrieve values of the 4 pixels (top-left, top-right, bottom-left, bottom-right)
    const float Q11_val = (float)read_from_cache(volume_src, P_left,  P_top,    i, j, k, Q11_index, SIZE, LAYERS, true, CACHE_STRATEGY(GribbonBailey));
    const float Q12_val = (float)read_from_cache(volume_src, P_right, P_top,    i, j, k, Q12_index, SIZE, LAYERS, true, CACHE_STRATEGY(GribbonBailey));
    const float Q21_val = (float)read_from_cache(volume_src, P_left,  P_bottom, i, j, k, Q21_index, SIZE, LAYERS, true, CACHE_STRATEGY(GribbonBailey));
    const float Q22_val = (float)read_from_cache(volume_src, P_right, P_bottom, i, j, k, Q22_index, SIZE, LAYERS, true, CACHE_STRATEGY(GribbonBailey));

    // projections of P_i and P_j on the x-axis and y-axis, in the box Q11-Q12-Q21-Q22
    const float R_i = P_i - P_left; // fractional part of P_i
    const float R_j = P_j - P_top;  // fractional part of P_j
    const float R_i_inv = 1.f - R_i;
    const float R_j_inv = 1.f - R_j;

    // bilinear interpolation: REPEATED LINEAR INTERPOLATION (left/right -> top/bottom)
    const float val_left = Q11_val * R_i_inv + Q12_val * R_i;
    const float val_right = Q21_val * R_i_inv + Q22_val * R_i;
    const float pixel = std::round(val_left * R_j_inv + val_right * R_j);

    return pixel;
}

void transform_volume(
    uint8_t *volume_src,
    uint8_t *volume_dest,
    const float TX,
    const float TY,
    const float ANG,
    const int SIZE,
    const int LAYERS,
    const bool bilinear_interpolation
) {
    float n_sin = -std::sin(ANG);
    float p_cos = std::cos(ANG);
    float p_sin = std::sin(ANG);
    float n_cos = -std::cos(ANG);

    float delta_row_fixed_row = p_sin;
    float delta_col_fixed_row = p_cos;

    float delta_row_fixed_col = p_cos;
    float delta_col_fixed_col = n_sin;

    float cc_first_tra = (0 - SIZE/2.f) * p_cos + (0 - SIZE/2.f) * n_sin + (SIZE/2.f) - TX - delta_col_fixed_row;
    float rr_first_tra = (0 - SIZE/2.f) * p_sin + (0 - SIZE/2.f) * p_cos + (SIZE/2.f) - TY - delta_row_fixed_row;

    float cc_tra = 0;
    float rr_tra = 0;

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < LAYERS; k++) {

                uint16_t pixel;
                if (bilinear_interpolation)
                    pixel = transform_bilinear(volume_src, TX, TY, ANG, SIZE, LAYERS, i, j, k);
                else
                    pixel = transform_nearest_neighbour(volume_src, TX, TY, ANG, SIZE, LAYERS, i, j, k);

                // compute dest index
                #ifndef USE_OLD_FORMAT
                const int dest_index = j*SIZE*LAYERS + i*LAYERS + k; // new formula
                #else
                const int dest_index = k*SIZE*SIZE + j*SIZE + i; // old formula
                #endif

                // write pixel in output volume
                volume_dest[dest_index] = pixel;
            }
        }
    }
}

// allows translation along x-y plane, and rotation around z-axis
int transform_coords(const int SIZE, const int LAYERS, const int TX,const int TY,const float ANG, const int i, const int j, const int k)
{   
    #ifndef USE_FLOAT_INDEX
    #ifndef USE_FLOOR
    const int new_i = std::round((i-SIZE/2.f)*std::cos(ANG) - (j-SIZE/2.f)*std::sin(ANG) - (float)TX + (SIZE/2.f));
    const int new_j = std::round((i-SIZE/2.f)*std::sin(ANG) + (j-SIZE/2.f)*std::cos(ANG) - (float)TY + (SIZE/2.f));
    #else
    const int new_i = std::floor((i-SIZE/2.f)*std::cos(ANG) - (j-SIZE/2.f)*std::sin(ANG) - (float)TX + (SIZE/2.f));
    const int new_j = std::floor((i-SIZE/2.f)*std::sin(ANG) + (j-SIZE/2.f)*std::cos(ANG) - (float)TY + (SIZE/2.f));
    #endif
    const int new_k = k;
    #else
    const float new_i = (i-SIZE/2.f)*std::cos(ANG) - (j-SIZE/2.f)*std::sin(ANG) - (float)TX + (SIZE/2.f);
    const float new_j = (i-SIZE/2.f)*std::sin(ANG) + (j-SIZE/2.f)*std::cos(ANG) - (float)TY + (SIZE/2.f);
    const float new_k = k;
    #endif

    // identify out-of-bound coordinates
    int out_index;
    if (new_i < 0 || new_i >= SIZE ||
        new_j < 0 || new_j >= SIZE ||
        new_k < 0 || new_k >= LAYERS)
        out_index = -1;
    else {
        #ifndef USE_FLOAT_INDEX
        out_index = compute_buffer_offset<float>(SIZE, LAYERS, new_i, new_j, new_k);
        #else
        #ifndef USE_FLOOR
        out_index = std::round(compute_buffer_offset<int>(SIZE, LAYERS, new_i, new_j, new_k));
        #else
        out_index = std::floor(compute_buffer_offset<int>(SIZE, LAYERS, new_i, new_j, new_k));
        #endif
        #endif
    }
    
    return out_index;
}

void write_slice_in_buffer(uint8_t *src, uint8_t *dest, const int slice_index, const int SIZE, const int LAYERS) {
    for (int i = 0; i < SIZE*SIZE; i++) {
        #ifndef USE_OLD_FORMAT
        const int dest_index = i * LAYERS + slice_index; // new formula
        #else
        const int dest_index = slice_index * SIZE * SIZE + i; // old formula
        #endif
        dest[dest_index] = src[i];
    }
}

void read_slice_from_buffer(uint8_t *src, uint8_t *dest, const int slice_index, const int SIZE, const int LAYERS) {
    for (int i = 0; i < SIZE*SIZE; i++) {
        #ifndef USE_OLD_FORMAT
        const int src_index = i * LAYERS + slice_index; // new formula
        #else
        const int src_index = slice_index * SIZE * SIZE + i; // old formula
        #endif
        dest[i] = src[src_index];
    }
}

/// Round up to next higher power of 2 (return x if it's already a power of 2).
inline unsigned int pow2roundup(unsigned int x)
{
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

inline uint8_t convertDepth8(uint32_t pixel, const int originalDepth) {
    const uint32_t maxPixelValue = ((int32_t)1 << originalDepth) - 1;
    const float normalized = pixel / (float)maxPixelValue;
    return std::round(255.0f * normalized);
}

int read_volume_from_file_DICOM(uint8_t *volume, const int SIZE, const int N_COUPLES, const std::string &path) {
    #ifndef COMPILE_WITHOUT_DCMTK
    uint8_t* imgData = new uint8_t[SIZE * SIZE];

    for (int i = 0; i < N_COUPLES; i++) {
        std::string s = path + "IM" + std::to_string(i+1) + ".dcm";
        DcmFileFormat fileformat;
        OFCondition status = fileformat.loadFile(s.c_str());

        if (!status.good()) {
            std::cerr << "Error: cannot load DICOM file (" << status.text() << ")" << std::endl;
            return -1;
        }
        
        DicomImage* dcmImage = new DicomImage(&fileformat, EXS_Unknown);
        if (dcmImage == nullptr || !dcmImage->isMonochrome()) {
            std::cerr << "Error: cannot read DICOM image (" << status.text() << ")" << std::endl;
            return -1;
        }

        const int bitsPerPixel = dcmImage->getDepth();
        const unsigned long numPixels = dcmImage->getWidth() * dcmImage->getHeight();
        if (numPixels != SIZE*SIZE) {
            std::cerr << "Error: size of image (" << dcmImage->getWidth() << "*" << dcmImage->getHeight() << ") different from required size (" << SIZE << "*" << SIZE << ")" << std::endl;
            return -1;
        }

        const size_t bufferSize = numPixels * (pow2roundup(bitsPerPixel) / 8);

        if (imgData == nullptr) {
            std::cerr << "Error: cannot allocate " << bufferSize << " bytes" << std::endl;
            perror("Error");
            return -1;
        }

        if (dcmImage->getOutputData(imgData, bufferSize, 8, 0) == false) {
            std::cerr << "Error: cannot read pixels (" << status.text() << ")" << std::endl;
            return -1;
        }
        
        write_slice_in_buffer((uint8_t*)imgData, volume, i, SIZE, N_COUPLES);
    }
    delete[] imgData;
    return 0;
    
    #else
    std::cerr << "Error: DICOM support is disabled" << std::endl;
    return -1;
    #endif
}


// NOTE: SIZE and N_COUPLES should reflect the shape of the dataset, and thus not include the padding.
int read_volume_from_file_PNG(uint8_t *volume, const int SIZE, const int N_COUPLES, const int BORDER_PADDING, const int DEPTH_PADDING, const std::string &path) {
    std::printf("Reading volume from file\n");
    for (int i = 0; i < N_COUPLES; i++) {
        std::string s = path + "IM" + std::to_string(i) + ".png";
        cv::Mat image = cv::imread(s, cv::IMREAD_GRAYSCALE);
        if (!image.data) {
            std::cout<<"Not Found " <<s<<std::endl;
            return -1;
        }

        // add border-padding of 1px around the image
        cv::copyMakeBorder(image, image, BORDER_PADDING, BORDER_PADDING, BORDER_PADDING, BORDER_PADDING, cv::BORDER_CONSTANT, 0);

        // copy the slice into the buffer
        std::vector<uint8_t> tmp((SIZE+2*BORDER_PADDING)*(SIZE+2*BORDER_PADDING));
        tmp.assign(image.begin<uint8_t>(), image.end<uint8_t>());
        write_slice_in_buffer(tmp.data(), volume, i, SIZE+2*BORDER_PADDING, N_COUPLES+DEPTH_PADDING);
    }

    for (int i = 0; i < DEPTH_PADDING; i++) {
        // copy the slice into the buffer
        std::vector<uint8_t> tmp((SIZE+2*BORDER_PADDING)*(SIZE+2*BORDER_PADDING));
        tmp.assign(tmp.size(), 0);
        write_slice_in_buffer(tmp.data(), volume, N_COUPLES+i, SIZE+2*BORDER_PADDING, N_COUPLES+DEPTH_PADDING);
    }

    return 0;
}


// NOTE: SIZE and N_COUPLES should reflect the shape of the dataset, and thus not include the padding.
void write_volume_to_file(uint8_t *volume, const int SIZE, const int N_COUPLES, const int BORDER_PADDING, const int DEPTH_PADDING, const std::string &path) {
    std::cout<<"Writing volume to file"<<std::endl;
    for (int i = 0; i < N_COUPLES; i++) {
        std::vector<uint8_t> tmp((SIZE+2*BORDER_PADDING)*(SIZE+2*BORDER_PADDING));
        read_slice_from_buffer(volume, tmp.data(), i, SIZE+2*BORDER_PADDING, N_COUPLES+DEPTH_PADDING);
        cv::Mat slice = (cv::Mat(SIZE+2*BORDER_PADDING, SIZE+2*BORDER_PADDING, CV_8U, tmp.data())).clone();
        slice = slice(cv::Rect(BORDER_PADDING, BORDER_PADDING, SIZE, SIZE)); // remove depth-padding
        std::string s = path + "IM" + std::to_string(i+1) + ".png";
        cv::imwrite(s, slice);
    }
}

// NOTE: SIZE and N_COUPLES should reflect the shape of the dataset, and thus not include the padding.
int read_volume_from_file(uint8_t *volume, const int SIZE, const int N_COUPLES, const int BORDER_PADDING, const int DEPTH_PADDING, const std::string &path, const ImageFormat imageFormat) {
    switch (imageFormat) {
        case ImageFormat::PNG:
            return read_volume_from_file_PNG(volume, SIZE, N_COUPLES, BORDER_PADDING, DEPTH_PADDING, path);
        case ImageFormat::DICOM:
            return read_volume_from_file_DICOM(volume, SIZE, N_COUPLES, path);
    }
    return -1;
}

// this function converts an std::vector<cv::Mat> into a 3D volume buffer with the correct pattern
int cast_mats_to_vector (uint8_t* volume, std::vector<cv::Mat> images, const int SIZE, int N_COUPLES, const int BORDER_PADDING, const int DEPTH_PADDING){
    for(int i = 0; i < N_COUPLES; i++){
        //cv::copyMakeBorder(images[i], images[i], BORDER_PADDING, BORDER_PADDING, BORDER_PADDING, BORDER_PADDING, cv::BORDER_CONSTANT, 0);
        std::vector<uint8_t> tmp((SIZE+2*BORDER_PADDING)*(SIZE+2*BORDER_PADDING));
        tmp.assign(images[i].begin<uint8_t>(), images[i].end<uint8_t>());
        write_slice_in_buffer(tmp.data(), volume, i, SIZE+2*BORDER_PADDING, N_COUPLES+DEPTH_PADDING);
    }
    for (int i = 0; i < DEPTH_PADDING; i++) {
        // copy the padding into the buffer
        std::vector<uint8_t> tmp((SIZE+2*BORDER_PADDING)*(SIZE+2*BORDER_PADDING));
        tmp.assign(tmp.size(), 0);
        write_slice_in_buffer(tmp.data(), volume, N_COUPLES+i, SIZE+2*BORDER_PADDING, N_COUPLES+DEPTH_PADDING);
        }
    return 0;
}

// Currently Unused
int vector_to_mats(uint8_t* volume, std::vector<cv::Mat> images, int SIZE, int N_COUPLES,int BORDER_PADDING,int DEPTH_PADDING)
{
    for (int i = 0; i < N_COUPLES; i++) {
        std::vector<uint8_t> tmp((SIZE+2*BORDER_PADDING)*(SIZE+2*BORDER_PADDING));
        read_slice_from_buffer(volume, tmp.data(), i, SIZE+2*BORDER_PADDING, N_COUPLES+DEPTH_PADDING);
        cv::Mat slice = (cv::Mat(SIZE+2*BORDER_PADDING, SIZE+2*BORDER_PADDING, CV_8U, tmp.data())).clone();
        slice = slice(cv::Rect(BORDER_PADDING, BORDER_PADDING, SIZE, SIZE)); // remove depth-padding
        images[i] = slice;
    }
}


uint8_t track_reads(uint8_t *mem, const int index, float *ratio) {
    static unsigned int sequential_count = 0;
    static unsigned int total_count = 0;
    static unsigned int last_index = -2;

    if (ratio == NULL) {
        ++total_count;
        if (index == last_index + 1) {
            ++sequential_count;
        }
        last_index = index;

        #ifdef DEBUG_ACCESSED_INDEXES
        std::cout << index << " ";
        #endif

        return mem[index];
    }

    if (total_count != 0)
        *ratio = 100.0f * sequential_count / (float)total_count;
    else
        *ratio = 0.0f;
    
    return -1;
}
