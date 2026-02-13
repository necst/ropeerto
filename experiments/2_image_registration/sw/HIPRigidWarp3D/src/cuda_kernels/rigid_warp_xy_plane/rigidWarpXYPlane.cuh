# pragma once

class RigidWarpXYPlane {
    int _device_id;
    dim3 _blockSize;
    dim3 _gridSize;

    uint8_t *device_input = nullptr;
    uint8_t *device_output = nullptr;

    int _size;
    int _depth;

public:
    RigidWarpXYPlane(const int device_id = 0);

    // function to move to the input volume to the gpu
    void transferToGPU(const uint8_t* input, const int size, const int depth);
    void transferFromGPU(uint8_t* output);

    void setupGrid(const dim3 blockSize, const dim3 gridSize);
    void setupGrid(const int threads_per_block = 1024);

    double run(
        const float tx, const float ty, const float ang
    );

    ~RigidWarpXYPlane();
};

