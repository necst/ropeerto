#ifndef PerfMonitor_HPP
#define PerfMonitor_HPP

#include <vector>
#include <thread>
#include <cstdint>

/**
 * @brief Class for monitoring GPU performance
 *
 * This class uses the ROCm SMI library to monitor the GPU power and utilization.
 * It runs in a separate thread, sampling the GPU power and utilization at regular intervals.
 * The results can be accessed via avg_power() and avg_util() methods.
 */
class PerfMonitor {
public:
    PerfMonitor(uint32_t gpu_id);

    ~PerfMonitor();

    void start();

    void stop();

    double avg_power() const;

    double avg_util() const;

private:
    bool running;
    uint32_t gpu_id;
    std::thread sampling_thread;
    std::vector<std::pair<double, double>> samples; // (GPU power W, utilization %)
};

#endif //PerfMonitor_HPP