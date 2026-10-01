#include "PerfMonitor.hpp"
#include <thread>
#include <chrono>
#include <rocm_smi/rocm_smi.h>

PerfMonitor::PerfMonitor(uint32_t gpu_id) : gpu_id(gpu_id), running(false) {
    rsmi_init(0);
}

PerfMonitor::~PerfMonitor() {
    stop();
    rsmi_shut_down();
}

void PerfMonitor::start() {
    running = true;
    samples.clear();
    sampling_thread = std::thread([&]() {
        while (running) {
            uint64_t power;
            RSMI_POWER_TYPE power_type = RSMI_CURRENT_POWER;
            rsmi_dev_power_get(gpu_id, &power, &power_type);    // GPU power, Microwatts

            uint32_t utilization;
            rsmi_dev_busy_percent_get(gpu_id, &utilization);    // GPU utilization in %

            samples.push_back({(double) power / 1e6, (double) utilization});

            // Sleep for 100 microseconds before the next sample
            // Hence, this sampling only makes sense for larger transfers
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });
}

void PerfMonitor::stop() {
    running = false;
    if (sampling_thread.joinable()) sampling_thread.join();
}

double PerfMonitor::avg_power() const {
    double sum = 0; 
    for (auto &s : samples) {
        sum += s.first;
    }
    return samples.empty() ? 0.0 : sum / samples.size();
}

double PerfMonitor::avg_util() const {
    double sum = 0; 
    for (auto &s : samples) {
        sum += s.second;
    }
    return samples.empty() ? 0.0 : sum / samples.size();
}
