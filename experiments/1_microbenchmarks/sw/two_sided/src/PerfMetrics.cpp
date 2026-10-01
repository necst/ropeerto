#include "PerfMetrics.hpp"
#include <vector>
#include <string>
#include <stdexcept>


PerfMetrics::PerfMetrics() {}

void PerfMetrics::add_new(double latency, double gpu_power, double gpu_util) {
    latencies.push_back(latency);
    gpu_powers.push_back(gpu_power);
    gpu_utils.push_back(gpu_util);
}

std::vector<double> PerfMetrics::get_all(std::string kind) {
    if (kind == "latency") {
        return latencies;
    } else if (kind == "gpu_power") {
        return gpu_powers;
    } else if (kind == "gpu_util") {
        return gpu_utils;
    } else {
        throw std::runtime_error("Unknown kind of performance metric requested: " + kind);
    }
}

double PerfMetrics::get_average(std::string kind) {
    if (kind == "latency") {
        return vector_average(latencies);
    } else if (kind == "gpu_power") {
        return vector_average(gpu_powers);
    } else if (kind == "gpu_util") {
        return vector_average(gpu_utils);
    } else {
        throw std::runtime_error("Unknown kind of performance metric requested: " + kind);
    }
}



double PerfMetrics::vector_average(std::vector<double> &vec) {
    double sum = 0.0;
    for (const auto &val : vec) {
        sum += val;
    }
    return vec.empty() ? 0.0 : sum / vec.size();
}
