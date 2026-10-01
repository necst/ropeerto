#ifndef PerfMetrics_HPP
#define PerfMetrics_HPP

#include <vector>
#include <string>

/**
 * @brief Utility class for collecting and storing performance metrics
 *
 * This class collects performance metrics such as latencies, GPU power, and GPU utilization.
 * It provides methods to add new metrics, retrieve all metrics of a specific kind, and calculate
 */
class PerfMetrics {    
public:
    PerfMetrics();

    void add_new(double latency, double gpu_power, double gpu_util);

    std::vector<double> get_all(std::string kind);

    double get_average(std::string kind);

private:
    std::vector<double> latencies;
    std::vector<double> gpu_powers;
    std::vector<double> gpu_utils;

    double vector_average(std::vector<double> &vec);
};

#endif //PerfMetrics_HPP