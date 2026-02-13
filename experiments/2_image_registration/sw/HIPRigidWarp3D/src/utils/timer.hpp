#pragma once

#include <iostream>
#include <ctime>
#include <chrono>

class Timer {
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    
    TimePoint start_time;
    bool is_running;

public:
    Timer();
    
    inline void start() {
        is_running = true;
        start_time = Clock::now();
    }
    
    /**
     * @return elapsed time in seconds
    */
    inline double stop() {
        auto now = Clock::now();

        if (!is_running) {
            return 0.0;
        }
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time);
        return duration.count() / 1000000.0;
    }
};
