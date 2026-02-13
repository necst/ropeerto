#include "timer.hpp"

Timer::Timer() : is_running(false) {
    start_time = Clock::now();
}
