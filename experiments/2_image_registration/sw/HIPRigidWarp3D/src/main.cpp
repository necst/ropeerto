#include <iostream>
#include <args_parser.h>
#include "tests.h"

int main(int argc, char** argv) {
    // Parse degli argomenti
    main_parsed_args args = main_parse_args(3, argv);
    args_pop_front(argc, argv);
    
    
    
    if (args.task == main_parsed_args::Task::RIGID_WARP) {
        // Chiamata alla trasformata HIP
        test_rigid_warp_hip(argc, argv);
    } else {
        std::cerr << "Invalid task" << std::endl;
    }

    return 0;
}
