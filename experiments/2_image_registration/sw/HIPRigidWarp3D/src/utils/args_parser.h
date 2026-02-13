#pragma once

struct rigid_warp_parsed_args {
    int size, depth;
    float tx, ty, ang;
    int runs_warmup, runs;
};

/**
 * @brief Parse command line arguments returning a struct with the 
 *        parsed values (size, depth, tx, ty, ang, runs_warmup, runs)
 * 
 * @param argc
 * @param argv
 * @return parsed_args
*/
rigid_warp_parsed_args rigid_warp_parse_args(int argc, char *argv[]);


struct main_parsed_args {
    enum Task {
        IRON_MI,
        IRON_MI_3D,
        RIGID_WARP,
        NONE
    } task;
};

/**
 * @brief Parse command line arguments returning a struct with the 
 *        parsed values (task)
 * 
 * @param argc
 * @param argv
 * @return parsed_args
*/
main_parsed_args main_parse_args(int argc, char *argv[]);

/**
 * @brief Pop N elements of the argv array starting from the second one
 * 
 * @param argc reference to the number of arguments, it will be decreased by `amount`
 * @param argv array of arguments, the second element will be removed
 * @param amount number of elements to remove (default is 2)
*/
void args_pop_front(int &argc, char **argv, int amount = 2);
