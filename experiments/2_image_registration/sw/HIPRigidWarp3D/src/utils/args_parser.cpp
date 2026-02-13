#include "args_parser.h"

#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <tclap/CmdLine.h>


rigid_warp_parsed_args rigid_warp_parse_args(int argc, char *argv[]) {
    int size, depth, runs_warmup, runs;
    float tx, ty, ang;

    try {
        TCLAP::CmdLine cmd("Test the rigid warp cuda kernel", ' ', "1.0");
        TCLAP::ValueArg<int> sizeArg("s", "size", "Size of the volume", false, 512, "int");
        TCLAP::ValueArg<int> depthArg("d", "depth", "Depth of the volume", false, 256, "int");
        TCLAP::ValueArg<float> txArg("x", "tx", "Translation in x", false, .0f, "float");
        TCLAP::ValueArg<float> tyArg("y", "ty", "Translation in y", false, .0f, "float");
        TCLAP::ValueArg<float> angArg("a", "ang", "Rotation angle (degrees)", false, .0f, "float");
        TCLAP::ValueArg<int> runsWarmupArg("w", "warmup", "Number of warmup runs", false, 2, "int");
        TCLAP::ValueArg<int> runsArg("r", "runs", "Number of runs", false, 10, "int");

        cmd.add(sizeArg);
        cmd.add(depthArg);
        cmd.add(txArg);
        cmd.add(tyArg);
        cmd.add(angArg);
        cmd.add(runsWarmupArg);
        cmd.add(runsArg);

        cmd.parse(argc, argv);        

        size = sizeArg.getValue();
        depth = depthArg.getValue();
        tx = txArg.getValue();
        ty = tyArg.getValue();
        ang = angArg.getValue() * M_PI / 180.0f;
        runs_warmup = runsWarmupArg.getValue();
        runs = runsArg.getValue();

    } catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        return rigid_warp_parsed_args{0, 0, 0.0f, 0.0f, 0.0f, 0, 0};
    }

    return rigid_warp_parsed_args{size, depth, tx, ty, ang, runs_warmup, runs};
}

main_parsed_args main_parse_args(int argc, char *argv[])
{
    main_parsed_args::Task task = main_parsed_args::Task::NONE;

    try {
        TCLAP::CmdLine cmd("Test the selected cuda kernel", ' ', "1.0");
        TCLAP::ValueArg<std::string> taskArg("t", "task", "Task to be executed", true, "", "string");
        cmd.add(taskArg);

        cmd.parse(argc, argv);

        if (taskArg.getValue() == "IRON") {
            task = main_parsed_args::Task::IRON_MI;
        } else if (taskArg.getValue() == "IRON3D") {
            task = main_parsed_args::Task::IRON_MI_3D;
        } else if (taskArg.getValue() == "WARP") {
            task = main_parsed_args::Task::RIGID_WARP;
        } else {
            std::cerr << "Invalid task: " << taskArg.getValue() << std::endl;
        }

    } catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

    return main_parsed_args{task};
}

void args_pop_front(int &argc, char **argv, int amount) {
    for (int i = 0; i < amount; i++) {
        for (int j = 1; j < argc - 1; j++) {
            argv[j] = argv[j + 1];
        }
        argc--;
    }
}
