#pragma once

// #include <iostream>
// #include <cmath>
// #include <random>
// #include <chrono>
// #include <stdio.h>

using namespace std;

/*********** NUMBER OF GPU THREADS **********/
#define GPU_THREADS 256 // MIN SHOULD BE 256 TO MAINTAIN CORRECTNESS

/*********** SIM used values **********/
#define DIMENSION 512

#define MYROWS DIMENSION
#define MYCOLS DIMENSION
typedef unsigned char MY_PIXEL;

#define J_HISTO_ROWS 256
#define J_HISTO_COLS J_HISTO_ROWS
#define ANOTHER_DIMENSION J_HISTO_ROWS // should be equal to j_histo_rows
/*********** SIM used values **********/
#define MAX_RANGE (int)(ANOTHER_DIMENSION - 1)

typedef float data_t;

void mutual_information_3D_master(MY_PIXEL *ref, MY_PIXEL *flt, MY_PIXEL *ref_d, MY_PIXEL *flt_d, data_t *j_h_d,
							   data_t *href_d, data_t *hflt_d, data_t *result_d, data_t *result, size_t n_images, double *kernel_only_time);

using Mat = MY_PIXEL*;
