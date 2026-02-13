/*
MIT License

Copyright (c) 2023 Paolo Salvatore Galfano, Giuseppe Sorrentino

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/***************************************************************
*
* Configuration header file for Versal 3D Image Registration
*
****************************************************************/
#ifndef CONSTANTS_H
#define CONSTANTS_H

typedef float data_t;
#define DATA_T_BITWIDTH 32

#define DIMENSION 512

#define HIST_PE 8
#define HIST_PE_EXPO 3
#define UNPACK_DATA_BITWIDTH 8
#define INPUT_DATA_BITWIDTH (HIST_PE*UNPACK_DATA_BITWIDTH)
// #define INPUT_DATA_BITWIDTH_INTERP 8 // TODO remove (use INPUT_DATA_BITWIDTH_FETCHER instead)
#define INPUT_DATA_BITWIDTH_FETCHER 256
#define INPUT_DATA_BITWIDTH_FETCHER_MIN 256
#define NUM_PIXELS_PER_READ 32
#define NUM_PIXELS_PER_READ_EXPO 5
#define NUM_INPUT_DATA (DIMENSION*DIMENSION/(HIST_PE))
#define N_COUPLES_MAX 512
#define J_HISTO_ROWS 256
#define J_HISTO_COLS J_HISTO_ROWS
#define ANOTHER_DIMENSION J_HISTO_ROWS // should be equal to j_histo_rows
#define HISTO_ROWS J_HISTO_ROWS
#define INTERVAL_NUMBER 256 // L, amount of levels we want for the binning process, thus at the output

#define SIZE_ROWS 512 // how many rows per aie tile
#define SIZE_COLS 512 // how many columns per aie tile

#define INIT_COLS {-256, -255, -254, -253, -252, -251, -250, -249, -248, -247, -246, -245, -244, -243, -242, -241, -240, -239, -238, -237, -236, -235, -234, -233, -232, -231, -230, -229, -228, -227, -226, -225} // the initial columns for the aie tiles

#define ENTROPY_PE 8
#define INT_PE 1
#define INT_PE_EXPO 0
#define DIV_EXPO 0

#define AIE_PATTERNS {{1UL, 0UL}}
#define AIE_PATTERN_OFFSETS {{0, 1}}

#define COORD_BITWIDTH 32

#define AIE_TOP_LEFT 0
#define AIE_TOP_RIGHT 1
#define AIE_BOTTOM_LEFT 2
#define AIE_BOTTOM_RIGHT 3

#endif
