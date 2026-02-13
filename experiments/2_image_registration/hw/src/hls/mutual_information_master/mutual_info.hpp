/******************************************
 * MIT License
 *
 *Copyright (c) 2023 Paolo Salvatore Galfano, Giuseppe Sorrentino
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in all
 *copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *SOFTWARE.
 *
 *
 *MIT License
 *
 *Copyright (c) 2023 Giuseppe Sorrentino, Marco Venere, Eleonora D'Arnese, Davide Conficconi, Marco D. Santambrogio
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in all
 *copies or substantial portions of the Software.
 * 
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *SOFTWARE.
 *
 * 
 *MIT License
 *
 *Copyright (c) [2020] [Davide Conficconi, Eleonora D'Arnese, Emanuele Del Sozzo, Marco Domenico Santambrogio]
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in all
 *copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *SOFTWARE.
 */
 /***************************************************************
 *
 * High-Level-Synthesis header file for Mutual Information computation
 *
 ****************************************************************/
 #ifndef MUTUAL_INF_H
 #define MUTUAL_INF_H
 #include "ap_int.h"
 #include <hls_stream.h>
#include "ap_axi_sdata.h"
 
 typedef float data_t;
 #define DATA_T_BITWIDTH 32
 typedef ap_uint<8> MY_PIXEL; 
 //0
 
 #define TWO_FLOAT 2.0f
 #define OUT_BUFF_SIZE 1
 
 #define ENTROPY_THRESH 0.000000000000001
 #define ENTROPY_FLT_THRESH 0.000000000001
 #define ENTROPY_REF_THRESH 0.000000000001
 
 /************/
 
 /*********** SIM used values **********/
 #define DIMENSION 512
 //1
 /*********** End **********/
 
 #define MYROWS DIMENSION // Y
 #define MYCOLS DIMENSION
 
 /*********** SIM used values **********/
 #define MAX_RANGE (int)(MAX_FREQUENCY - 1)
 /*********** End **********/
 
 /*
  Joint Histogram computations
 */
 
 #define HIST_PE 8
 //2
 #define UNPACK_DATA_BITWIDTH 8
 //0
 #define N_COUPLES_MAX 512
 // 13
 #define UNPACK_DATA_TYPE ap_uint<UNPACK_DATA_BITWIDTH>
 
 #define INPUT_DATA_BITWIDTH (HIST_PE*UNPACK_DATA_BITWIDTH)
 #define INPUT_DATA_TYPE ap_uint<INPUT_DATA_BITWIDTH>
 
 #define NUM_INPUT_DATA (DIMENSION*DIMENSION/(HIST_PE))
 
 #define WRAPPER_HIST2(num) wrapper_joint_histogram_##num
 #define WRAPPER_HIST(num) WRAPPER_HIST2(num)
 
 #define WRAPPER_ENTROPY2(num) wrapper_entropy_##num
 #define WRAPPER_ENTROPY(num) WRAPPER_ENTROPY2(num)
 
 #define J_HISTO_ROWS 256
 //3
 #define J_HISTO_COLS J_HISTO_ROWS
 #define MIN_HIST_BITS 28
 //4
 #define MIN_HIST_BITS_NO_OVERFLOW MIN_HIST_BITS - 1
//#define MIN_J_HISTO_BITS (int)(std::ceil(std::log2(N_COUPLES_MAX * MYROWS * MYCOLS)))
 //
 #if HIST_PE == 1
 	#define MIN_HIST_PE_BITS (MIN_HIST_BITS)
 #endif
 
 #if HIST_PE == 2
 	#define MIN_HIST_PE_BITS (MIN_HIST_BITS - 1)
#endif

 #if HIST_PE == 4
	#define MIN_HIST_PE_BITS (MIN_HIST_BITS - 2)
#endif

 #if HIST_PE == 8
	#define MIN_HIST_PE_BITS (MIN_HIST_BITS - 3)
#endif

 #if HIST_PE == 16
	#define MIN_HIST_PE_BITS (MIN_HIST_BITS - 4)
#endif

 #if HIST_PE == 32
	#define MIN_HIST_PE_BITS (MIN_HIST_BITS - 5)
#endif
//MIN_HIST_PE_BITS=std::ceil(std::log2(ROW_PE_KRNL*COLS_PE_KRNL)))

 
 typedef ap_uint<MIN_HIST_BITS> MinHistBits_t;
 typedef ap_uint<MIN_HIST_PE_BITS> MinHistPEBits_t;
 
 
 #define ENTROPY_PE 8
 //6
 const unsigned int ENTROPY_PE_CONST = ENTROPY_PE;
 
 #define PACKED_HIST_PE_DATA_BITWIDTH (MIN_HIST_PE_BITS*ENTROPY_PE)
 #define PACKED_HIST_PE_DATA_TYPE ap_uint<PACKED_HIST_PE_DATA_BITWIDTH>
 
 #define PACKED_HIST_DATA_BITWIDTH (MIN_HIST_BITS*ENTROPY_PE)
 #define PACKED_HIST_DATA_TYPE ap_uint<PACKED_HIST_DATA_BITWIDTH>
 
 //#define PACKED_DATA_T_DATA_BITWIDTH (INNER_ENTROPY_TYPE_BITWIDTH*ENTROPY_PE)
 //#define PACKED_DATA_T_DATA_TYPE ap_uint<PACKED_DATA_T_DATA_BITWIDTH>
 
 #define UINT_OUT_ENTROPY_TYPE_BITWIDTH 32
 //7
 // MAX std::ceil(std::log2( log2(N_COUPLES_MAX*MYROWS*MYCOLS) * (N_COUPLES_MAX*MYROWS*MYCOLS) )) + 1
 #define UINT_OUT_ENTROPY_TYPE ap_uint<UINT_OUT_ENTROPY_TYPE_BITWIDTH>
 
 #define FIXED_BITWIDTH 42
 #define FIXED_INT_BITWIDTH UINT_OUT_ENTROPY_TYPE_BITWIDTH
 //#define FIXED ap_ufixed<42, 32>
 //8
 #ifndef FIXED
     #define ENTROPY_TYPE data_t
     #define OUT_ENTROPY_TYPE data_t
 #else
     #define ENTROPY_TYPE FIXED
     #define OUT_ENTROPY_TYPE UINT_OUT_ENTROPY_TYPE
 #endif
 
 
 #define ANOTHER_DIMENSION J_HISTO_ROWS // should be equal to j_histo_rows
 
 
 //UNIFORM QUANTIZATION
 #define INTERVAL_NUMBER 256 // L, amount of levels we want for the binning process, thus at the output
 //9
 #define MAX_FREQUENCY J_HISTO_ROWS // the maximum number of levels at the input stage//
#define MINIMUM_FREQUENCY 0
 #define INTERVAL_LENGTH ( (MAX_FREQUENCY - MINIMUM_FREQUENCY) / INTERVAL_NUMBER ) // Q = (fmax - fmin )/L
 #define INDEX_QUANTIZED(i) (i/INTERVAL_LENGTH) // Qy(i) =  f - fmin / Q
 
 /*****************/
 
  //#ifndef CACHING
     extern "C" void mutual_information_master(hls::stream<hls::axis<ap_uint<INPUT_DATA_BITWIDTH>, 0, 0, 0>> & input_img, hls::stream<hls::axis<ap_uint<INPUT_DATA_BITWIDTH>, 0, 0, 0>> & input_ref, hls::stream<hls::axis<float, 0, 0, 0>> & mutual_info, hls::stream<hls::axis<ap_uint<INPUT_DATA_BITWIDTH>, 0, 0, 0>> & n_couples, ap_uint<64> axi_ctrl );
 //#else
 //    extern "C" void mutual_information_master(INPUT_DATA_TYPE * input_img,  data_t * mutual_info, unsigned int functionality, int* status, unsigned int n_couples);
 //#endif

 
 //11 
 #define ACC_SIZE 8
 // 12
 


 
 #include "histogram.hpp"
 #include "utils.hpp"
 #include "entropy.hpp"

#endif