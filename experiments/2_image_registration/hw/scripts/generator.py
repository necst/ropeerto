# MIT License
#
# Copyright (c) 2023 Paolo Salvatore Galfano, Giuseppe Sorrentino
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import os
import numpy
import math
###################################################################################################################################


def coord_idx_generator(i, id, chunks, pe):
    return math.floor((pe * i + id) / chunks)


def generate_pattern(NUM_INT_PE, aie_idx):
    all_loops = []
    for j in range(1, NUM_INT_PE + 1):
        loops = [0 for i in range(0, NUM_INT_PE)]
        for i in range(0, NUM_INT_PE):
            q = coord_idx_generator(i, aie_idx, j, NUM_INT_PE)
            if q < NUM_INT_PE:
                loops[q] = 1
        all_loops.append(loops)
    return all_loops


def find_shift(sequence1, sequence2):
    # Ensure sequences have the same length
    assert len(sequence1) == len(sequence2), "Sequences must have the same length"

    sequence1 = sequence1 * 3

    for shift in range(0, len(sequence2)):
        shifted = sequence1[shift:len(sequence2) + shift]
        if shifted == sequence2:
            return shift
    return None


def convert_to_MSWLSW(bitarray):
    LSW = 0
    MSW = 0

    r = 128 - len(bitarray)
    assert r >= 0, "too many bits"

    bitarray = bitarray + [0] * r

    for i, b in enumerate(bitarray):
        if i < 64:
            LSW += 2**i if b else 0
        else:
            MSW += 2**(i - 64) if b else 0

    return f"{{{LSW}ULL, {MSW}ULL}}"


def generate_patterns_and_offsets(NUM_INT_PE):
    all_patterns = [generate_pattern(NUM_INT_PE, i) for i in range(0, NUM_INT_PE)]  # [AIE][N_class][seq]
    offsets = []

    if NUM_INT_PE == 1:
        return '[[1UL, 0UL]]', [[0, 1]], all_patterns[0]

    for N_class in range(0, NUM_INT_PE):
        base_shift = 0
        shift = 0
        equal_count = 1
        for aie_idx in range(1, NUM_INT_PE):
            shift = find_shift(all_patterns[0][N_class], all_patterns[aie_idx][N_class])
            if shift == base_shift:
                equal_count += 1
            else:
                break
        offsets.append([shift, equal_count])

    converted_patterns = '[' + ', '.join([convert_to_MSWLSW(p) for p in all_patterns[0]]) + ']'

    return converted_patterns, offsets, all_patterns[0]


def print_mi_config(num_pe, inp_img_bits, inp_img_dim, derived, pe_entropy, fixed, caching, uram, vitis, out_path, pixels_per_read, interpolator_pe_number):
    if fixed:
        fixerd_or_not = ""
    else:
        fixerd_or_not = "//"

    # extern "C" only when generating the Vitis version
    if vitis:
        vitis_externC = "\"C\""
    else:
        vitis_externC = ""

    # Generate the function prototype block:
    # - Vitis mode: AXIS streams + axi_ctrl, caching block commented out (matches gold header)
    # - Non-vitis mode: legacy prototype with CACHING switch
    if vitis:
        mi_decl = (
            " //#ifndef CACHING\n"
            "     extern \"C\" void mutual_information_master("
            "hls::stream<hls::axis<ap_uint<INPUT_DATA_BITWIDTH>, 0, 0, 0>> & input_img, "
            "hls::stream<hls::axis<ap_uint<INPUT_DATA_BITWIDTH>, 0, 0, 0>> & input_ref, "
            "hls::stream<hls::axis<float, 0, 0, 0>> & mutual_info, "
            "hls::stream<hls::axis<ap_uint<INPUT_DATA_BITWIDTH>, 0, 0, 0>> & n_couples, "
            "ap_uint<64> axi_ctrl );\n"
            " //#else\n"
            " //    extern \"C\" void mutual_information_master(INPUT_DATA_TYPE * input_img,  data_t * mutual_info, unsigned int functionality, int* status, unsigned int n_couples);\n"
            " //#endif\n"
        )
    else:
        mi_decl = (
            "#ifndef CACHING\n"
            f"    extern {vitis_externC} void mutual_information_master(hls::stream<INPUT_DATA_TYPE> &stream_input_img, INPUT_DATA_TYPE * input_ref, data_t * mutual_info, unsigned int n_couples, unsigned int padding);\n"
            "#else\n"
            f"    extern {vitis_externC} void mutual_information_master(INPUT_DATA_TYPE * input_img,  data_t * mutual_info, unsigned int functionality, int* status, unsigned int n_couples);\n"
            "#endif\n"
        )

    mi_header = open(os.path.join(out_path, "mutual_info.hpp"), "w+")
    mi_header.write("/******************************************\n \
* MIT License\n \
*\n \
*Copyright (c) 2023 Paolo Salvatore Galfano, Giuseppe Sorrentino\n \
*\n \
*Permission is hereby granted, free of charge, to any person obtaining a copy\n \
*of this software and associated documentation files (the \"Software\"), to deal\n \
*in the Software without restriction, including without limitation the rights\n \
*to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n \
*copies of the Software, and to permit persons to whom the Software is\n \
*furnished to do so, subject to the following conditions:\n \
*\n \
*The above copyright notice and this permission notice shall be included in all\n \
*copies or substantial portions of the Software.\n \
*\n \
*THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n \
*IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n \
*FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n \
*AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n \
*LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n \
*OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n \
*SOFTWARE.\n \
*\n \
*\n \
*MIT License\n \
*\n \
*Copyright (c) 2023 Giuseppe Sorrentino, Marco Venere, Eleonora D'Arnese, Davide Conficconi, Marco D. Santambrogio\n \
*\n \
*Permission is hereby granted, free of charge, to any person obtaining a copy\n \
*of this software and associated documentation files (the \"Software\"), to deal\n \
*in the Software without restriction, including without limitation the rights\n \
*to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n \
*copies of the Software, and to permit persons to whom the Software is\n \
*furnished to do so, subject to the following conditions:\n \
*\n \
*The above copyright notice and this permission notice shall be included in all\n \
*copies or substantial portions of the Software.\n \
* \n \
*THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n \
*IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n \
*FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n \
*AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n \
*LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n \
*OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n \
*SOFTWARE.\n \
*\n \
* \n \
*MIT License\n \
*\n \
*Copyright (c) [2020] [Davide Conficconi, Eleonora D'Arnese, Emanuele Del Sozzo, Marco Domenico Santambrogio]\n \
*\n \
*Permission is hereby granted, free of charge, to any person obtaining a copy\n \
*of this software and associated documentation files (the \"Software\"), to deal\n \
*in the Software without restriction, including without limitation the rights\n \
*to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n \
*copies of the Software, and to permit persons to whom the Software is\n \
*furnished to do so, subject to the following conditions:\n \
*\n \
*The above copyright notice and this permission notice shall be included in all\n \
*copies or substantial portions of the Software.\n \
*\n \
*THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n \
*IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n \
*FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n \
*AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n \
*LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n \
*OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n \
*SOFTWARE.\n \
*/\n \
/***************************************************************\n \
*\n \
* High-Level-Synthesis header file for Mutual Information computation\n \
*\n \
****************************************************************/\n \
#ifndef MUTUAL_INF_H\n \
#define MUTUAL_INF_H\n \
#include \"ap_int.h\"\n \
#include <hls_stream.h>\n\
#include \"ap_axi_sdata.h\"\n \
\n \
typedef float data_t;\n \
#define DATA_T_BITWIDTH 32\n \
typedef ap_uint<{0}> MY_PIXEL; \n \
//0\n \
\n \
#define TWO_FLOAT 2.0f\n \
#define OUT_BUFF_SIZE 1\n \
\n \
#define ENTROPY_THRESH 0.000000000000001\n \
#define ENTROPY_FLT_THRESH 0.000000000001\n \
#define ENTROPY_REF_THRESH 0.000000000001\n \
\n \
/************/\n \
\n \
/*********** SIM used values **********/\n \
#define DIMENSION {1}\n \
//1\n \
/*********** End **********/\n \
\n \
#define MYROWS DIMENSION // Y\n \
#define MYCOLS DIMENSION\n \
\n \
/*********** SIM used values **********/\n \
#define MAX_RANGE (int)(MAX_FREQUENCY - 1)\n \
/*********** End **********/\n \
\n \
/*\n \
 Joint Histogram computations\n \
*/\n \
\n \
#define HIST_PE {2}\n \
//2\n \
#define UNPACK_DATA_BITWIDTH {0}\n \
//0\n \
#define N_COUPLES_MAX {13}\n \
// 13\n \
#define UNPACK_DATA_TYPE ap_uint<UNPACK_DATA_BITWIDTH>\n \
\n \
#define INPUT_DATA_BITWIDTH (HIST_PE*UNPACK_DATA_BITWIDTH)\n \
#define INPUT_DATA_TYPE ap_uint<INPUT_DATA_BITWIDTH>\n \
\n \
#define NUM_INPUT_DATA (DIMENSION*DIMENSION/(HIST_PE))\n \
\n \
#define WRAPPER_HIST2(num) wrapper_joint_histogram_##num\n \
#define WRAPPER_HIST(num) WRAPPER_HIST2(num)\n \
\n \
#define WRAPPER_ENTROPY2(num) wrapper_entropy_##num\n \
#define WRAPPER_ENTROPY(num) WRAPPER_ENTROPY2(num)\n \
\n \
#define J_HISTO_ROWS {3}\n \
//3\n \
#define J_HISTO_COLS J_HISTO_ROWS\n \
#define MIN_HIST_BITS {4}\n \
//4\n \
#define MIN_HIST_BITS_NO_OVERFLOW MIN_HIST_BITS - 1\n\
//#define MIN_J_HISTO_BITS (int)(std::ceil(std::log2(N_COUPLES_MAX * MYROWS * MYCOLS)))\n \
//\n \
#if HIST_PE == 1\n \
	#define MIN_HIST_PE_BITS (MIN_HIST_BITS)\n \
#endif\n \
\n \
#if HIST_PE == 2\n \
	#define MIN_HIST_PE_BITS (MIN_HIST_BITS - 1)\n\
#endif\n\
\n \
#if HIST_PE == 4\n\
	#define MIN_HIST_PE_BITS (MIN_HIST_BITS - 2)\n\
#endif\n\
\n \
#if HIST_PE == 8\n\
	#define MIN_HIST_PE_BITS (MIN_HIST_BITS - 3)\n\
#endif\n\
\n \
#if HIST_PE == 16\n\
	#define MIN_HIST_PE_BITS (MIN_HIST_BITS - 4)\n\
#endif\n\
\n \
#if HIST_PE == 32\n\
	#define MIN_HIST_PE_BITS (MIN_HIST_BITS - 5)\n\
#endif\n\
//MIN_HIST_PE_BITS=std::ceil(std::log2(ROW_PE_KRNL*COLS_PE_KRNL)))\n\
\n \
\n \
typedef ap_uint<MIN_HIST_BITS> MinHistBits_t;\n \
typedef ap_uint<MIN_HIST_PE_BITS> MinHistPEBits_t;\n \
\n \
\n \
#define ENTROPY_PE {6}\n \
//6\n \
const unsigned int ENTROPY_PE_CONST = ENTROPY_PE;\n \
\n \
#define PACKED_HIST_PE_DATA_BITWIDTH (MIN_HIST_PE_BITS*ENTROPY_PE)\n \
#define PACKED_HIST_PE_DATA_TYPE ap_uint<PACKED_HIST_PE_DATA_BITWIDTH>\n \
\n \
#define PACKED_HIST_DATA_BITWIDTH (MIN_HIST_BITS*ENTROPY_PE)\n \
#define PACKED_HIST_DATA_TYPE ap_uint<PACKED_HIST_DATA_BITWIDTH>\n \
\n \
//#define PACKED_DATA_T_DATA_BITWIDTH (INNER_ENTROPY_TYPE_BITWIDTH*ENTROPY_PE)\n \
//#define PACKED_DATA_T_DATA_TYPE ap_uint<PACKED_DATA_T_DATA_BITWIDTH>\n \
\n \
#define UINT_OUT_ENTROPY_TYPE_BITWIDTH {7}\n \
//7\n \
// MAX std::ceil(std::log2( log2(N_COUPLES_MAX*MYROWS*MYCOLS) * (N_COUPLES_MAX*MYROWS*MYCOLS) )) + 1\n \
#define UINT_OUT_ENTROPY_TYPE ap_uint<UINT_OUT_ENTROPY_TYPE_BITWIDTH>\n \
\n \
#define FIXED_BITWIDTH 42\n \
#define FIXED_INT_BITWIDTH UINT_OUT_ENTROPY_TYPE_BITWIDTH\n \
{8}#define FIXED ap_ufixed<42, {7}>\n \
//8\n \
#ifndef FIXED\n \
    #define ENTROPY_TYPE data_t\n \
    #define OUT_ENTROPY_TYPE data_t\n \
#else\n \
    #define ENTROPY_TYPE FIXED\n \
    #define OUT_ENTROPY_TYPE UINT_OUT_ENTROPY_TYPE\n \
#endif\n \
\n \
\n \
#define ANOTHER_DIMENSION J_HISTO_ROWS // should be equal to j_histo_rows\n \
\n \
\n \
//UNIFORM QUANTIZATION\n \
#define INTERVAL_NUMBER {9} // L, amount of levels we want for the binning process, thus at the output\n \
//9\n \
#define MAX_FREQUENCY J_HISTO_ROWS // the maximum number of levels at the input stage\
//\n\
#define MINIMUM_FREQUENCY 0\n \
#define INTERVAL_LENGTH ( (MAX_FREQUENCY - MINIMUM_FREQUENCY) / INTERVAL_NUMBER ) // Q = (fmax - fmin )/L\n \
#define INDEX_QUANTIZED(i) (i/INTERVAL_LENGTH) // Qy(i) =  f - fmin / Q\n \
\n \
/*****************/\n \
\n \
{14}\n \
\n \
//11 \n \
#define ACC_SIZE {12}\n \
// 12\n \
\n".format(inp_img_bits, \
        inp_img_dim, \
        num_pe, \
        derived.hist_dim, \
        derived.histos_bits, \
        derived.pe_bits, \
        pe_entropy, \
        derived.uint_fixed_bitwidth, \
        fixerd_or_not, \
        derived.quant_levels, derived.maximum_freq, \
        vitis_externC, \
        derived.entr_acc_size, \
        derived.n_couples_max, \
        mi_decl))
    if caching:
        mi_header.write(" \n \
#define CACHING\n\
//14")
    if uram:
        mi_header.write(" \n \
#define URAM\n \
//15")
    mi_header.write("\n\n \n #include \"histogram.hpp\"\n #include \"utils.hpp\"\n #include \"entropy.hpp\"\n\n#endif")

    x = -int(inp_img_dim / 2)
    y = x + 32
    init_cols = [str(i) for i in range(x, y)]
    init_cols_str = ", ".join(init_cols)
    init_cols_str = "{" + init_cols_str + "}"

    input_db_interp = 8  # TODO parametrizzare
    num_pixels_per_read = pixels_per_read  # how many pixels at once are passed to the setup_interpolator
    input_db_fetcher = 8 * num_pixels_per_read
    input_db_fetcher_min = 8 * 32

    aie_patterns, aie_offsets, _ = generate_patterns_and_offsets(interpolator_pe_number)

    aie_patterns = str(aie_patterns).replace("[", "{").replace("]", "}")
    aie_offsets = str(aie_offsets).replace("[", "{").replace("]", "}")

    constants_header = open(os.path.join(out_path, "constants.h"), "w+")
    constants_header.write(
        f'''/*
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

#define DIMENSION {inp_img_dim}

#define HIST_PE {num_pe}
#define HIST_PE_EXPO {int(math.log2(num_pe))}
#define UNPACK_DATA_BITWIDTH {inp_img_bits}
#define INPUT_DATA_BITWIDTH (HIST_PE*UNPACK_DATA_BITWIDTH)
// #define INPUT_DATA_BITWIDTH_INTERP {input_db_interp} // TODO remove (use INPUT_DATA_BITWIDTH_FETCHER instead)
#define INPUT_DATA_BITWIDTH_FETCHER {input_db_fetcher}
#define INPUT_DATA_BITWIDTH_FETCHER_MIN {input_db_fetcher_min}
#define NUM_PIXELS_PER_READ {num_pixels_per_read}
#define NUM_PIXELS_PER_READ_EXPO {int(math.log2(num_pixels_per_read))}
#define NUM_INPUT_DATA (DIMENSION*DIMENSION/(HIST_PE))
#define N_COUPLES_MAX {derived.n_couples_max}
#define J_HISTO_ROWS {derived.hist_dim}
#define J_HISTO_COLS J_HISTO_ROWS
#define ANOTHER_DIMENSION J_HISTO_ROWS // should be equal to j_histo_rows
#define HISTO_ROWS J_HISTO_ROWS
#define INTERVAL_NUMBER {derived.quant_levels} // L, amount of levels we want for the binning process, thus at the output

#define SIZE_ROWS {inp_img_dim} // how many rows per aie tile
#define SIZE_COLS {inp_img_dim} // how many columns per aie tile

#define INIT_COLS {init_cols_str} // the initial columns for the aie tiles

#define ENTROPY_PE {pe_entropy}
#define INT_PE {interpolator_pe_number}
#define INT_PE_EXPO {int(math.log2(interpolator_pe_number))}
#define DIV_EXPO {int(math.log2(math.ceil(interpolator_pe_number * 32 / num_pixels_per_read)))}

#define AIE_PATTERNS {aie_patterns}
#define AIE_PATTERN_OFFSETS {aie_offsets}

#define COORD_BITWIDTH 32

#define AIE_TOP_LEFT 0
#define AIE_TOP_RIGHT 1
#define AIE_BOTTOM_LEFT 2
#define AIE_BOTTOM_RIGHT 3

#endif
''')
    constants_header.close()


######################################################
######################################################
######################################################
######################################################


class ParametersDerived:
    def __init__(self):
        self.histos_bits = 0
        self.quant_levels = 0
        self.reduced_lvls = 0
        self.hist_dim = 0
        self.j_idx_bits = 0
        self.idx_bits = 0
        self.reduced_histos_bits = 0
        self.maximum_freq = 0
        self.in_dim = 0
        self.in_bits = 0
        self.bin_val = 0
        self.pe_number = 0
        self.entr_acc_size = 0
        self.bit_entropy = 0
        self.pe_bits = 0
        self.uint_fixed_bitwidth = 0
        # cc and mse
        self.sumbitwidth = 0
        self.tmp_sumbitwidth = 0
        self.dim_inverse = 0
        self.n_couples_max = 0

    def derive_bitwidth(self, data_container):
        return 32

    def derive(self, in_dim, in_bits, bin_val, pe_number, entr_acc_size, histotype, n_couples_max):
        self.in_dim = in_dim
        self.in_bits = in_bits
        self.bin_val = bin_val
        self.pe_number = pe_number
        self.entr_acc_size = entr_acc_size
        self.histos_bits = math.ceil(numpy.log2(n_couples_max * in_dim * in_dim)) + 1
        self.reduced_lvls = math.ceil(in_bits - bin_val)
        self.quant_levels = math.ceil(2**self.reduced_lvls)
        self.hist_dim = math.ceil(2**self.reduced_lvls)
        self.j_idx_bits = math.ceil(numpy.log2(self.hist_dim * self.hist_dim))
        self.idx_bits = math.ceil(numpy.log2(self.hist_dim))
        self.reduced_histos_bits = math.ceil(numpy.log2(n_couples_max * in_dim * in_dim / pe_number)) + 1
        self.maximum_freq = math.ceil(2**in_bits)
        self.bit_entropy = self.derive_bitwidth(histotype)
        self.pe_bits = math.ceil(numpy.log2(pe_number))
        self.uint_fixed_bitwidth = math.ceil(math.log2(math.log2(n_couples_max * in_dim * in_dim) * n_couples_max * in_dim * in_dim))
        self.sumbitwidth = math.ceil(in_bits * 2 + numpy.log2(in_dim) * 2)
        self.tmp_sumbitwidth = math.ceil(self.sumbitwidth - numpy.log2(pe_number))
        self.n_couples_max = n_couples_max

    def getScaleFactor(self):
        return self.scale_factor

    def printDerived(self):
        print("Starting params:\n in_dim {0}\n in_bits {1}\n bin_val {2}\n pe_number {3}\n entr_acc_size {4}\n"\
            .format(self.in_dim,\
            self.in_bits, self.bin_val, self.pe_number, self.entr_acc_size))
        print("Derived Configuration: \nhisto bits " + str(self.histos_bits))
        print("quant_levels " + str(self.quant_levels))
        print("hist_dim " + str(self.hist_dim))
        print("j_idx_bits " + str(self.j_idx_bits))
        print("idx_bits " + str(self.idx_bits))
        print("reduced_histos_bits " + str(self.reduced_histos_bits))
        print("maximum_freq " + str(self.maximum_freq))
        print("entropies bitwdith " + str(self.bit_entropy))
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################


def main():
    parser = argparse.ArgumentParser(description='Configuration Generation for the MI and histogram accelerator,\n\
        Default configuration is to use \'float\' datatypes with a 512x512 8 bits input matrix and a histogram of 256 levels with one PE')
    parser.add_argument("-op", "--out_path", nargs='?', help='output path for the out files, default ./', default='./')
    parser.add_argument("-c", "--clean", help='clean previously created files befor starting', action='store_true')

    parser.add_argument("-ht", "--histotype", nargs='?', help='data type for the floating point computation, default float', default='float')
    parser.add_argument("-pe", "--pe_number", nargs='?', help='number of PEs assigned to the joint histogram computation, default 1', default='1', type=int)
    parser.add_argument("-ib", "--in_bits", nargs='?', help='number of bits for the target input image, default 8', default='8', type=int)
    parser.add_argument("-id", "--in_dim", nargs='?', help='maximum dimension of the target input image, default 512', default='512', type=int)
    parser.add_argument("-bv", "--bin_val", nargs='?', help='reduction factor of in the binning process of the histogram \n\
                        (i.e. a factor of 2 means 2** (in bits - bin_val) bin levels) , default 0', default='0', type=int)
    parser.add_argument("-es", "--entr_acc_size", nargs='?', help=' accumulator size for the entropies computation', default='8', type=int)
    parser.add_argument("-pen", "--pe_entropy", nargs='?', help='number of PEs assigned to the entropy computation, default 1', default='1', type=int)

    parser.add_argument("-vts", "--vitis", help='generate vitis version?', action='store_true')
    parser.add_argument("-mem", "--cache_mem", help='use the caching version or not', action='store_true')
    parser.add_argument("-uram", "--use_uram", help="using a caching version with urams, no sens to use without caching", action='store_true')
    parser.add_argument("-ncm", "--n_couples_max", help="sets the positive maximum number of couples of ref and flt passed, default 1", default='1', type=int)
    # parser.add_argument("-sr", "--size_rows", help="number of rows per aie", default='512', type=int)     # TODO decommentare se serve
    # parser.add_argument("-sc", "--size_cols", help="number of columns per aie", default='32', type=int)   # TODO decommentare se serve
    parser.add_argument("-ppr", "--pixels_per_read", help="number of pixels read in one transaction", default='32', type=int)
    parser.add_argument("-intpe", "--interpolator_pe_number", nargs='?', help='number of AIE interpolator PEs, default 1', default='1', type=int)
    args = parser.parse_args()

    derived = ParametersDerived()
    derived.derive(args.in_dim, args.in_bits, args.bin_val, args.pe_number, args.entr_acc_size, args.histotype, args.n_couples_max)

    fixed = (args.histotype == "fixed")
    if args.clean:
        os.remove(os.path.join(args.out_path, "mutual_info.hpp"))

    print_mi_config(
        args.pe_number, args.in_bits,
        args.in_dim, derived, args.pe_entropy,
        fixed, args.cache_mem, args.use_uram, args.vitis, args.out_path,
        args.pixels_per_read, args.interpolator_pe_number
    )


if __name__ == "__main__":
    main()
