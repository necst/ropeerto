
/***************************************************************
*
* High-Level-Synthesis implementation file for Mutual Information computation
*
****************************************************************/

#include <stdio.h>
#include <string.h>
#include "assert.h"
#include "mutual_info.hpp"
#include "hls_math.h"

#include "stdlib.h"

const unsigned int fifo_in_depth =  (N_COUPLES_MAX*MYROWS*MYCOLS)/(HIST_PE);
const unsigned int fifo_out_depth = 1;
const unsigned int pe_j_h_partition = HIST_PE;
const unsigned int maxCouples=N_COUPLES_MAX;

typedef MinHistBits_t HIST_TYPE;
typedef MinHistPEBits_t HIST_PE_TYPE;


typedef enum FUNCTION_T {
	LOAD_IMG = 0,
	COMPUTE = 1
} FUNCTION;


void compute(hls::stream<hls::axis<ap_uint<INPUT_DATA_BITWIDTH>, 0, 0, 0>> & input_img, hls::stream<hls::axis<ap_uint<INPUT_DATA_BITWIDTH>, 0, 0, 0>> & input_ref,  hls::stream<hls::axis<float, 0, 0, 0>> & mutual_info, uint64_t n_couples, unsigned padding){
	//The end_reset params resets the content of j_h;
	//If not set, the PE memories will accumulate over different iterations.
	//It is set to 1 at the end of the data flow.

/*
 * Removes a function as a separate entity in the hierarchy.
 * After inlining, the function is dissolved into the calling function and
 * no longer appears as a separate level of hierarchy in the RTL.
 *
 * In some cases, inlining a function allows operations within the function
 * to be shared and optimized more effectively with the calling function.
 * However, an inlined function cannot be shared or reused,
 * so if the parent function calls the inlined function multiple times,
 * this can increase the area required for implementing the RTL.
 * */

#pragma HLS DATAFLOW

static	hls::stream<INPUT_DATA_TYPE> ref_stream("ref_stream");
#pragma HLS STREAM variable=ref_stream depth=2 dim=1
static	hls::stream<INPUT_DATA_TYPE> flt_stream("flt_stream");
#pragma HLS STREAM variable=flt_stream depth=2 dim=1

static  hls::stream<UNPACK_DATA_TYPE> ref_pe_stream[HIST_PE];
#pragma HLS STREAM variable=ref_pe_stream depth=2 dim=1
static  hls::stream<UNPACK_DATA_TYPE> flt_pe_stream[HIST_PE];
#pragma HLS STREAM variable=flt_pe_stream depth=2 dim=1

static	hls::stream<PACKED_HIST_PE_DATA_TYPE> j_h_pe_stream[HIST_PE];
#pragma HLS STREAM variable=j_h_pe_stream depth=2 dim=1

static	hls::stream<PACKED_HIST_DATA_TYPE> joint_j_h_stream("joint_j_h_stream");
#pragma HLS STREAM variable=joint_j_h_stream depth=2 dim=1
static	hls::stream<PACKED_HIST_DATA_TYPE> joint_j_h_stream_0("joint_j_h_stream_0");
#pragma HLS STREAM variable=joint_j_h_stream_0 depth=2 dim=1
static	hls::stream<PACKED_HIST_DATA_TYPE> joint_j_h_stream_1("joint_j_h_stream_1");
#pragma HLS STREAM variable=joint_j_h_stream_1 depth=2 dim=1
static	hls::stream<PACKED_HIST_DATA_TYPE> joint_j_h_stream_2("joint_j_h_stream_2");
#pragma HLS STREAM variable=joint_j_h_stream_2 depth=2 dim=1

static	hls::stream<PACKED_HIST_DATA_TYPE> row_hist_stream("row_hist_stream");
#pragma HLS STREAM variable=row_hist_stream depth=2 dim=1
static	hls::stream<PACKED_HIST_DATA_TYPE> col_hist_stream("col_hist_stream");
#pragma HLS STREAM variable=col_hist_stream depth=2 dim=1

static	hls::stream<OUT_ENTROPY_TYPE> full_entropy_stream("full_entropy_stream");
#pragma HLS STREAM variable=full_entropy_stream depth=2 dim=1
static	hls::stream<OUT_ENTROPY_TYPE> row_entropy_stream("row_entropy_stream");
#pragma HLS STREAM variable=row_entropy_stream depth=2 dim=1
static	hls::stream<OUT_ENTROPY_TYPE> col_entropy_stream("col_entropy_stream");
#pragma HLS STREAM variable=col_entropy_stream depth=2 dim=1

static	hls::stream<HIST_TYPE> full_hist_split_stream[ENTROPY_PE];
#pragma HLS STREAM variable=full_hist_split_stream depth=2 dim=1
static	hls::stream<HIST_TYPE> row_hist_split_stream[ENTROPY_PE];
#pragma HLS STREAM variable=row_hist_split_stream depth=2 dim=1
static	hls::stream<HIST_TYPE> col_hist_split_stream[ENTROPY_PE];
#pragma HLS STREAM variable=col_hist_split_stream depth=2 dim=1

static	hls::stream<OUT_ENTROPY_TYPE> full_entropy_split_stream[ENTROPY_PE];
#pragma HLS STREAM variable=full_entropy_split_stream depth=2 dim=1
static	hls::stream<OUT_ENTROPY_TYPE> row_entropy_split_stream[ENTROPY_PE];
#pragma HLS STREAM variable=row_entropy_split_stream depth=2 dim=1
static	hls::stream<OUT_ENTROPY_TYPE> col_entropy_split_stream[ENTROPY_PE];
#pragma HLS STREAM variable=col_entropy_split_stream depth=2 dim=1


static	hls::stream<data_t> mutual_information_stream("mutual_information_stream");
#pragma HLS STREAM variable=mutual_information_stream depth=2 dim=1


	// Step 1: read data from DDR and split them
	
	stream2stream_volume<hls::axis<ap_uint<INPUT_DATA_BITWIDTH>, 0, 0, 0>, INPUT_DATA_TYPE, NUM_INPUT_DATA>( input_img, flt_stream, n_couples);
#ifndef CACHING
	stream2stream_volume<hls::axis<ap_uint<INPUT_DATA_BITWIDTH>, 0, 0, 0>, INPUT_DATA_TYPE, NUM_INPUT_DATA>( input_ref, ref_stream, n_couples);
#else
	bram2stream<INPUT_DATA_TYPE, NUM_INPUT_DATA>(ref_stream, input_ref);
#endif

	split_stream_volume<INPUT_DATA_TYPE, UNPACK_DATA_TYPE, UNPACK_DATA_BITWIDTH, NUM_INPUT_DATA, HIST_PE>(ref_stream, ref_pe_stream,n_couples);
	split_stream_volume<INPUT_DATA_TYPE, UNPACK_DATA_TYPE, UNPACK_DATA_BITWIDTH, NUM_INPUT_DATA, HIST_PE>(flt_stream, flt_pe_stream,n_couples);
	// End Step 1


	// Step 2: Compute two histograms in parallel
	WRAPPER_HIST(HIST_PE)<UNPACK_DATA_TYPE, NUM_INPUT_DATA, HIST_PE_TYPE, PACKED_HIST_PE_DATA_TYPE, MIN_HIST_PE_BITS>(ref_pe_stream, flt_pe_stream, j_h_pe_stream,n_couples);
	sum_joint_histogram<PACKED_HIST_PE_DATA_TYPE, J_HISTO_ROWS*J_HISTO_COLS/ENTROPY_PE, PACKED_HIST_DATA_TYPE, HIST_PE, HIST_PE_TYPE, MIN_HIST_PE_BITS, HIST_TYPE, MIN_HIST_BITS>(j_h_pe_stream, joint_j_h_stream, padding);
	// End Step 2


	// Step 3: Compute histograms per row and column
	tri_stream<PACKED_HIST_DATA_TYPE, J_HISTO_ROWS*J_HISTO_COLS/ENTROPY_PE>(joint_j_h_stream, joint_j_h_stream_0, joint_j_h_stream_1, joint_j_h_stream_2);

	hist_row<PACKED_HIST_DATA_TYPE, J_HISTO_ROWS, J_HISTO_COLS/ENTROPY_PE, PACKED_HIST_DATA_TYPE, HIST_TYPE, MIN_HIST_BITS>(joint_j_h_stream_0, row_hist_stream);
	hist_col<PACKED_HIST_DATA_TYPE, J_HISTO_ROWS, J_HISTO_COLS/ENTROPY_PE>(joint_j_h_stream_1, col_hist_stream);
	// End Step 3


	// Step 4: Compute Entropies
	WRAPPER_ENTROPY(ENTROPY_PE)<PACKED_HIST_DATA_TYPE, HIST_TYPE, OUT_ENTROPY_TYPE, J_HISTO_ROWS*J_HISTO_COLS/ENTROPY_PE>(joint_j_h_stream_2, full_hist_split_stream, full_entropy_split_stream, full_entropy_stream);
	WRAPPER_ENTROPY(ENTROPY_PE)<PACKED_HIST_DATA_TYPE, HIST_TYPE, OUT_ENTROPY_TYPE, J_HISTO_ROWS/ENTROPY_PE>(row_hist_stream, row_hist_split_stream, row_entropy_split_stream, row_entropy_stream);
	WRAPPER_ENTROPY(ENTROPY_PE)<PACKED_HIST_DATA_TYPE, HIST_TYPE, OUT_ENTROPY_TYPE, J_HISTO_COLS/ENTROPY_PE>(col_hist_stream, col_hist_split_stream, col_entropy_split_stream, col_entropy_stream);
	// End Step 4


	// Step 6: Mutual information
	compute_mutual_information<OUT_ENTROPY_TYPE, data_t>(row_entropy_stream, col_entropy_stream, full_entropy_stream, mutual_information_stream, n_couples, padding);
	// End Step 6


	// Step 7: Write result back to DDR
	stream2stream_mi<data_t,hls::axis<float, 0, 0, 0>, fifo_out_depth>( mutual_information_stream, mutual_info);

}


template<typename T, unsigned int size>
void copyData(T* in, T* out){
	for(int i = 0; i < size; i++){
#pragma HLS PIPELINE
		out[i] = in[i];
	}
}


//#ifndef CACHING

#ifdef KERNEL_NAME
extern "C"{
	void KERNEL_NAME
#else
	void mutual_information_master
#endif //KERNEL_NAME
(hls::stream<hls::axis<ap_uint<INPUT_DATA_BITWIDTH>, 0, 0, 0>> & input_img, hls::stream<hls::axis<ap_uint<INPUT_DATA_BITWIDTH>, 0, 0, 0>> & input_ref, hls::stream<hls::axis<float, 0, 0, 0>> & mutual_info, hls::stream<hls::axis<ap_uint<INPUT_DATA_BITWIDTH>, 0, 0, 0>> & n_couples, ap_uint<64> axi_ctrl ){

unsigned padding = 0;
#pragma HLS INTERFACE mode=axis register port=input_img name=s_input_img
#pragma HLS INTERFACE mode=axis register port=input_ref name=s_input_ref
#pragma HLS INTERFACE mode=axis register port=mutual_info name=s_mutual_info
#pragma HLS INTERFACE mode=axis register port=n_couples name=s_n_couples

#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS INTERFACE s_axilite port=axi_ctrl bundle=control

// N-couples pragma correctly added


	hls::axis<ap_uint<INPUT_DATA_BITWIDTH>, 0, 0, 0> tmp = n_couples.read();
	uint64_t n_couples_value = tmp.data;

	if(n_couples_value > N_COUPLES_MAX)
		n_couples_value = N_COUPLES_MAX;

	compute(input_img, input_ref, mutual_info, n_couples_value, padding);
}

#ifdef KERNEL_NAME

} // extern "C"


#endif //KERNEL_NAME
