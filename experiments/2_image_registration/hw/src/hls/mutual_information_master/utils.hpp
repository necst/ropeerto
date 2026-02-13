
/***************************************************************
*
* Utilities for the code to be synthesized
*
****************************************************************/
#ifndef UTILS_HPP
#define UTILS_HPP

#include "hls_stream.h"
#include "mutual_info.hpp"

constexpr int TRIP_VARIABLE_UTILS = N_COUPLES_MAX;

template<typename T, unsigned int size>
void axi2stream(hls::stream<T> &out,const T* in){
    for(int i = 0; i <size; i++){
        #pragma HLS PIPELINE
        T tmp = in[i];
        out.write(tmp);
    }
}

template<typename T, unsigned int size>
void axi2stream_volume(hls::stream<T> &out,const T* in,int n_couples){
    for(int i = 0; i <size*n_couples; i++){
		#pragma HLS LOOP_TRIPCOUNT min=1 max=TRIP_VARIABLE_UTILS

        #pragma HLS PIPELINE
        T tmp = in[i];
        out.write(tmp);
    }
}

template<typename T, typename U, unsigned int size>
void stream2stream_volume(hls::stream<T> &in, hls::stream<U> &out,int n_couples){
    for(int i = 0; i <size*n_couples; i++){
		#pragma HLS LOOP_TRIPCOUNT min=1 max=TRIP_VARIABLE_UTILS

        #pragma HLS PIPELINE
        T tmp = in.read();
        out.write(tmp.data);
    }
}

template<typename T, typename U, unsigned int size>
void stream2stream_mi(hls::stream<T> &in, hls::stream<U> &out){
    U tmp;
    tmp.data = in.read();
    tmp.last = 1;
    tmp.keep = 0xFF;
    out.write(tmp);
    
}


template<typename T, unsigned int size>
void bram2stream(hls::stream<T> &out, const T* in){
    for(int i = 0; i <size; i++){
        #pragma HLS PIPELINE
        T tmp = in[i];
        out.write(tmp);
    }
}

template<typename T, unsigned int size>
void axi2stream_split(hls::stream<T> &out0, hls::stream<T> &out1, const T* in){
    for(int i = 0; i <size; i++){
#pragma HLS PIPELINE
        T tmp = in[i];
        if(!(i%2)){
			out0.write(tmp);
		} else {
			out1.write(tmp);
		}
    }
}


template<typename Tin, typename Tout, unsigned int out_bitwidth, unsigned int size, unsigned int STREAM>
void split_stream(hls::stream<Tin> &in, hls::stream<Tout> out[STREAM]){
    for(int i = 0; i <size; i++){

        #pragma HLS PIPELINE
        Tin tmp = in.read();
        for(int j = 0; j < STREAM; j++){
        	ap_uint<out_bitwidth> unpacked = tmp.range((j+1)*out_bitwidth - 1, j*out_bitwidth);
        	Tout elem = *((Tout *)&unpacked);
        	out[j].write(elem);
        }
    }
}


template<typename Tin, typename Tout, unsigned int out_bitwidth, unsigned int size, unsigned int STREAM>
void split_stream_volume(hls::stream<Tin> &in, hls::stream<Tout> out[STREAM],int n_couples){
    for(int i = 0; i <size*n_couples; i++){
		#pragma HLS LOOP_TRIPCOUNT min=1 max=TRIP_VARIABLE_UTILS

        #pragma HLS PIPELINE
        Tin tmp = in.read();
        for(int j = 0; j < STREAM; j++){
        	ap_uint<out_bitwidth> unpacked = tmp.range((j+1)*out_bitwidth - 1, j*out_bitwidth);
        	Tout elem = *((Tout *)&unpacked);
        	out[j].write(elem);
        }
    }
}


template<typename Tin, typename Tout, unsigned int out_bitwidth, unsigned int size>
void convert_stream(hls::stream<Tin> &in, hls::stream<Tout> &out){
    for(int i = 0; i <size; i++){
        #pragma HLS PIPELINE
        Tin in_val = in.read();
        unsigned int tmp = in_val;
        Tout elem = *((Tout *)&tmp);
        out.write(elem);
    }
}


template<typename T, unsigned int size>
void tri_stream(hls::stream<T> &in, hls::stream<T> &out0, hls::stream<T> &out1, hls::stream<T> &out2){
    for(int i = 0; i <size; i++){
        #pragma HLS PIPELINE
        T tmp = in.read();
        out0.write(tmp);
		out1.write(tmp);
		out2.write(tmp);
    }
}


// template<typename T, unsigned int size>
// void stream2axi(T* out, hls::stream<T> &in, bool end_reset){

// 		for(int i = 0; i <size; i++){
// 			#pragma HLS PIPELINE
// 			T tmp = in.read();
// 			if(end_reset)
// 				out[i] = tmp;
// 		}

// }

template<typename T, unsigned int size>
void stream2axi(T* out, hls::stream<T> &in){

		for(int i = 0; i <size; i++){
			#pragma HLS PIPELINE
			T tmp = in.read();
			out[i] = tmp;
		}

}

template<typename T, unsigned int size>
void join_and_sum(hls::stream<T> &in0, hls::stream<T> &in1, hls::stream<T> &out){
    for(int i = 0; i < size; i++){
        #pragma HLS PIPELINE
        T tmp0 = in0.read();
        T tmp1 = in1.read();
        T tmp2 = tmp0 + tmp1;
        out.write(tmp2);
    }
}



#endif // UTILS_HPP
