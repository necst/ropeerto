#include "mutual-info-3D.cuh"

#include <iostream>

#include <timer.hpp>

#define cudaErrchk(ans)                       \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{

	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

// #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// #else
// inline __device__ double atomicAdd(double *address, double val)
// {
// 	unsigned long long int *address_as_ull =
// 		(unsigned long long int *)address;
// 	unsigned long long int old = *address_as_ull, assumed;
// 	do
// 	{
// 		assumed = old;
// 		old = atomicCAS(address_as_ull, assumed,
// 						__double_as_longlong(val +
// 											 __longlong_as_double(assumed)));
// 	} while (assumed != old);
// 	return __longlong_as_double(old);
// }
// #endif

template <typename T>
__inline__ __device__ void warpReduce(volatile T *input,
									  size_t threadId, size_t dim)
{
	input[threadId] += input[threadId + 32];
	input[threadId] += input[threadId + 16];
	input[threadId] += input[threadId + 8];
	input[threadId] += input[threadId + 4];
	input[threadId] += input[threadId + 2];
	input[threadId] += input[threadId + 1];
}

template <typename T>
inline __device__ T accumulate(T *input, size_t dim)
{
	size_t threadId = threadIdx.x;
	if (dim > 32)
	{
		for (size_t i = dim / 2; i > 32; i >>= 1)
		{
			for (size_t j = 0; j < i; j += GPU_THREADS)
			{
				if (threadId + j < i)
				{
					input[threadId + j] += input[threadId + j + i];
				}
				__syncthreads();
			}
		}
	}
	if (threadId < 32)
		warpReduce(input, threadId, dim);
	__syncthreads();
	return input[0];
}

template <typename T>
__inline__ __device__ void zeroVector(T *input, size_t dim)
{
	size_t threadId = threadIdx.x;
	for (size_t i = 0; i < dim; i += GPU_THREADS)
	{
		if (threadId + i < dim)
			input[threadId + i] = 0;
	}
	__syncthreads();
}

/*
Idea to make it 3D:
Allocate href and hflt marginal histograms in global memory: one per block.
Each thread will generate the marginal histograms for the block it belongs to.
Then, we will have to reduce the histograms (joint and both marginals) to get the final histograms.
Finally, we will calculate the entropies and mutual information.
Should this final step be done in a single block?
*/

__global__ void mutual_information_3D(
	MY_PIXEL *ref_d,
	MY_PIXEL *flt_d,
	data_t *j_h_d,
	data_t *href_d,
	data_t *hflt_d,
	data_t *mutual_info)
{

	size_t blockId = blockIdx.x;
	size_t threadId = threadIdx.x;

	MY_PIXEL *ref = &ref_d[blockId * DIMENSION * DIMENSION];
	MY_PIXEL *flt = &flt_d[blockId * DIMENSION * DIMENSION];

	data_t *j_h = &j_h_d[blockId * J_HISTO_ROWS * J_HISTO_ROWS];
	// data_t *href = &href_d[blockId * ANOTHER_DIMENSION];
	// data_t *hflt = &hflt_d[blockId * ANOTHER_DIMENSION];
	__shared__ data_t href[ANOTHER_DIMENSION];
	__shared__ data_t hflt[ANOTHER_DIMENSION];
	//
	// zeroVector(j_h,J_HISTO_ROWS * J_HISTO_ROWS);
	zeroVector(href,ANOTHER_DIMENSION);
	zeroVector(hflt,ANOTHER_DIMENSION);
	// __shared__ unsigned atomicAdd_cache[GPU_THREADS];
	for (size_t i = 0; i < DIMENSION * DIMENSION; i += GPU_THREADS)
	{
		size_t a = ref[i + threadId];
		size_t b = flt[i + threadId];
		atomicAdd(&j_h[a + b * J_HISTO_COLS], 1);
	}
	__syncthreads();

	for (size_t i = 0; i < ANOTHER_DIMENSION; i++)
	{
		for (size_t j = 0; j < ANOTHER_DIMENSION; j += GPU_THREADS)
		{
			if (threadId + j < ANOTHER_DIMENSION)
			{
				href[threadId + j] += j_h[(threadId + j) + ANOTHER_DIMENSION * i] / (DIMENSION * DIMENSION);
			}
		}
	}
	__syncthreads();

	// The access here is uncoalesced, it might be possible to optimize it
	for (size_t i = 0; i < ANOTHER_DIMENSION; i++)
	{
		for (size_t j = 0; j < ANOTHER_DIMENSION; j += GPU_THREADS)
		{
			if (threadId + j < ANOTHER_DIMENSION)
			{
				hflt[threadId + j] += j_h[(threadId + j) * ANOTHER_DIMENSION + i] / (DIMENSION * DIMENSION);
			}
		}
	}
	__syncthreads();

	// ENTROPIES

	// ********************************************************** //
	// Some imprecision is introduced in this code section due to multiple reductions
	__shared__ data_t tmp_entr[GPU_THREADS];
	zeroVector(tmp_entr, GPU_THREADS);
	// data_t entropy = 0.0;

	for (size_t i = 0; i < J_HISTO_ROWS * J_HISTO_COLS; i += GPU_THREADS)
	{
		data_t v = j_h[threadId + i] / (DIMENSION * DIMENSION);
		data_t toAdd = v * log2f(v);
		if (v != 0)
		{
			tmp_entr[threadId] += toAdd;
		}
	}
	__syncthreads();
	data_t entropy = -accumulate(tmp_entr, GPU_THREADS);
	// ********************************************************** //

	__shared__ data_t tmp_er[ANOTHER_DIMENSION];
	__shared__ data_t tmp_ef[ANOTHER_DIMENSION];
	zeroVector(tmp_er, ANOTHER_DIMENSION);
	zeroVector(tmp_ef, ANOTHER_DIMENSION);
	// data_t eref = 0.0;
	for (size_t i = 0; i < ANOTHER_DIMENSION; i += GPU_THREADS)
	{
		if (threadId + i < ANOTHER_DIMENSION)
		{
			data_t tmp_val_r = href[threadId + i];
			data_t tmp_val_t = hflt[threadId + i];

			if (tmp_val_r != 0)
			{
				tmp_er[threadId + i] = tmp_val_r * log2f(tmp_val_r); /// log2;
			}
			if (tmp_val_t != 0)
			{
				tmp_ef[threadId + i] = tmp_val_t * log2f(tmp_val_t); /// log2;
			}
		}
	}
	__syncthreads();

	data_t eref = -accumulate(tmp_er, ANOTHER_DIMENSION);
	data_t eflt = -accumulate(tmp_ef, ANOTHER_DIMENSION);

	if (threadId == 0)
		mutual_info[blockId] = eref + eflt - entropy;
}

void mutual_information_3D_master(MY_PIXEL *ref, MY_PIXEL *flt, MY_PIXEL *ref_d, MY_PIXEL *flt_d, data_t *j_h_d,
							   data_t *href_d, data_t *hflt_d, data_t *result_d, data_t *result, size_t n_images, double *kernel_only_time)
{

	std::cout << "NUM IMAGE PAIRS: " << n_images << std::endl;
	cudaErrchk(cudaMemset(j_h_d, 0, n_images * J_HISTO_ROWS * J_HISTO_ROWS * sizeof(data_t)));
	cudaErrchk(cudaMemset(href_d, 0, n_images * ANOTHER_DIMENSION * sizeof(data_t)));
	cudaErrchk(cudaMemset(hflt_d, 0, n_images * ANOTHER_DIMENSION * sizeof(data_t)));

	// copy images on the GPU
	cudaErrchk(cudaMemcpyAsync(ref_d, ref, n_images * DIMENSION * DIMENSION * sizeof(MY_PIXEL), cudaMemcpyHostToDevice, 0));
	cudaErrchk(cudaMemcpyAsync(flt_d, flt, n_images * DIMENSION * DIMENSION * sizeof(MY_PIXEL), cudaMemcpyHostToDevice, 0));
	// schedule kernel execution
	
	Timer timer;
	timer.start();
	mutual_information_3D<<<n_images, GPU_THREADS, 0>>>(ref_d, flt_d, j_h_d, href_d, hflt_d, result_d);
	cudaDeviceSynchronize();
	double time_GPU_kernel_only = timer.stop();

	// copy back results
	cudaErrchk(cudaMemcpyAsync(result, result_d, n_images * sizeof(data_t), cudaMemcpyDeviceToHost, 0));
	// save time
	*kernel_only_time = time_GPU_kernel_only;
}
