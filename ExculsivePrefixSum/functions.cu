#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void upsweep(int twod, int offset, int* output)
{
	int index = threadIdx.x;

	if (index < twod) {
		int ai = offset * (2 * index + 1) - 1;
		int bi = offset * (2 * index + 2) - 1;

		output[bi] += output[ai];

	}
}


__global__ void downsweep(int twod, int offset, int* output)
{
	int index = threadIdx.x;

	if (index < twod) {
		int ai = offset * (2 * index + 1) - 1;
		int bi = offset * (2 * index + 2) - 1;

		int t = output[ai];
		output[ai] = output[bi];
		output[bi] += t;

	}
}


__global__ void pairs_repeat(int n, int* x, int* x_shift, int* repeat)
{
	int index = threadIdx.x;

	if (index > 0)
		repeat[index-1] = x[index] == x_shift[index];


}

extern void use_upsweep(int twod, int offset, int* output)
{
	int threadsPerBlock = 1024;
	//int numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

	upsweep << <1, threadsPerBlock >> > (twod, offset, output);
	cudaDeviceSynchronize();
}

extern void use_downsweep(int twod, int offset, int* output)
{
	int threadsPerBlock = 1024;
	//int numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

	downsweep << <1, threadsPerBlock >> > (twod, offset, output);
	cudaDeviceSynchronize();
}

extern void use_pairs_repeat(int n, int* x, int* x_shift, int* repeat)
{
	int threadsPerBlock = 1024;
	//int numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

	pairs_repeat << <1, threadsPerBlock >> > (n, x, x_shift, repeat);
	cudaDeviceSynchronize();
}
