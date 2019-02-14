#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <Windows.h>
#include <cuda_runtime.h>

using namespace std;

//extern void use_saxpy_cuda(int n, float a, float * x, float * y);
extern void use_upsweep(int twod, int offset, int* output);
extern void use_downsweep(int twod, int offset, int* output);
extern void use_pairs_repeat(int n, int* x, int* x_shift, int* repeat);

void exclusive_scan(int* start, int* end, int* output)
{
	int N = end - start;
	//memmove(output, start, N * sizeof(int));
	// upsweep phase.
	int offset = 1;
	for (int twod = N>>1; twod > 0; twod >>= 1)
	{
		use_upsweep(twod, offset, output);
		offset *= 2;
	}
	cudaMemset(output+(N-1), 0,sizeof(int));
	//output[N - 1] = 0;
	// downsweep phase.
	for (int twod = 1; twod < N; twod *= 2)
	{
		offset >>= 1;
		use_downsweep(twod, offset, output);
	}
}


int find_repeats(int n, int* x) {

	int* x_shift;
	int* repeat;
	int* num_repeats;

	cudaMalloc(&x_shift, (n + 1) * sizeof(int));
	//cudaMallocManaged(&x_shift, (n + 1) * sizeof(int)); //DEBUG
	cudaMemcpy(x_shift+1, x, n * sizeof(int), cudaMemcpyDeviceToDevice);

	cudaMalloc(&repeat, (n) * sizeof(int));
	//cudaMallocManaged(&repeat, (n) * sizeof(int));
	//cudaMemset(&repeat, 0, (n) * sizeof(int)); // May not need

	use_pairs_repeat(n, x, x_shift, repeat);

	int* start = &repeat[0];
	int* end = &repeat[n];
	exclusive_scan(start, end, repeat);

	num_repeats = new int;

	cudaMemcpy(num_repeats, repeat+(n-1), sizeof(int), cudaMemcpyDeviceToHost);

	int ret_repeats = *num_repeats;

	cudaFree(x_shift);
	cudaFree(repeat);
	delete num_repeats;

	return ret_repeats;
}

int main(int argc, const char* argv[])
{
	int n = 1 << 10;
	int *x_cu;
	int num_repeats;

	//x_cu = (int*) malloc(sizeof(int)*n);

	cudaMallocManaged(&x_cu, n * sizeof(int));
	
	
	for (int i = 0; i < n; i++) {
		x_cu[i] = 1;
	}

	/*
	for (int i = 0; i < 5; i++) {
		cout << "x: " << x_cu[i] << endl;
	}
	*/

	num_repeats = find_repeats(n, x_cu);

	/*
	//saxpy(n, a, x, y); // CPU version
	for (int i = 0; i < 5; i++) {
		cout << "x: " << x_cu[i] << endl;
	}
	*/

	cout << "num_repeats is " << num_repeats << endl;

	Sleep(5000);

	
	//free(x_cu); // CPU version
	cudaFree(x_cu);
}