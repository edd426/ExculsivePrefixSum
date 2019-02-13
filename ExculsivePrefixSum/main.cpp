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


void exclusive_scan(int* start, int* end, int* output)
{
	int N = end - start;

	memmove(output, start, N * sizeof(int));


	// upsweep phase.
	int offset = 1;
	for (int twod = N>>1; twod > 0; twod >>= 1)
	{
		use_upsweep(twod, offset, output);
		offset *= 2;
	}

	output[N - 1] = 0;

	// downsweep phase.
	for (int twod = 1; twod < N; twod *= 2)
	{
		offset >>= 1;
		use_downsweep(twod, offset, output);

	}

}

int main(int argc, const char* argv[])
{
	int n = 1 << 10;
	int s = 0;
	int *x_cu;

	//x_cu = (int*) malloc(sizeof(int)*n);

	cudaMallocManaged(&x_cu, n * sizeof(int));

	for (int i = 0; i < n; i++) {
		x_cu[i] = 1;
	}

	for (int i = 0; i < 5; i++) {
		cout << "x: " << x_cu[i] << endl;
	}
	int* start = &x_cu[0];
	int* end = &x_cu[n];
	exclusive_scan_iterative(start, end, x_cu);

	//saxpy(n, a, x, y); // CPU version
	for (int i = 0; i < 5; i++) {
		cout << "x: " << x_cu[i] << endl;
	}

	Sleep(5000);

	
	//free(x_cu); // CPU version
	cudaFree(x_cu);
}