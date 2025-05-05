#include "Kernels.cuh"
#include <iostream>

#include <cuda_runtime.h>

int main(int argc, char ** argv)
{
	if (argc != 3)
	{
		std::cerr << "Usage: " << argv[0] << " <mode> <N> " << std::endl;
		return -1;
	}

	unsigned int N = atoi(argv[2]);
	int mode = atoi(argv[1]);
	std::cout << "Running with mode: " << mode << std::endl;
	unsigned int size = N * N;

	float* h_a, * h_b, * h_c;

	h_a = (float*)malloc(size * sizeof(float));
	h_b = (float*)malloc(size * sizeof(float));
	h_c = (float*)malloc(size * sizeof(float));

	generate_float_matrix(h_a, N, N);
	generate_float_matrix(h_b, N, N);

	float* d_a, * d_b, * d_c;
	cudaMalloc((void**)&d_a, size * sizeof(float));
	cudaMalloc((void**)&d_b, size * sizeof(float));
	cudaMalloc((void**)&d_c, size * sizeof(float));

	cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, size * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	float elapsed = 0.0f;
    
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	run_kernel(mode, N, N, N, d_a, d_b, d_c, 1.0f, 0.0f);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);

	// show result
	float flop = 1.0 * N * N * (2 * N + 1);
	float gflop = flop / 1000000000.0f;
	elapsed = elapsed / 1000.0f; // to second

	std::cout
		<< "Time:                                   \t" << elapsed << " s.\n"
		<< "GFlop:                                  \t" << gflop << "\n"
		<< "GFLOPS:                                 \t" << gflop / elapsed << "\n";

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}