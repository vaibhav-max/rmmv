#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <cassert>


// Create other necessary functions here
__global__ void RMM(int N, const int* A, const int* B, int* O)
{
	// enforce N to be power of 2 and greater than 2
	assert(N >= 4 and N == (N & ~(N - 1)));

	// Compute each thread's global row and column index
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row * (N>>1) + col;//will calculate the index of next thread
	row = row<<1;
	col = col<<1;
	int sum = 0;
	for (int k = 0; k < N; k++)
	{
		sum += A[row * N + k] *	 B[k * N + col];
		sum += A[(row + 1) * N + k] * B[k * N + col];
		sum += A[row * N + k] * B[k * N + (col + 1)];
		sum += A[(row + 1) * N + k] * B[k * N + (col + 1)];
	}
	O[index] = sum;
}

// Fill in this function
void gpuThread(int N, int *matA, int *matB, int *output)
{
auto begin = TIME_NOW;
	size_t bytes = N * N * sizeof(int);

	// Allocating CUDA memory
	int* dA, * dB, * dO;
	cudaMalloc(&dA, bytes);
	cudaMalloc(&dB, bytes);
	cudaMalloc(&dO, bytes);

	//transfer date from CPU to MEM
	cudaMemcpy(dA, matA, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, matB, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dO, output, bytes, cudaMemcpyHostToDevice);

	
	int THREADS = 32;// no. of threads per block ,considering 32 because it has to be translate to warps that is of size 32 therefore we take threads as multiple of 32 for performance
	int BLOCKS = N / (2*THREADS);//no. of blocks
	
	dim3 blocks(BLOCKS, BLOCKS);
	dim3 threads(THREADS, THREADS);
	
	RMM << <blocks, threads >> > (N, dA, dB, dO);//Kernal call

	// transfering data(output) from GPU ot CPU memory
	cudaMemcpy(output, dO, bytes, cudaMemcpyDeviceToHost);
auto end = TIME_NOW;
cout << "GPU execution time: " <<(double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n";
}
