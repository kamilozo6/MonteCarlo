
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <time.h>
#include <iostream>
#include <chrono> 
using namespace std::chrono;

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__device__ void ChangeState(int result, int& yes, int& no, int& unknown)
{
	switch (result)
	{
	case 0:
		// both the same
		break;
	case 1:
		// yes + no
		no--;
		yes--;
		unknown += 2;
		break;
	case 2:
		// yes + unknown
		yes++;
		unknown--;
		break;
	case 3:
		// no + unknown
		no++;
		unknown--;
		break;
	default:
		break;
	}
}

int CountSize(int n)
{
	int size = 0;
	for (int i = 1; i <= n + 1; i++)
	{
		size += i;
	}
	return size;
}

__device__ int EvaluateCase(int yes, int no, int unknown, int n, curandState* randState)
{
	int randomValue = (int)(curand_uniform(&randState[0]) * n) % n + 1;
	// 0 yes, 1 no, 2 unknown
	int firstSelection;
	int secondSelection;
	if (randomValue <= yes)
	{
		firstSelection = 0;
		yes--;
	}
	else if (randomValue <= yes + no)
	{
		firstSelection = 1;
		no--;
	}
	else
	{
		firstSelection = 2;
		unknown--;
	}

	n--;
	randomValue = (int)(curand_uniform(&randState[0]) * n) % n + 1;
	if (randomValue <= yes)
	{
		secondSelection = 0;
		yes--;
	}
	else if (randomValue <= yes + no)
	{
		secondSelection = 1;
		no--;
	}
	else
	{
		secondSelection = 2;
		unknown--;
	}
	int result = firstSelection ^ secondSelection;
	return result;
}

__device__ void GetYesNoFromIndex(int index, int n, int& yes, int& no)
{
	int currentIndex = n;
	int iterator = 1;
	if (index <= n)
	{
		yes = 0;
		no = index;
		return;
	}

	while ((currentIndex + n + 1 - iterator) < index)
	{
		currentIndex += n + 1 - iterator;
		iterator++;
	}
	yes = iterator;
	no = index - currentIndex - 1;
}

// index | yes|no
//   0   |  0 | 0
//   1   |  0 | 1
//   2   |  0 | 2
//   .   |  . | .
//   n   |  0 | n
//  n+1  |  1 | 0
//  n+2  |  1 | 1
//   .   |  . | .
// n+n-1 |  1 | n - yes
//  n+n  |  2 | 0
//  and  |  . | .
//   so  | n-1| n - yes
//   on  |  n | 0
__device__ int ReturnIndex(int yes, int no, int n)
{
	int index = 0;

	for (int i = 1; i <= yes; i++)
	{
		index += n + 2 - i;
	}

	index += no;
	return index;
}

__global__ void MonteCarlo(int* outNumerators, int* outDenominators, int states, int peoples, int iterationsNum)
{
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tId >= states)
	{
		return;
	}
	// Init random numbers generator
	curandState randState;
	curand_init((unsigned long long)clock() + tId, 0, 0, &randState);

	for (int i = 0; i < iterationsNum; i++)
	{
		bool end = false;
		bool isYesResult = false;
		// Begining state
		int test = tId;
		int yes = peoples / 2, no = peoples / 2, unknown = peoples - yes - no;
		// Get yes, no numbers according to state
		GetYesNoFromIndex(test, peoples, yes, no);
		unknown = peoples - yes - no;
		while (!end)
		{
			int result = EvaluateCase(yes, no, unknown, peoples, &randState);
			ChangeState(result, yes, no, unknown);

			// If yes == 0 there is no chance to "win"
			// If no == 0 there is no chance to "lose"
			if (yes == 0 || no == 0)
			{
				if (yes > 0)
				{
					isYesResult = true;
				}
				end = true;
			}
		}
		if (isYesResult)
		{
			outNumerators[tId] ++;
		}
		outDenominators[tId] ++;
	}
}

extern __shared__ int shared[];
__global__ void MonteCarloOpt(double* winProbabilities, int states, int peoples, int iterationsNum)
{
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tId >= states)
	{
		return;
	}

	// Init random numbers generator
	curandState randState;
	curand_init((unsigned long long)clock() + tId, 0, 0, &randState);

	for (int i = 0; i < iterationsNum; i++)
	{
		bool end = false;
		bool isYesResult = false;
		// Begining state
		int test = tId;
		int yes = peoples / 2, no = peoples / 2, unknown = peoples - yes - no;
		// Get yes, no numbers according to state
		GetYesNoFromIndex(test, peoples, yes, no);
		unknown = peoples - yes - no;
		while (!end)
		{
			int result = EvaluateCase(yes, no, unknown, peoples, &randState);
			ChangeState(result, yes, no, unknown);

			// If yes == 0 there is no chance to "win"
			// If no == 0 there is no chance to "lose"
			if (yes == 0 || no == 0)
			{
				if (yes > 0)
				{
					isYesResult = true;
				}
				end = true;
			}
		}
		if (isYesResult)
		{
			winProbabilities[tId] += 1.0;
		}
	}
	winProbabilities[tId] /= iterationsNum;
}

__global__ void CalculateProbabilities(int* numerators, int* denominators, double* winProbabilities, int states)
{
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tId < states)
	{
		if (denominators[tId] != 0)
			winProbabilities[tId] = numerators[tId] * 1.0 / denominators[tId];
		else
			winProbabilities[tId] = 0;
	}
}

void NonOptimized();
void Optimized();

// Size control varaibles
#define PEOPLE_NUM 250
#define ITERATIONS_NUM 10000
#define THREAD_NUM 256
//#define PRINT_RES

int main()
{
	std::cout << "People number: " << PEOPLE_NUM << std::endl;
	std::cout << "Iterations number: " << ITERATIONS_NUM << std::endl;
	std::cout << "States number: " << CountSize(PEOPLE_NUM) << std::endl;	

	auto start = high_resolution_clock::now();
	NonOptimized();
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	std::cout << "Non optimized time: \t" << duration.count() << "ms" << std::endl;

	start = high_resolution_clock::now();
	Optimized();
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start);
	std::cout << "Optimized time: \t" << duration.count() << "ms" << std::endl;

	return 0;
}

void NonOptimized()
{
	// Monte carlo init numbers
	const int iterationsNumber = ITERATIONS_NUM;
	const int peopleNumber = PEOPLE_NUM;
	const int statesNumber = CountSize(peopleNumber);
	double* winProbabilities = new double[statesNumber];
	// Threads and blocks
	const int threadsNumber = THREAD_NUM;
	int blocksNumber;
	// Cuda variables
	double* cu_winProbabilities;
	int* cu_statesNumerators;
	int* cu_statesDenominators;
	cudaError_t cudaStatus;	

	// Allocate memory on gpu
	cudaStatus = cudaMalloc((void**)& cu_winProbabilities, statesNumber * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate (error code %s)!\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaMalloc((void**)& cu_statesNumerators, statesNumber * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate (error code %s)!\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaMalloc((void**)& cu_statesDenominators, statesNumber * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate (error code %s)!\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}

	// Set numerators and denominators values to 0
	cudaMemset(cu_statesNumerators, 0, statesNumber * sizeof(int));
	cudaMemset(cu_statesDenominators, 0, statesNumber * sizeof(int));
	cudaMemset(cu_winProbabilities, 0, statesNumber * sizeof(double));

	// Monte Carlo simulation
	blocksNumber = (statesNumber + threadsNumber - 1) / threadsNumber;
	MonteCarlo << <blocksNumber, threadsNumber >> > (cu_statesNumerators, cu_statesDenominators, statesNumber, peopleNumber, iterationsNumber);
	

	// Calculate states probabilities to win
	blocksNumber = (statesNumber + threadsNumber - 1) / threadsNumber;
	CalculateProbabilities << <blocksNumber, threadsNumber >> > (cu_statesNumerators, cu_statesDenominators, cu_winProbabilities, statesNumber);

	// Copy results to host
	cudaMemcpy(winProbabilities, cu_winProbabilities, statesNumber * sizeof(double), cudaMemcpyDeviceToHost);

#ifdef PRINT_RES
	// Print results
	for (int i = 0; i < statesNumber; i++)
	{
		std::cout << winProbabilities[i] << std::endl;
	}
#endif

	// Free memory
	cudaFree(cu_winProbabilities);
	cudaFree(cu_statesNumerators);
	cudaFree(cu_statesDenominators);

	delete winProbabilities;
}

void Optimized()
{
	// Monte carlo init numbers
	const int iterationsNumber = ITERATIONS_NUM;
	const int peopleNumber = PEOPLE_NUM;
	const int statesNumber = CountSize(peopleNumber);
	double* winProbabilities = new double[statesNumber];
	// Threads and blocks
	const int threadsNumber = THREAD_NUM;
	int blocksNumber;
	// Cuda variables
	double* cu_winProbabilities;
	cudaError_t cudaStatus;

	// Allocate memory on gpu
	cudaStatus = cudaMalloc((void**)& cu_winProbabilities, statesNumber * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate (error code %s)!\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}

	// Set numerators and denominators values to 0
	cudaMemset(cu_winProbabilities, 0, statesNumber * sizeof(double));

	// Monte Carlo simulation
	blocksNumber = (statesNumber + threadsNumber - 1) / threadsNumber;
	MonteCarloOpt << <blocksNumber, threadsNumber >> > (cu_winProbabilities, statesNumber, peopleNumber, iterationsNumber);
	
	// Copy results to host
	cudaMemcpy(winProbabilities, cu_winProbabilities, statesNumber * sizeof(double), cudaMemcpyDeviceToHost);

#ifdef PRINT_RES
	// Print results
	for (int i = 0; i < statesNumber; i++)
	{
		std::cout << winProbabilities[i] << std::endl;
	}
#endif

	// Free memory
	cudaFree(cu_winProbabilities);

	delete winProbabilities;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)& dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
