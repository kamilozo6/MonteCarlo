
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

__global__ void MonteCarloOpt(double* winProbabilities, int states, int peoples, int iterationsNum)
{
	extern __shared__ double s[];
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tId >= states)
	{
		return;
	}
	// Init random numbers generator
	curandState randState;
	curand_init((unsigned long long)clock() + tId, 0, 0, &randState);

	int iterationsPerThread = iterationsNum / blockDim.x + 1;
	s[threadIdx.x] = 0.0;

	for (int i = 0; i < iterationsPerThread; i++)
	{
		if (iterationsPerThread * threadIdx.x + i >= iterationsNum)
		{
			break;
		}
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

__global__ void MonteCarloMPI(double* winProbabilities, int states, int peoples, int iterationsNum, int rank, int allStates)
{
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tId + rank * states >= allStates)
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
		int test = tId + rank * states;
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
		winProbabilities[tId] /= iterationsNum;
	}
}

double* NonOptimizedMPI(int procSize, int rank, int sizePerProc);

// Size control varaibles
#define PEOPLE_NUM 150
#define ITERATIONS_NUM 10000
#define THREAD_NUMBER 256
#define OPT_THREAD_NUMBER 256
//#define PRINT_RES

double* mains(int rank, int proccount, int* outSize, int* outProcSize)
{
	int size = CountSize(PEOPLE_NUM);
	int sizePerProc = size / proccount;
	int procSize;
	// if (proccount - 1 == rank)
	// {
	// 	procSize = (size - (proccount - 1) * sizePerProc);
	// }
	// else
	// {
	// 	procSize = sizePerProc;
	// }
	procSize = sizePerProc + 1;
	*outProcSize = procSize;
	*outSize = size;
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	int count;
	cudaGetDeviceCount(&count);
	printf("free: %d total: %d count: %d\n", free, total, count);

	return NonOptimizedMPI(procSize, rank, sizePerProc);
}

double* NonOptimizedMPI(int procSize, int rank, int sizePerProc)
{
	// Monte carlo init numbers
	const int iterationsNumber = ITERATIONS_NUM;
	const int peopleNumber = PEOPLE_NUM;
	const int statesNumber = CountSize(peopleNumber);
	double* winProbabilities = new double[procSize];

	// Threads and blocks
	const int threadsNumber = THREAD_NUMBER;
	int blocksNumber;
	// Cuda variables
	double* cu_winProbabilities;
	cudaError_t cudaStatus;
	printf("rank %d\n", rank);

	printf("procSize %d\n", procSize);
	// Allocate memory on gpu
	cudaStatus = cudaMalloc((void**)& cu_winProbabilities, procSize * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate (error code %s)!\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}

	printf("rank %d\n", procSize);
	// Set numerators and denominators values to 0
	cudaMemset(cu_winProbabilities, 0, procSize * sizeof(double));

	// Monte Carlo simulation
	blocksNumber = (procSize + threadsNumber - 1) / threadsNumber;
	MonteCarloMPI << <blocksNumber, threadsNumber >> > (cu_winProbabilities, procSize, peopleNumber, iterationsNumber, rank, sizePerProc);


	// Copy results to host
	cudaMemcpy(winProbabilities, cu_winProbabilities, procSize * sizeof(double), cudaMemcpyDeviceToHost);

#ifdef PRINT_RES
	// Print results
	for (int i = 0; i < procSize; i++)
	{
		std::cout << winProbabilities[i] << std::endl;
	}
#endif

	// Free memory
	cudaFree(cu_winProbabilities);

	return winProbabilities;
}

void SequentialRun();
void NonOptimized();
void Optimized();

int main()
{
	std::cout << "People number: " << PEOPLE_NUM << std::endl;
	std::cout << "Iterations number: " << ITERATIONS_NUM << std::endl;
	std::cout << "States number: " << CountSize(PEOPLE_NUM) << std::endl;

	auto start = high_resolution_clock::now();
	//SequentialRun();
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	std::cout << "Sequential time: \t" << duration.count() << "ms" << std::endl;

	start = high_resolution_clock::now();
	Optimized();
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start);
	std::cout << "Optimized time: \t" << duration.count() << "ms" << std::endl;

	start = high_resolution_clock::now();
	NonOptimized();
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start);
	std::cout << "Non optimized time: \t" << duration.count() << "ms" << std::endl;

	return 0;
}

void SequentialRun()
{
	// Monte carlo init numbers
	const int iterationsNumber = ITERATIONS_NUM;
	const int peopleNumber = PEOPLE_NUM;
	const int statesNumber = CountSize(peopleNumber);
	double* winProbabilities = new double[statesNumber];
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
	cudaMemset(cu_winProbabilities, 0, statesNumber * sizeof(double));

	// Monte Carlo simulation
	MonteCarloSeq << <1, 1 >> > (cu_winProbabilities, statesNumber, peopleNumber, iterationsNumber);


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

void NonOptimized()
{
	// Monte carlo init numbers
	const int iterationsNumber = ITERATIONS_NUM;
	const int peopleNumber = PEOPLE_NUM;
	const int statesNumber = CountSize(peopleNumber);
	double* winProbabilities = new double[statesNumber];
	// Threads and blocks
	const int threadsNumber = THREAD_NUMBER;
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
	MonteCarlo << <blocksNumber, threadsNumber >> > (cu_winProbabilities, statesNumber, peopleNumber, iterationsNumber);

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

void Optimized()
{
	// Monte carlo init numbers
	const int iterationsNumber = ITERATIONS_NUM;
	const int peopleNumber = PEOPLE_NUM;
	const int statesNumber = CountSize(peopleNumber);
	double* winProbabilities = new double[statesNumber];
	// Threads and blocks
	const int threadsNumber = OPT_THREAD_NUMBER;
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
	blocksNumber = statesNumber;
	MonteCarloOpt << <blocksNumber, threadsNumber, threadsNumber * sizeof(double) >> > (cu_winProbabilities, statesNumber, peopleNumber, iterationsNumber);

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

