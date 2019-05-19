
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vector.cuh"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__device__ void ChangeState(int result, int & yes, int & no, int & unknown)
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

__device__ int CountSize(int n)
{
	int size = 0;
	for (int i = 1; i <= n + 1; i++)
	{
		size += i;
	}
	return size;
}

__device__ int EvaluateCase(int yes, int no, int unknown, int n)
{
	int randomValue = rand() % n + 1;
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
	randomValue = rand() % n + 1;
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

__global__ MyMatrix* MonteCarlo(int n)
{
	int iterationNumber = 100000, size = CountSize(n);
	MyMatrix *X = Vector::Create(size);
	MyMatrix *numerator = Vector::Create(size);
	MyMatrix *denominator = Vector::Create(size);
	bool *casesUsed = new bool[size];
	for (int i = 0; i < iterationNumber; i++)
	{
		bool end = false;
		bool isYesResult = false;
		int test = rand() % size;
		int yes = n / 2, no = n / 2, unknown = n - yes - no;
        Vector::GetYesNoFromIndex(test, n, yes, no);
		unknown = n - yes - no;
		for (int j = 0; j < size; j++)
		{
			casesUsed[j] = false;
		}
		casesUsed[Vector::ReturnIndex(yes, no, n)] = true;
		while (!end)
		{
			int result = EvaluateCase(yes, no, unknown, n);
			ChangeState(result, yes, no, unknown);

			int index = Vector::ReturnIndex(yes, no, n);
			casesUsed[index] = true;

			if (yes == 0 || no == 0)
			{
				if (yes > 0)
				{
					isYesResult = true;
				}
				end = true;
			}
		}
		for (int j = 0; j < size; j++)
		{
			if (casesUsed[j])
			{
				if (isYesResult)
				{
					numerator->matrix[j][0]++;
				}
				denominator->matrix[j][0]++;
			}
		}
	}
	delete casesUsed;
	for (int i = 0; i < size; i++)
	{
		if (denominator->matrix[i][0] != 0)
		{
			X->matrix[i][0] = numerator->matrix[i][0] / denominator->matrix[i][0];
		}
	}

	delete numerator;
	delete denominator;
	return X;
}


int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
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
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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
