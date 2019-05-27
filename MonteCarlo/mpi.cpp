#include "mpi.h"
#include <chrono>
#include <iostream>
using namespace std::chrono;
//#define PRINT_RESULT

unsigned int PEOPLE_NUM_MPI; //100
unsigned int THREAD_NUMBER_MPI; //256

double* mains(int rank, int proccount, int* outSize, int* outProcSize, unsigned int peopleNum, unsigned int threadNum);

int main(int argc, char* argv[])
{
	int myrank, proccount;
	int size, procSize;
	double* winProbabilities;

    PEOPLE_NUM_MPI = 5;
    THREAD_NUMBER_MPI = 5;
    
    auto start = high_resolution_clock::now();
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);

	MPI_Init(&argc, &argv);
	// find out my rank
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	// find out the number of processes in MPI_COMM_WORLD
	MPI_Comm_size(MPI_COMM_WORLD, &proccount);
    std::cout << "PEPOPLE" << std::endl;
    THREAD_NUMBER_MPI = 256;
    for (PEOPLE_NUM_MPI = 5; PEOPLE_NUM_MPI < 150; PEOPLE_NUM_MPI++)
    {
        start = high_resolution_clock::now();
        
        winProbabilities = mains(myrank,proccount,&size,&procSize, PEOPLE_NUM_MPI, THREAD_NUMBER_MPI);
        double* allWinProbabilities;
        if (myrank == 0)
            allWinProbabilities = new double[(size/proccount + 1) * proccount];
        MPI_Gather(winProbabilities, procSize, MPI_DOUBLE, allWinProbabilities, (size/proccount + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        std::cout << PEOPLE_NUM_MPI << "," << THREAD_NUMBER_MPI << "," << duration.count() << std::endl;
    }
    std::cout << "THREAD_NUM" << std::endl;
    for (THREAD_NUMBER_MPI = 32; THREAD_NUMBER_MPI < 256; THREAD_NUMBER_MPI++)
    {
        start = high_resolution_clock::now();
        
        winProbabilities = mains(myrank,proccount,&size,&procSize, PEOPLE_NUM_MPI, THREAD_NUMBER_MPI);
        double* allWinProbabilities;
        if (myrank == 0)
            allWinProbabilities = new double[(size/proccount + 1) * proccount];
        MPI_Gather(winProbabilities, procSize, MPI_DOUBLE, allWinProbabilities, (size/proccount + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        std::cout << PEOPLE_NUM_MPI << "," << THREAD_NUMBER_MPI << "," << duration.count() << std::endl;
    }

#ifdef PRINT_RESULT
    int i;
    for (i = 0; i < (size/proccount + 1) * proccount; i++)
    {
        printf("%f\n", allWinProbabilities[i]);
    }
#endif

	delete winProbabilities;

	MPI_Finalize();
	return 0;
}
