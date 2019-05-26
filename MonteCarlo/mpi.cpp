#include "mpi.h"

double* mains(int rank, int proccount, int* outSize, int* outProcSize);

int main(int argc, char* argv[])
{
	int myrank, proccount;
	int size, procSize;
	double* winProbabilities;

	MPI_Init(&argc, &argv);
	// find out my rank
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	// find out the number of processes in MPI_COMM_WORLD
	MPI_Comm_size(MPI_COMM_WORLD, &proccount);

	winProbabilities = mains(myrank,proccount,&size,&procSize);
    double* allWinProbabilities;
    if (myrank == 0)
        allWinProbabilities = new double[(size/proccount + 1) * proccount];
    MPI_Gather(winProbabilities, procSize, MPI_DOUBLE, allWinProbabilities, (size/proccount + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	delete winProbabilities;

	MPI_Finalize();
	return 0;
}
