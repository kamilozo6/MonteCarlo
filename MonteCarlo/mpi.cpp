#include "mpi.h"
#include <chrono>
using namespace std::chrono;
//#define PRINT_RESULT

unsigned int PEOPLE_NUM; //100
unsigned int THREAD_NUMBER; //256

double* mains(int rank, int proccount, int* outSize, int* outProcSize);

int main(int argc, char* argv[])
{
	int myrank, proccount;
	int size, procSize;
	double* winProbabilities;

    PEOPLE_NUM = 5;
    THREAD_NUMBER = 5;
    OPT_THREAD_NUMBER = 5;

	MPI_Init(&argc, &argv);
	// find out my rank
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	// find out the number of processes in MPI_COMM_WORLD
	MPI_Comm_size(MPI_COMM_WORLD, &proccount);

    for (PEOPLE_NUM = 5; PEOPLE_NUM < 7; PEOPLE_NUM++)
    {
        start = high_resolution_clock::now();
        
        winProbabilities = mains(myrank,proccount,&size,&procSize);
        double* allWinProbabilities;
        if (myrank == 0)
            allWinProbabilities = new double[(size/proccount + 1) * proccount];
        MPI_Gather(winProbabilities, procSize, MPI_DOUBLE, allWinProbabilities, (size/proccount + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        std::cout << PEOPLE_NUM << "," << THREAD_NUMBER << "," << duration.count() << std::endl;
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
