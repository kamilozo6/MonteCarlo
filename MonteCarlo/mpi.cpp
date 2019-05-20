#include "mpi.h"

int mains();

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	mains();

	MPI_Finalize();
	return 0;
}
