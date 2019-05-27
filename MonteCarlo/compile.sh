rm kernel.o
rm main.o
rm a.out

mpicxx -std=c++11 -c mpi.cpp -o main.o
nvcc -c kernel.cu -o kernel.o -std=c++11
mpicxx -std=c++11 main.o kernel.o -lcudart -L/usr/local/cuda/lib64