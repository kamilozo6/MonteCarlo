#pragma once
#include <stdint.h>
#include <cmath>

class Vector
{
private:
    double * vectorTable;
    uint32_t vectorSize;
public:
    Vector() :
        vectorSize(0),
        vectorTable(NULL)
    {
    }


    ~Vector()
    {
        delete vectorTable;
    }


    __device__ void GetVectorSize(uint32_t& size)
    {
        size = vectorSize;
    }

    __device__ void GetVectorCell(uint32_t row, uint32_t& cell)
    {
        cell = vectorTable[row];
    }

    __device__ void SetVectorCell(uint32_t row, double value)
    {
        vectorTable[row] = value;
    }

    __device__ static Vector* Create(uint32_t size)
    {
        Vector *newVector = new Vector();

        newVector->vectorSize = size;
        newVector->vectorTable = new double[size];
        return newVector;
    }

    Vector& operator = (Vector*& input)
    {
        vectorSize = input->GetVectorSize();
        vectorTable = new double[vectorSize];
        for (uint32_t i = 0; i < vectorSize; i++)
        {
            vectorTable[i] = input->GetVectorCell(i);
        }
        return *this;
    }

    __device__ void ReturnIndex(uint32_t yes, uint32_t no, uint32_t n, uint32_t& index)
    {
        index = 0;

        for (uint32_t i = 1; i <= yes; i++)
        {
            index += n + 2 - i;
        }

        index += no;
    }

};