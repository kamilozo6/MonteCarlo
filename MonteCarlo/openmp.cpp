#include <random>
#include <time.h>
#include <omp.h>

#define ITERATIONS_NUM 10000

unsigned int PEOPLE_NUM; //100
unsigned int THREAD_NUMBER; //256
unsigned int OPT_THREAD_NUMBER; //256

void ChangeState(int result, int& yes, int& no, int& unknown)
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

int EvaluateCase(int yes, int no, int unknown, int n)
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

void GetYesNoFromIndex(int index, int n, int& yes, int& no)
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
int ReturnIndex(int yes, int no, int n)
{
    int index = 0;

    for (int i = 1; i <= yes; i++)
    {
        index += n + 2 - i;
    }

    index += no;
    return index;
}

void MonteCarlo(double* winProbabilities, int states, int peoples, int iterationsNum, int rank, int allStates)
{
    omp_set_dynamic(0);
    omp_set_num_threads(THREAD_NUMBER);

#pragma omp parallel
    {
        double* threadResults = new double[states]();
        if (omp_get_thread_num() == 0)
        {
            printf("%d\n", omp_get_num_threads());
        }
#pragma omp for collapse(2)
        for (int state = rank * states; state < (rank + 1) * states; state++)
        {
            //printf("thread: %d - %d\n", omp_get_thread_num(), state);
            for (int i = 0; i < iterationsNum; i++)
            {
                if (state >= allStates) {
                    continue;
                }
                bool end = false;
                bool isYesResult = false;
                // Begining state
                int yes = peoples / 2, no = peoples / 2, unknown = peoples - yes - no;
                // Get yes, no numbers according to state
                GetYesNoFromIndex(state, peoples, yes, no);
                unknown = peoples - yes - no;
                while (!end)
                {
                    int result = EvaluateCase(yes, no, unknown, peoples);
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
                    threadResults[state % states]++;
                }
            }
        }

#pragma omp critical
        for (int i = 0; i < states; i++)
        {
            winProbabilities[i] += threadResults[i];
        }
#pragma omp barrier
#pragma omp single
        for (int i = 0; i < states; i++)
        {
            winProbabilities[i] /= iterationsNum;
        }
    }
}

double* mains(int rank, int proccount, int* outSize, int* outProcSize, unsigned int peopleNum, unsigned int threadNum)
{
    srand(time(NULL));
    int size = CountSize(peopleNum);
    printf("size: %d\n", size);
    int sizePerProc = size / proccount;
    int procSize;

    THREAD_NUMBER = threadNum;

    procSize = sizePerProc + 1;
    *outProcSize = procSize;
    *outSize = size;

    double* results = new double[procSize]();

    MonteCarlo(results, procSize, peopleNum, ITERATIONS_NUM, rank, size);

    return results;
}

int main(int argc, char* argv[])
{
    int size, procSize;
    unsigned int peopleNum = 10, threadNum = 8;
    double *results = mains(0, 1, &size, &procSize, peopleNum, threadNum);

    for (int i = 0; i < size; i++)
    {
        int yes, no;
        GetYesNoFromIndex(i, peopleNum, yes, no);
        printf("%d, %d - %f\n", yes, no, results[i]);
    }

    return 0;
}
