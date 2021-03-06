#include <stdio.h>
#include <stdlib.h>

void initialData(float *ip, int size)
{
    // generate different seed for random number
    
    for (int i = 0; i < size; i++)
    {
        ip[i] = i;
    }

    return;
}

void bubbleArraysOnHost(float *A, const int N)
{
float swap;
for (int c = 0 ; c < ( N - 1 ); c++)
  {
    for (int d = 0 ; d < N - c - 1; d++)
    {
      if (A[d] > A[d+1]) /* For decreasing order use < */
      {
        swap       = A[d];
        A[d]   = A[d+1];
        A[d+1] = swap;
      }
    }
  }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up data size of vectors
    int nElem = 1 << 16;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    //  initialData(h_A, nElem);
    bubbleArraysOnHost(h_A, nElem);
 
    // free host memory
    free(h_A);
  
    return(0);
}
