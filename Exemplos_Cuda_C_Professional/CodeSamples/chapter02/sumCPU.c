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

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up data size of vectors
    int nElem = 1 << 24;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
     initialData(h_A, nElem);
    initialData(h_B, nElem);
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
 
    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
  
    return(0);
}
