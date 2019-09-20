#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
 * GPU. Only a single thread block is used in this small case, for simplicity.
 * sumArraysOnHost sequentially iterates through vector elements on the host.
 * This version of sumArrays adds host timers to measure GPU and CPU
 * performance.
 */

void checkResult(double *hostRef, double *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                   gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");

    return;
}

void initialData(double *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (double)( rand() & 0xFF ) / 10.0f;
    }

    return;
}

void sumc(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}


__global__ void sum(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (i < N) C[i] = A[i] + B[i];
}


__global__ void sum4(float4 *A, float4 *B, float4 *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) 
    { C[i].x = A[i].x + B[i].x;C[i].y = A[i].y + B[i].y;C[i].z = A[i].z + B[i].z;C[i].w = A[i].w + B[i].w;}
}



int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem = 1 << 23;
    printf("Vector size %d\n", nElem*4);

    // malloc host memory
    size_t nBytes4 = nElem * sizeof(float4);

    float4 *h_A4, *h_B4, *hostRef4, *gpuRef4;
    h_A4     = (float4 *)malloc(nBytes4);
    h_B4     = (float4 *)malloc(nBytes4);
    hostRef4 = (float4 *)malloc(nBytes4);
    gpuRef4  = (float4 *)malloc(nBytes4);
 
    memset(hostRef4, 0, nBytes4);
    memset(gpuRef4,  0, nBytes4);


    // malloc device global memory
    float4 *d_A4, *d_B4, *d_C4;
    CHECK(cudaMalloc((float4**)&d_A4, nBytes4));
    CHECK(cudaMalloc((float4**)&d_B4, nBytes4));
    CHECK(cudaMalloc((float4**)&d_C4, nBytes4));


    // transfer data from host to device
    // transfer data from host to device
    CHECK(cudaMemcpy(d_A4, h_A4, nBytes4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B4, h_B4, nBytes4, cudaMemcpyHostToDevice));
    // invoke kernel at host side
    int iLen = 512;
    dim3 block (iLen);
    dim3 grid  (((nElem + block.x - 1) / block.x));

    sum4<<<grid, block>>>(d_A4, d_B4, d_C4, nElem);

    CHECK(cudaDeviceSynchronize());
    printf("sumArraysOnGPU <<<  %d, %d  >>>  \n", grid.x,
           block.x);

    // check kernel error
    CHECK(cudaGetLastError()) ;

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef4, d_C4, nBytes4, cudaMemcpyDeviceToHost));

    return(0);
}
