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


__global__ void poli_warp(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float x; 
    if (idx < N) {
       x = poli[idx];
       poli[idx] = 5 + x * ( 7 - x * (9 + x * (5 + x * (5 + x))))- 1.0f/x + 3.0f/(x*x) + x/5.0f;                 
    }
    poli[idx] = x;
}

int main() {
    int nElem = 1 << POWER;
    float elapsed_time;

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);


    size_t nBytes = nElem * sizeof(float);

    float* h_polinomy = (float*)malloc(nBytes);

    float* d_polinomy;
    float* d_results;
    cudaMalloc((float**)&d_polinomy, nBytes);
    cudaMalloc((float**)&d_results, nBytes);
  
    int iLen = 512;
    dim3 block (iLen);
    dim3 grid  ((nElem + block.x - 1) / block.x);
    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyHostToDevice);
    poli_warp<<<grid, block>>>(d_polinomy, nElem);
    poli_warp<<<grid, block>>>(d_polinomy, nElem);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event
    cudaEventRecord(start, 0); 
    poli_without_divergence<<<grid, block>>>(d_polinomy, nElem);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // calculate elapsed time
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Without divergence - Measured time for parallel execution = %.6fms\n",
           elapsed_time );
 
    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyHostToDevice);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event
    cudaEventRecord(start, 0); 
    poli_div<<<grid, block>>>(d_polinomy, nElem);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // calculate elapsed time
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("With Divergence - Measured time for parallel execution = %.6fms\n",
           elapsed_time );

    cudaMemcpy(d_polinomy, h_polinomy, nBytes, cudaMemcpyDeviceToHost);
    
    
    cudaFree(d_polinomy);
    free(h_polinomy);
}
