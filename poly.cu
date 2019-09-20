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

__global__ void poli1(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = poli[idx];
    
    if (idx < N) {
        
        poli[idx] = 3 * x * x - 7 * x + 5;
    }
}

__global__ void poli2(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = poli[idx];

    if (idx < N)
        poli[idx] = 4 * x * x * x + 3 * x * x - 7 * x + 5;
}

__global__ void poli3(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = poli[idx];

    if (idx < N)
        poli[idx] = 5 + 5 * x + 5 * x * x + 5 * x * x * x + 5 * x * x * x * x + 5 * x * x * x * x * x;
}

__global__ void poli4(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = poli[idx];

    if (idx < N)
        poli[idx] = 5 + 5 * x + 5 * x * sqrt(x) + 5 * sqrt(x) * x * x + 5 * x *
            sqrt(x) * x * x + 5 * x * sqrt(x) * sqrt(x) * x * x;
}

int main() {
    int nElem = 1 << 27;

    size_t nBytes = nElem * sizeof(float);

    float* h_polinomy = (float*)malloc(nBytes);

    float* d_polinomy;
    float* d_results;
    cudaMalloc((float**)&d_polinomy, nBytes);
    cudaMalloc((float**)&d_results, nBytes);
  
    int iLen = 512;
    dim3 block (iLen);
    dim3 grid  ((nElem + block.x - 1) / block.x);

    poli1<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyDeviceToHost);
    
    poli2<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyDeviceToHost);

    poli3<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyDeviceToHost);

    poli4<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_polinomy);
    free(h_polinomy);
}
