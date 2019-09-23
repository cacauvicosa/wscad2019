#include <cuda_runtime.h>
#include <stdio.h>

/*
 * 
 */

__global__ void poly_div1(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
          
    if (idx < N) {
        float x = poli[idx];
        poli[idx] = 5 + x * ( 7 - x * (9 + x * (5 + x * (5 + x))))+x/5.0;
    }
}

__global__ void poly_div2(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
          
    if (idx < N) {
        float x = poli[idx];
        poli[idx] = 5 + x * ( 7 - x * (9 + x * (5 + x * (5 + x))))+x*0.2;
    }
}

__global__ void poly_div3(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
          
    if (idx < N) {
        float x = poli[idx];
        poli[idx] = 5 + x * ( 7 - x * (9 + x * (5 + x * (5 + x))))+5.0/x;
    }
}

__global__ void poly_div4(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
          
    if (idx < N) {
        float x = poli[idx];
        float y = 5.0/x;
        poli[idx] = 5 + x * ( 7 - x * (9 + x * (5 + x * (5 + x))))+y;
    }
}

__global__ void poly_div5(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
          
    if (idx < N) {
        float x = poli[idx];
        poli[idx] = 5 + x * ( 7 - x * (9 + x * (5 + x * (5 + x))))+1.0/x;
    }
}

__global__ void poly_div6(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
          
    if (idx < N) {
        float x = poli[idx];
        float y = 1.0/x;
        poli[idx] = 5 + x * ( 7 - x * (9 + x * (5 + x * (5 + x))))+y;
    }
}


__global__ void poly_div7(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
          
    if (idx < N) {
        float x = poli[idx];
        poli[idx] = 5 + x * ( 7 - x * (9 + x * (5 + x * (5 + x))))+1.0f/x;
    }
}

__global__ void poly_div8(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
          
    if (idx < N) {
        float x = poli[idx];
        poli[idx] = 5 + x * ( 7 - x * (9 + x * (5 + x * (5 + x))))+5.0f/x;
    }
}

int main() {
    int nElem = 1 << 27;
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


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event
    cudaEventRecord(start, 0); 
    poly_div1<<<grid, block>>>(d_polinomy, nElem);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // calculate elapsed time
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Measured time for parallel execution = %.6fms\n",
           elapsed_time );

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event
    cudaEventRecord(start, 0); 
    poly_div2<<<grid, block>>>(d_polinomy, nElem);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // calculate elapsed time
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Measured time for parallel execution = %.6fms\n",
           elapsed_time );
    
cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event
    cudaEventRecord(start, 0); 
    poly_div3<<<grid, block>>>(d_polinomy, nElem);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // calculate elapsed time
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Measured time for parallel execution = %.6fms\n",
           elapsed_time );

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event
    cudaEventRecord(start, 0); 
    poly_div4<<<grid, block>>>(d_polinomy, nElem);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // calculate elapsed time
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Measured time for parallel execution = %.6fms\n",
           elapsed_time );
     cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event
    cudaEventRecord(start, 0); 
    poly_div5<<<grid, block>>>(d_polinomy, nElem);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // calculate elapsed time
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Measured time for parallel execution = %.6fms\n",
           elapsed_time );

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event
    cudaEventRecord(start, 0); 
    poly_div6<<<grid, block>>>(d_polinomy, nElem);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // calculate elapsed time
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Measured time for parallel execution = %.6fms\n",
           elapsed_time );
    
cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event
    cudaEventRecord(start, 0); 
    poly_div7<<<grid, block>>>(d_polinomy, nElem);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // calculate elapsed time
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Measured time for parallel execution = %.6fms\n",
           elapsed_time );

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event
    cudaEventRecord(start, 0); 
    poly_div8<<<grid, block>>>(d_polinomy, nElem);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // calculate elapsed time
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Measured time for parallel execution = %.6fms\n",
           elapsed_time );
     
    

    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_polinomy);
    free(h_polinomy);
}
