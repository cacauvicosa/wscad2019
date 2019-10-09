#include <iostream>
#include <cuda_runtime_api.h>
#include <chrono>

#define RAIO 10
#define TAMANHO 20

__constant__ float const_stencilWeight[10000];

// base case
__global__ void stencil(float *src, float *dst, int size, int raio, float *stencilWeight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += raio+1;
    if (idx >= size)
        return;
    float out = 0;
    #pragma unroll
    for(int i = -raio;i < raio; i++)
    {
        out += src[idx+i] * stencilWeight[i+raio];
    }
    dst[idx] = out;
}

// read only cache stencil coefficients
__global__ void stencilReadOnly1(float *src, float *dst, int size, int raio, float* stencilWeight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += raio+1;
    if (idx >= size)
        return;
    float out = 0;
    #pragma unroll
    for(int i = -raio;i < raio; i++){
        out += src[idx+i] * stencilWeight[i+raio];
    }
    dst[idx] = out;
}

// read only data
__global__ void stencilReadOnly2(float *src, float *dst, int size, int raio, float* stencilWeight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += raio+1;
    if (idx >= size)
        return;
    float out = 0;
    #pragma unroll
    for(int i = -raio;i < raio; i++)
    {
        out += src[idx+i] * stencilWeight[i+raio];
    }
    dst[idx] = out;
}

// read only coefficients and data
__global__ void stencilReadOnly3(float *src, float *dst, int size, int raio, float* stencilWeight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += raio+1;
    if (idx >= size)
        return;
    float out = 0;
    #pragma unroll
    for(int i = -raio;i < raio; i++)
    {
        out += src[idx+i] * stencilWeight[i+raio];
    }
    dst[idx] = out;
}

// constat memory coefficients
__global__ void stencilConst1(float *src, float *dst, int size, int raio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += raio+1;
    if (idx >= size)
        return;
    float out = 0;
    #pragma unroll
    for(int i = -raio;i < raio; i++)
    {
        out += src[idx+i] * const_stencilWeight[i+raio];
    }
    dst[idx] = out;
}

// constant memory coefficients and data through read only cache
__global__ void stencilConst2(float *src, float *dst, int size, int raio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += raio+1;
    if (idx >= size)
        return;
    float out = 0;
    #pragma unroll
    for(int i = -raio;i < raio; i++)
    {
        out += src[idx+i] * const_stencilWeight[i+raio];
    }
    dst[idx] = out;
}

// constant memory coefficients and data from shared 
__global__ void stencilShared1(float *src, float *dst, int size, int raio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float buffer[1024+11];
    for(int i = threadIdx.x; i < 1024+21; i = i + 1024)
    {
        buffer[i] = src[idx+i];
    }
    idx += raio+1;
    if (idx >= size)
        return;
    
    __syncthreads();
    float out = 0;
    #pragma unroll
    for(int i = -raio;i < raio; i++)
    {
        out += buffer[threadIdx.x+raio+i] * const_stencilWeight[i+raio];
    }
    dst[idx] = out;
}

// constant memory coefficients and data from shared thorugh read only
__global__ void stencilShared2(float *src, float *dst, int size, int raio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float buffer[1024+11];
    for(int i = threadIdx.x; i < 1024+21; i = i + 1024)
    {
        buffer[i] = src[idx+i];
    }
    idx += raio;
    if (idx >= size)
        return;
    
    __syncthreads();
    float out = 0;
    #pragma unroll
    for(int i = -raio;i < raio; i++)
    {
        out += buffer[threadIdx.x+raio+i] * const_stencilWeight[i+raio];
    }
    dst[idx] = out;
}

bool verify(float *arr, float *corr, int count, int raio){
    // skip the first elements since they may be wrong
    for(int i = raio+1; i < count; i++){
        if(arr[i] != corr[i]){   
            std::cout << "check failed" << i << " " << arr[i] << " != " << corr[i] << std::endl;
            exit(1);
        }
    }
    return true;
}

int main()
{

    int tamanho = 1 << TAMANHO; 

    float *a;
    float *b;
    float *bOut;
    float *bCorr;
    float *weights;
    cudaMalloc(&a, sizeof(float)*tamanho);
    cudaMalloc(&b, sizeof(float)*tamanho);
    cudaMallocHost(&bOut, sizeof(float)*tamanho);
    cudaMallocManaged(&bCorr, sizeof(float)*tamanho);
    cudaMallocManaged(&weights, sizeof(float)*(2*RAIO+1));

    cudaDeviceSynchronize();

    for(int i = 0; i < tamanho;i++){
        //a[i] = 0;
        //b[i] = 0;
        bCorr[i] = 0;
    }

    cudaMemset(a, 1, tamanho);
    cudaMemset(b, 1, tamanho);
    cudaMemset(bCorr, 1, tamanho);
    cudaMemset(bOut, 1, tamanho);

    cudaDeviceSynchronize();    
    
    int blockSize = 1024;
    int blocks = 10000;
    for(int i = 0; i < 2*RAIO+1; ++i)
        weights[i] = i-10;
    
    
    cudaDeviceSynchronize();    

    // copy to constant memory    
    cudaMemcpyToSymbol(const_stencilWeight, weights, sizeof(float)*(2*RAIO+1));
    //cudaMemcpy(const_stencilWeight, weights, sizeof(float)*(2*RAIO+1), cudaMemcpyDefault);

    // run the basic case once to get the "correct" results
    stencil<<<blocks, blockSize>>>(a, bCorr, tamanho, RAIO, weights);
    cudaDeviceSynchronize();    

    stencil<<<blocks, blockSize>>>(a, b, tamanho, RAIO, weights);
    cudaDeviceSynchronize(); 
    cudaMemcpy(bOut, b, sizeof(float)*tamanho, cudaMemcpyDefault);
    verify(bOut, bCorr, 1000, RAIO);

    cudaSetDevice(0); 

    float minTime = 10000;
    for(int i  = 0; i < 10; i++){
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        stencil<<<blocks, blockSize>>>(a, b, tamanho, RAIO, weights);
        cudaDeviceSynchronize();    
        end = std::chrono::system_clock::now();
        
        cudaMemcpy(bOut, b, sizeof(float)*tamanho, cudaMemcpyDefault);
        verify(bOut, bCorr, 1000, RAIO);  

	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "Non optimized " << (blockSize*blocks)/minTime << " updates/s" << std::endl;
    minTime = 10000;
    std::cout << std::endl;

    for(int i  = 0; i < 10; i++)
    {
        cudaDeviceSynchronize();  
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        stencilReadOnly1<<<blocks, blockSize>>>(a, b, tamanho, RAIO, weights);
        cudaDeviceSynchronize();  
        end = std::chrono::system_clock::now();
        
        cudaMemcpy(bOut, b, sizeof(float)*tamanho, cudaMemcpyDefault);
        verify(bOut, bCorr, 1000, RAIO);  
        
	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "read only cache stencil coefficients " <<(blockSize*blocks)/minTime << " updates/s" << std::endl;
    minTime = 10000;
    for(int i  = 0; i < 10; i++)
    {
        cudaDeviceSynchronize();  
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        stencilReadOnly2<<<blocks, blockSize>>>(a, b, tamanho, RAIO, weights);
        cudaDeviceSynchronize();  
        end = std::chrono::system_clock::now();
        
        (cudaMemcpy(bOut, b, sizeof(float)*tamanho, cudaMemcpyDefault));
        verify(bOut, bCorr, 1000, RAIO);  
        
	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "read only data " << (blockSize*blocks)/minTime << " updates/s" << std::endl;
    minTime = 10000;
    for(int i  = 0; i < 10; i++)
    {
        cudaDeviceSynchronize();  
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        stencilReadOnly3<<<blocks, blockSize>>>(a, b, tamanho, RAIO, weights);
        cudaDeviceSynchronize();  
        end = std::chrono::system_clock::now();
        
        cudaMemcpy(bOut, b, sizeof(float)*tamanho, cudaMemcpyDefault);
        verify(bOut, bCorr, 1000, RAIO);  
        
	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "read only coefficients and data " << (blockSize*blocks)/minTime << " updates/s" << std::endl;
    minTime = 10000;

    std::cout << std::endl;

        for(int i  = 0; i < 10; i++)
    {
        cudaDeviceSynchronize();  
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        stencilConst1<<<blocks, blockSize>>>(a, b, tamanho, RAIO);
        cudaDeviceSynchronize();    
        end = std::chrono::system_clock::now();

        (cudaMemcpy(bOut, b, sizeof(float)*tamanho, cudaMemcpyDefault));
        verify(bOut, bCorr, 1000, RAIO);  
	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "constant memory coefficients " << (blockSize*blocks)/minTime << " updates/s" << std::endl;

    minTime = 10000;

    for(int i  = 0; i < 10; i++){
        cudaDeviceSynchronize();  
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        stencilConst2<<<blocks, blockSize>>>(a, b, tamanho, RAIO);
        cudaDeviceSynchronize();    
        end = std::chrono::system_clock::now();

        (cudaMemcpy(bOut, b, sizeof(float)*tamanho, cudaMemcpyDefault));
        verify(bOut, bCorr, 1000, RAIO);  
	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "constant memory coefficients and data through read only cache " << (blockSize*blocks)/minTime << " updates/s" << std::endl;
    std::cout << std::endl;


    minTime = 10000;
            for(int i  = 0; i < 10; i++)
    {
        cudaDeviceSynchronize();  
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        stencilShared1<<<blocks, blockSize>>>(a, b, tamanho, RAIO);
        cudaDeviceSynchronize();    
        end = std::chrono::system_clock::now();
        
        (cudaMemcpy(bOut, b, sizeof(float)*tamanho, cudaMemcpyDefault));
        verify(bOut, bCorr, 1000, RAIO);  
	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "constant memory coefficients and data from shared " << (blockSize*blocks)/minTime << " updates/s" << std::endl;
    minTime = 10000;
    minTime = 10000;
    for(int i  = 0; i < 10; i++)
    {
        cudaDeviceSynchronize();  
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        stencilShared2<<<blocks, blockSize>>>(a, b, tamanho, RAIO);
        cudaDeviceSynchronize();    
        end = std::chrono::system_clock::now();
        
        cudaMemcpy(bOut, b, sizeof(float)*tamanho, cudaMemcpyDefault);
        verify(bOut, bCorr, 1000, RAIO);  
	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "constant memory coefficients and data from shared thorugh read only " << (blockSize*blocks)/minTime << " updates/s" << std::endl;
    minTime = 10000;

}
