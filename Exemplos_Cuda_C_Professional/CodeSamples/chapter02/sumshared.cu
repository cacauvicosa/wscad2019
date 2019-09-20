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

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    float epsilon = 1.0E-8;
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

void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
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


__global__ void sum4(float *A, float *B, float *C, const int N)
{
    int j;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
#pragma unroll
    for (j=0; j < 4; j++) 
     if  (i < N) {
      C[i] = A[i] + B[i];
      i += blockDim.x * gridDim.x;
     }
}



__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if  (i < N) {
      C[i] = A[i] + B[i];
    }
}

__global__ void sum10ops(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) C[i] = A[i] + B[i] - A[i]*A[i] + 3*B[i] - 4*A[i]*B[i] + B[i]*B[i]*7- 8;
}



__global__ void shared1R1W1G(float *A, float *B, float *C, const int N)
{
 __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
      Smem[threadIdx.x] = i;
      C[i] = Smem[(threadIdx.x+1)%512];
    }
}

__global__ void shared4R1W1G(float *A, float *B, float *C, const int N)
{
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
      Smem[threadIdx.x] = i;
      C[i] = Smem[(threadIdx.x+1)%512]+Smem[(threadIdx.x+2)%512]+Smem[(threadIdx.x+3)%512]+Smem[(threadIdx.x+4)%512];
    }
}


__global__ void shared4R1W1Gs(float *A, float *B, float *C, const int N)
{
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    Smem[threadIdx.x] = i;
    __syncthreads();

    if (i < N) {
       C[i] = Smem[(threadIdx.x+1)%512]+Smem[(threadIdx.x+2)%512]+Smem[(threadIdx.x+3)%512]+Smem[(threadIdx.x+4)%512];
    }
}




__global__ void shared4R1Ws10ops2RG1WG(float *A, float *B, float *C, const int N)
{
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    	Smem[threadIdx.x] = A[i];
    __syncthreads();

    if (i < N) {
       C[i] = A[i] + B[i] - A[i]*A[i] + 3*B[i] - 4*A[i]*B[i] + B[i]*B[i]*7- 8+Smem[(threadIdx.x+1)%512]+Smem[(threadIdx.x+2)%512]+Smem[(threadIdx.x+3)%512]+Smem[(threadIdx.x+4)%512];
    }
}


__global__ void shared4R20ops(float *A, float *B, float *C, const int N)
{
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    	Smem[threadIdx.x] = A[i];
    __syncthreads();

    float x;
    if (i < N) {
       x = A[i]/3 + 17*B[i] - A[i]*A[i] + 3*B[i] - 4*A[i]*B[i] + B[i]*B[i]*7;
       C[i] = x- 8 +Smem[(threadIdx.x+1)%512]*A[i] + 4*Smem[(threadIdx.x+2)%512]+3*B[i]*Smem[(threadIdx.x+3)%512]+A[i]*Smem[(threadIdx.x+4)%512];
    }
}


__global__ void shared4R24ops(float *A, float *B, float *C, const int N)
{
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    	Smem[threadIdx.x] = A[i];
    __syncthreads();

    float x;
    if (i < N) {
       x = (A[i]/3 + 17*B[i] - A[i]*A[i] + 3*B[i] - 4*A[i]*B[i] + B[i]*B[i]*7) / (A[i]/9 + 13*B[i]) ;
       C[i] = x- 8 +Smem[(threadIdx.x+1)%512]*A[i] + 4*Smem[(threadIdx.x+2)%512]+3*B[i]*Smem[(threadIdx.x+3)%512]+A[i]*Smem[(threadIdx.x+4)%512];
    }
}


__global__ void shared4R40ops(float *A, float *B, float *C, const int N)
{
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    	Smem[threadIdx.x] = A[i];
    __syncthreads();

    float x;
    if (i < N) {
       x = tan(0.2) * tan(0.3) + B[i]/4.0 + A[i]*B[i]/3.0 + tan(0.1)*A[i] + tan(0.5)*B[i];
       x += A[i]/3 + 17*B[i] - A[i]*A[i] + 3*B[i] - 4*A[i]*B[i] + B[i]*B[i]*7;
       C[i] = x- 8 +Smem[(threadIdx.x+1)%512]*A[i] + 4*Smem[(threadIdx.x+2)%512]+3*B[i]*Smem[(threadIdx.x+3)%512]+A[i]*Smem[(threadIdx.x+4)%512];
    }
}



__global__ void shared4R25ops(float *A, float *B, float *C, const int N)
{
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    	Smem[threadIdx.x] = A[i];
    __syncthreads();

    float x;
    if (i < N) {
       x = tan(0.2) *B[i];
       x += A[i]/3 + 17*B[i] - A[i]*A[i] + 3*B[i] - 4*A[i]*B[i] + B[i]*B[i]*7;
       C[i] = x- 8 +Smem[(threadIdx.x+1)%512]*A[i] + 4*Smem[(threadIdx.x+2)%512]+3*B[i]*Smem[(threadIdx.x+3)%512]+A[i]*Smem[(threadIdx.x+4)%512];
    }
}


__global__ void shared4RNops(float *A, float *B, float *C, const int N)
{
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    	Smem[threadIdx.x] = A[i];
    __syncthreads();

    float x;
    if (i < N) {
       x = A[i]/4 + 15*B[i] - A[i]/7.0*A[i] + 4*B[i] + 7*A[i]*A[i] + A[i]*B[i]*7;
       x += A[i]/3 + 17*B[i] - A[i]*A[i] + 3*B[i] - 4*A[i]*x + x*B[i]*7;
       C[i] = x- 8 +Smem[(threadIdx.x+1)%512]*A[i] + 4*Smem[(threadIdx.x+2)%512]+3*B[i]*Smem[(threadIdx.x+3)%512]+A[i]*Smem[(threadIdx.x+4)%512];
    }
}



__global__ void shared4RMops(float *A, float *B, float *C, const int N)
{
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    	Smem[threadIdx.x] = A[i];
    __syncthreads();

    float x;
    if (i < N) {
       x = A[i]/7.0*A[i];
       x += A[i]/3 + 17*B[i] + 3*B[i] - A[i]*x + x*B[i]*7;
       C[i] = x- 8 +Smem[(threadIdx.x+1)%512]*A[i] + 4*Smem[(threadIdx.x+2)%512]+3*B[i]*Smem[(threadIdx.x+3)%512]+A[i]*Smem[(threadIdx.x+4)%512];
    }
}




__global__ void sum4M(float *A, float *B, float *C, const int N)
{
    int j;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x[4];
 
#pragma unroll
    for (j=0; j < 4; j++) 
     if  (i < N) {
       x[j] = A[i]/7.0*A[i];
       C[i] += A[i]/3 + 17*B[i] + 3*B[i] - A[i]*x[j] + x[j]*B[i]*7;
       i += blockDim.x * gridDim.x;
     }
}



__global__ void sum4K(float *A, float *B, float *C, const int N)
{
    int j;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x[4];
 
#pragma unroll
    for (j=0; j < 4; j++) 
     if  (i < N) {
       x[j] = A[i]*A[i];
       C[i] += A[i]*3 + 17*B[i] + 3*B[i] - A[i]*x[j] + x[j]*B[i]*7;
       i += blockDim.x * gridDim.x;
     }
}

__global__ void sum4Man(float *A, float *B, float *C, const int N)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x[4],a[4],b[4],c[4];
 
    a[0] = A[i];
    b[0] = B[i];
    x[0] = a[0]/7.0;
    c[0] = a[0]/3 + 17*b[0] + 3*b[0]; 
i += blockDim.x * gridDim.x;
    a[1] = A[i];
    b[1] = B[i];
    x[0]*= a[0];
    x[1] = a[1]/7.0;
c[1] = a[1]/3 + 17*b[1] + 3*b[1];
    x[0]= a[0]*x[0] + x[0]*b[0]*7;
i += blockDim.x * gridDim.x;
    a[2] = A[i];
    b[2] = B[i];
    x[1]*= a[1];
    x[2] = a[2]/7.0;
c[2] = a[2]/3 + 17*b[2] + 3*b[2];
x[1]= a[1]*x[1] + x[1]*b[1]*7;
i += blockDim.x * gridDim.x;
     if  (i < N) {
    a[3] = A[i];
    b[3] = B[i];
    }
    x[2]*= a[2];
    x[3] = a[3]/7.0;
c[3] = a[3]/3 + 17*b[3] + 3*b[3];
x[2]= a[2]*x[2] + x[2]*b[2]*7;
x[3]*= a[3];
x[3]= a[3]*x[3] + x[3]*b[3]*7;

     

i = blockIdx.x * blockDim.x + threadIdx.x;
       C[i] += c[0]- x[0];
       i += blockDim.x * gridDim.x;
       C[i] += c[1]- x[1];
       i += blockDim.x * gridDim.x;
       C[i] += c[2]- x[2];
       i += blockDim.x * gridDim.x;
      if  (i < N) C[i] += c[3]- x[3];
}






__global__ void shared4R15ops(float *A, float *B, float *C, const int N)
{
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    	Smem[threadIdx.x] = A[i];
    __syncthreads();

    float x;
    if (i < N) {
       x = tan(0.2) *B[i];
       x += A[i]/3 + 17*B[i];
       C[i] = x- 8 +Smem[(threadIdx.x+1)%512]*A[i] + 4*Smem[(threadIdx.x+2)%512]+3*B[i]*Smem[(threadIdx.x+3)%512]+A[i]*Smem[(threadIdx.x+4)%512];
    }
}

__global__ void shared2R1W1G(float *A, float *B, float *C, const int N)
{
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
      Smem[threadIdx.x] = i;
      C[i] = Smem[(threadIdx.x+1)%512]+Smem[(threadIdx.x+5)%512];
    }
}


__global__ void shared1RC1W1G(float *A, float *B, float *C, const int N)
{
// compilador é esperto e aproveita o valor de i, mas faz 1W, 2 R nas outras posições da Shared
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
      Smem[(threadIdx.x+1)%512] = i;
      C[i] = Smem[(threadIdx.x*2)%512];
    }
}


__global__ void shared1R8C1W1G(float *A, float *B, float *C, const int N)
{
// compilador é esperto e aproveita o valor de i, mas faz 1W, 2 R nas outras posições da Shared
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
      Smem[(threadIdx.x+1)%512] = i;
      C[i] = Smem[(threadIdx.x*8)%512];
    }
    /*if ( blockIdx.x ==  2 && threadIdx.x < 32 ) {
       printf("th %d smem %d\n",threadIdx.x,(threadIdx.x*8)%512);
    }*/
}


__global__ void shared1R8C1W1G1RG(float *A, float *B, float *C, const int N)
{
// compilador é esperto e aproveita o valor de i, mas faz 1W, 2 R nas outras posições da Shared
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
      Smem[(threadIdx.x+1)%512] = A[i];
      C[i] = Smem[(threadIdx.x*8)%512];
    }
    /*if ( blockIdx.x ==  2 && threadIdx.x < 32 ) {
       printf("th %d smem %d\n",threadIdx.x,(threadIdx.x*8)%512);
    }*/
}

__global__ void shared1R8C1W8C1G(float *A, float *B, float *C, const int N)
{
// compilador é esperto e aproveita o valor de i, mas faz 1W, 2 R nas outras posições da Shared
    __shared__ float Smem[512];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
      Smem[((threadIdx.x+1)*8)%512] = i;
      C[i] = Smem[(threadIdx.x*8)%512];
    }
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
    int nElem = 1 << 25;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    float iStart, iElaps;

    // initialize data at host side
    iStart = seconds();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = seconds() - iStart;
    printf("initialData Time elapsed %f sec\n", iElaps);
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    // add vector at host side for result checks
    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
  
    iElaps = seconds() - iStart;
    printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int iLen = 512;
    dim3 block (iLen);
    dim3 grid  ((nElem + block.x - 1) / block.x);

    iStart = seconds();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    sum10ops<<<grid, block>>>(d_A, d_B, d_C, nElem);

    shared1R1W1G<<<grid, block>>>(d_A, d_B, d_C, nElem);
    shared2R1W1G<<<grid, block>>>(d_A, d_B, d_C, nElem);
    shared1RC1W1G<<<grid, block>>>(d_A, d_B, d_C, nElem);
    shared1R8C1W1G<<<grid, block>>>(d_A, d_B, d_C, nElem);
    shared1R8C1W1G1RG<<<grid, block>>>(d_A, d_B, d_C, nElem);
    shared1R8C1W8C1G<<<grid, block>>>(d_A, d_B, d_C, nElem);
    shared4R1W1G<<<grid, block>>>(d_A, d_B, d_C, nElem);
    shared4R1W1Gs<<<grid, block>>>(d_A, d_B, d_C, nElem);
shared4R1Ws10ops2RG1WG<<<grid, block>>>(d_A, d_B, d_C, nElem);
shared4R20ops<<<grid, block>>>(d_A, d_B, d_C, nElem);
shared4R24ops<<<grid, block>>>(d_A, d_B, d_C, nElem);
shared4R40ops<<<grid, block>>>(d_A, d_B, d_C, nElem);
shared4R25ops<<<grid, block>>>(d_A, d_B, d_C, nElem);
shared4R15ops<<<grid, block>>>(d_A, d_B, d_C, nElem);
shared4RNops<<<grid, block>>>(d_A, d_B, d_C, nElem);
shared4RMops<<<grid, block>>>(d_A, d_B, d_C, nElem);

    dim3 grid4  (((nElem + block.x - 1) / block.x)/4);
    sum4<<<grid4, block>>>(d_A, d_B, d_C, nElem);
    sum4M<<<grid4, block>>>(d_A, d_B, d_C, nElem);
    sum4Man<<<grid4, block>>>(d_A, d_B, d_C, nElem);
    sum4K<<<grid4, block>>>(d_A, d_B, d_C, nElem);
   

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
           block.x, iElaps);

    // check kernel error
    CHECK(cudaGetLastError()) ;

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return(0);
}
