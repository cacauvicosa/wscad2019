#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * kmeans example 2D
*/

void initialData(short int *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (short int)( rand() & 0xFFFF );
    }

    return;
}


/*
nvcc -arch=sm_35 -Xptxas -v kmeans_v2.cu
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z7kmeans4PsS_PiS0_i' for 'sm_35'
ptxas info    : Function properties for _Z7kmeans4PsS_PiS0_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, 356 bytes cmem[0]

*/

__global__ void kmeans4 (short int *input, short int*centroids, int*newcentroids, int *counter, const int n)
{
   int Dim = 4;
   int i = (blockIdx.x * blockDim.x + threadIdx.x)*Dim;
 if ( i < n ) {
// map
   int point_d0 = input[i+0];
   int point_d1 = input[i+1];
   int point_d2 = input[i+2];
   int point_d3 = input[i+3];

   int k0_d0 = point_d0 - centroids[0];
   int k0_d1 = point_d1 - centroids[1];
   int k0_d2 = point_d2 - centroids[2];
   int k0_d3 = point_d3 - centroids[3];

   int k1_d0 = point_d0 - centroids[4];
   int k1_d1 = point_d1 - centroids[5];
   int k1_d2 = point_d2 - centroids[6];
   int k1_d3 = point_d3 - centroids[7];

   int k2_d0 = point_d0 - centroids[8];
   int k2_d1 = point_d1 - centroids[9];
   int k2_d2 = point_d2 - centroids[10];
   int k2_d3 = point_d3 - centroids[11];

   int k3_d0 = point_d0 - centroids[12];
   int k3_d1 = point_d1 - centroids[13];
   int k3_d2 = point_d2 - centroids[14];
   int k3_d3 = point_d3 - centroids[15];


   k0_d0 *= k0_d0;
   k0_d1 *= k0_d1;
   k0_d2 *= k0_d2;
   k0_d3 *= k0_d3;

   k1_d0 *= k1_d0;
   k1_d1 *= k1_d1;
   k1_d2 *= k1_d2;
   k1_d3 *= k1_d3;

   k2_d0 *= k2_d0;
   k2_d1 *= k2_d1;
   k2_d2 *= k2_d2;
   k2_d3 *= k2_d3;

   k3_d0 *= k3_d0;
   k3_d1 *= k3_d1;
   k3_d2 *= k3_d2;
   k3_d3 *= k3_d3;

// reduce sum
   k0_d0 = k0_d0 + k0_d1 + k0_d2 + k0_d3;
   k1_d0 = k1_d0 + k1_d1 + k1_d2 + k1_d3;
   k2_d0 = k2_d0 + k2_d1 + k2_d2 + k2_d3;
   k3_d0 = k3_d0 + k3_d1 + k3_d2 + k3_d3;

// reduce min 
   int k01 = (k0_d0 < k1_d0 ) ? 0 : 1;
   int mink01 = (k0_d0 < k1_d0 ) ?  k0_d0: k1_d0 ;
   int k23 = (k2_d0 < k3_d0 ) ? 2 : 3;
   int mink23 = (k2_d0 < k3_d0 ) ?  k2_d0: k3_d0 ;
   
   int k = (mink01 < mink23) ? k01 : k23;

// add current point to new centroids sum

   newcentroids[Dim*k] += point_d0;
   newcentroids[Dim*k+1] +=point_d1;
   newcentroids[Dim*k+2] += point_d2;
   newcentroids[Dim*k+3] += point_d3;
   counter[k]++;

 } // if

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
    int Dim  = 4;
    int k = 4;
    int nElem = (1 << 21)*Dim;
    printf("Vector Size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(short int);
    size_t cBytes = Dim * k * sizeof(short int);
    size_t sBytes = Dim * k * sizeof(int);
    size_t tBytes = k * sizeof(int);


    short int *h_data, *h_centroids;
    h_data     = (short int *)malloc(nBytes);
    initialData(h_data,nElem);
    h_centroids     = (short int *)malloc(cBytes);
    //initialData(h_centroids,k*Dim);
    h_centroids[0]= 15000;
    h_centroids[1]= 15000;
    h_centroids[2]= 15000;
    h_centroids[3]= 15000;
    h_centroids[4]= 5000;
    h_centroids[5]= 5000;
    h_centroids[6]= 5000;
    h_centroids[7]= 5000;
    h_centroids[8]= -15000;
    h_centroids[9]= -15000;
    h_centroids[10]= -15000;
    h_centroids[11]= -15000;
    h_centroids[12]= -5000;
    h_centroids[13]= -5000;
    h_centroids[14]= -5000;
    h_centroids[15]= -5000;

    int *h_newcentroids;
    h_newcentroids = (int *)malloc(sBytes);
    int *h_counter;
    h_counter = (int *)malloc(tBytes);

    memset(h_newcentroids, 0, sBytes);
    memset(h_counter,  0, tBytes);

    // malloc device global memory
    short int *d_data, *d_centroids;
    CHECK(cudaMalloc((short int**)&d_data, nBytes));
    CHECK(cudaMalloc((short int**)&d_centroids, cBytes));
    int *d_newcentroids;
    CHECK(cudaMalloc((int**)&d_newcentroids, sBytes));
    int *d_counter;
    CHECK(cudaMalloc((int**)&d_counter, tBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_data, h_data, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_centroids, h_centroids, cBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_newcentroids, h_newcentroids, sBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_counter, h_counter, tBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int iLen = 256;
    dim3 block (iLen);
    dim3 grid  (((nElem/Dim + block.x - 1) / block.x));
  for (int i=0;i < 20; i++) {
    kmeans4<<<grid, block>>>(d_data, d_centroids, d_newcentroids, d_counter, nElem);
    CHECK(cudaDeviceSynchronize());
    printf("kmeans <<<  %d, %d  >>>  \n", grid.x,
           block.x);
    // check kernel error
    CHECK(cudaGetLastError()) ;
    // copy kernel result back to host side
    CHECK(cudaMemcpy(h_newcentroids, d_newcentroids, sBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_counter, d_counter, tBytes, cudaMemcpyDeviceToHost));
    int total = nElem/Dim;

    for (int j=0; j < k*Dim; j++) {
       printf(" centroids dim %d value %d  \n", j, h_centroids[j]);    
       h_centroids[j] = (short int) (h_newcentroids[j]/h_counter[j/4]);
       if ( j% 4) printf(" counter j %d =  %d  \n", j/4, h_counter[j/4]);
       if ( j% 4 == 0 )total -= h_counter[j/4];
    }
    if ( total== 0 ) printf(" ---OK------------ \n");  
    else   printf(" ---total %d------------ \n",total);

    memset(h_newcentroids, 0, sBytes);
    memset(h_counter,  0, tBytes);
    CHECK(cudaMemcpy(d_centroids, h_centroids, cBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_newcentroids, h_newcentroids, sBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_counter, h_counter, tBytes, cudaMemcpyHostToDevice));
  }
    // check device results

    // free device global memory
    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_centroids));
    CHECK(cudaFree(d_newcentroids));
    CHECK(cudaFree(d_counter));

    // free host memory
    free(h_data);
    free(h_centroids);
    free(h_newcentroids);
    free(h_counter);

    return(0);
}
