#include "../common/common.h"
#include <stdio.h>

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */

__global__ void helloFromGPU()
{
   if ( threadIdx.x == 0 ) {
    printf("Hello World from GPU! %d\n",blockIdx.x);
   }
}

int main(int argc, char **argv)
{
   
    helloFromGPU<<<1024,1024>>>();
 CHECK(cudaDeviceReset());
 printf("Hello World from CPU!\n");
   
    return 0;
}


