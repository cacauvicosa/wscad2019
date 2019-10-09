#include <stdio.h>
#include <cuda_runtime_api.h>

#define SIZE 27
#define RADIUS 3
#define GRID_SIZE 512
#define BLOCK_SIZE 256

__global__ void stencil_1d(int *in, int *out, int dim) {

    __shared__ int temp[BLOCK_SIZE + 2*RADIUS];

  int lindex = threadIdx.x + RADIUS;
  int gindex = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = gridDim.x * blockDim.x;
  int left, right;

  // Go through all data
  // Step all threads in a block to avoid synchronization problem
  while ( gindex < (dim + blockDim.x) ) {

    // Read input elements into shared memory
    temp[lindex] = 0;
    if (gindex < dim) 
      temp[lindex] = in[gindex];

    // Populate halos, set to zero if we are at the boundary
    if (threadIdx.x < RADIUS) {

      temp[lindex - RADIUS] = 0;
      left = gindex - RADIUS;
      if (left >= 0)
        temp[lindex - RADIUS] = in[left];

      temp[lindex + blockDim.x] = 0;
      right = gindex + blockDim.x;
      if (right < dim)
        temp[lindex + blockDim.x] = in[right];
    }

    // Synchronize threads - make sure all data is available!
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
      result += temp[lindex + offset];
    }

    // Store the result
    if (gindex < dim)
      out[gindex] = result;

    // Update global index and quit if we are done
    gindex += stride;

    __syncthreads();

  }
}

void verify(int *h_in, int *h_out, int N) {
    int i, j, ij, result, err;
    // check results
    err = 0;
    for (i=0; i < N; i++){
        result = 0;
        for (j = -RADIUS; j <= RADIUS; j++){
            ij = i+j;
            if (ij >= 0 && ij < N)
                result += h_in[ij];
        }
        if (h_out[i] != result) {
            err++;
            printf("\nERROR:\n");
            printf("result = %d != %d = h_out[%d]\n", result, h_out[i], i);
            exit(1);
        }
    }

    if (err != 0){
        printf("\n Error, %d elements do not match!\n\n", err);
    } else {
        printf("\n Success! All elements match CPU result.\n\n");
    }
}

int main(void) {
    
    int *h_in, *h_out; // host copies of a, b, c
    int *d_in, *d_out; // device copies of a, b, c

    int N = 1 << SIZE;

    int size = N * sizeof(int);

    // Apply stencil by launching a sufficient number of blocks
    printf("---------------------------\n");
    printf("Launching 1D stencil kernel\n");
    printf("---------------------------\n");
    printf("Vector length     = %d (%d MB)\n",N,size/1024/1024);
    printf("Stencil radius    = %d\n",RADIUS);
    printf("Blocks            = %d\n",GRID_SIZE);
    printf("Threads per block = %d\n",BLOCK_SIZE);
    printf("Total threads     = %d\n",GRID_SIZE*BLOCK_SIZE);
    
    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);
    
    // Alloc space for host copies of a, b, c and setup input values
    h_in = (int *)malloc(size);
    h_out = (int *)malloc(size);

    // initialize vector
    for (int i=0; i < N; i++){
        h_in[i] = 1;
    }

    // Copy inputs to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, size, cudaMemcpyHostToDevice);
    
    // Launch stencil_1d kernel on GPU
    stencil_1d<<<GRID_SIZE,BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize(); 
    
    // Copy result back to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Verify results
    verify(h_in, h_out, N);
    
    // Cleanup
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
