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

cudaEvent_t start, stop;
float elapsed_time;

void start_event() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);   
    cudaEventRecord(stop, 0);
}

void end_event(char* name) {
	cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("\n%s - Execution time = %.6fms\n", name, elapsed_time);
}

int main(void) {
    
    int *h_in, *h_out; // host copies of a, b, c
    int *d_in, *d_out; // device copies of a, b, c

    int N = 1 << SIZE;

    int size = N * sizeof(int);

    
    
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
    start_event();
    stencil_1d<<<GRID_SIZE,BLOCK_SIZE>>>(d_in, d_out, N);
    end_event("Stencil_1d");
    cudaDeviceSynchronize(); 
    
    // Copy result back to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Apply stencil by launching a sufficient number of blocks
    printf("---------------------------\n");
    printf("Launching 1D stencil kernel\n");
    printf("---------------------------\n");
    printf("Vector length     = %d (%d MB)\n",N,size/1024/1024);
    printf("Stencil radius    = %d\n",RADIUS);
    printf("Blocks            = %d\n",GRID_SIZE);
    printf("Threads per block = %d\n",BLOCK_SIZE);
    printf("Total threads     = %d\n",GRID_SIZE*BLOCK_SIZE);
    printf("GOPS              = %d\n",N/elapsed_time/1000000000);

    // Verify results
    verify(h_in, h_out, N);
    
    // Cleanup
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
