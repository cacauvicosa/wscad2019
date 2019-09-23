#include <cstdio>

void initialize(float* polynomial, const size_t N) {
	for (size_t i = 0; i < N; ++i)
		polynomial[i] = static_cast<float>(i);
}

__global__ void kernel(float* polynomial, const size_t N) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread < N) {
		float x = polynomial[thread];

		polynomial[thread] = 3 * x * x - 7 * x + 5;
	}
}

int main(int argc, char** argv) {
	const size_t BLOCK_DIM = 128;
	const size_t N_STREAMS = 4;

	// Number of elements in the arrays
	size_t n_elem = 1u << 27u;
	size_t n_bytes = n_elem * sizeof(float);

	// Allocating the array in pinned host memory for async memcpy
	float* h_polynomial;
	cudaHostAlloc(&h_polynomial, n_bytes, cudaHostAllocDefault);

	// Initializing data on host
	initialize(h_polynomial, n_elem);

	// Allocating the device array
	float* d_polynomial;
	cudaMalloc(&d_polynomial, n_bytes);

	// Number of elements per stream
	size_t n_elem_per_stream = n_elem / N_STREAMS;
	size_t n_bytes_per_stream = n_elem_per_stream * sizeof(float);

	// Events for time recording
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Grid and block
	dim3 block(BLOCK_DIM);
	dim3 grid((n_elem_per_stream + block.x - 1) / block.x);

	// Creating streams
	cudaStream_t* streams = new cudaStream_t[N_STREAMS];

	for (size_t i = 0; i < N_STREAMS; ++i)
		cudaStreamCreate(&streams[i]);

	cudaEventRecord(start, 0);

	//------------------------------------------------------- Asynchronous work
	for (size_t i = 0; i < N_STREAMS; ++i) {
		size_t offset = i * n_elem_per_stream;

		cudaMemcpyAsync(&d_polynomial[offset], &h_polynomial[offset],
			n_bytes_per_stream, cudaMemcpyHostToDevice, streams[i]);

		kernel<<<grid, block>>>(d_polynomial + offset, n_elem_per_stream);

		cudaMemcpyAsync(&h_polynomial[offset], &d_polynomial[offset],
			n_bytes_per_stream, cudaMemcpyDeviceToHost, streams[i]);
	}
	//-------------------------------------------------------------------------

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time;

	cudaEventElapsedTime(&time, start, stop);

	printf("Elapsed time: %.5f\n", time);

	// Destroying events
	cudaEventDestroy(stop);
	cudaEventDestroy(start);

	// Freeing memory
	delete[] streams;
	cudaFree(d_polynomial);
	cudaFreeHost(h_polynomial);
}
