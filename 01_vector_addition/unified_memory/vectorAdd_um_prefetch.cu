#include <stdio.h>
#include <cassert>
#include <iostream>

using std::cout;

__global__ void vectorAdd(int *a, int *b, int *c, int N){
    // Calculate global thread ID
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Boundary check
    if(tid < N){
        c[tid] = a[tid] + b[tid];
    }
}

int main(){
    // Array size of 2^16
    const int N = 1 << 16;
    size_t bytes = N * sizeof(int);

    // Declare unified memory pointers
    int *a, *b, *c;

    // Allocate memory for these pointers
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Get the device ID for prefetching calls
    int id = cudaGetDevice(&id);

    // Set some hints about the data and do some prefetching 
    cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, id);

    // Initialze vectors
    for(int i = 0; i < N; i++){
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    // Pre-fetch 'a' and 'b' array to the specified deivce (GPU)
    cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);

    // Threads per CTA 
    int BLOCK_SIZE = 1 << 10;
    int GRID_SIZE = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;

    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);

    // Wait for all previous operations before using values
    cudaDeviceSynchronize();

    // Prefetch to the host (CPU)
    cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    // Verify the result on CPU
    for(int i = 0; i < N; i++){
        if(c[i] != a[i] + b[i]){
            cout << "c is " << c[i] << " ; a + b is : " << a[i] + b[i] << std::endl;
        }
    }

    // Free unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    cout << "Completed\n";

    return 0;

}
