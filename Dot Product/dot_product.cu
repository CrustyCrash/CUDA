#include <stdio.h>
#include <cuda_runtime.h>
#include "book.h"

#define min(a, b) a < b ? a : b
#define ThreadsPerBlock 256
const int t_elements = 31 * 1024;
const int BlocksPerGrid = min(32, ((t_elements + (ThreadsPerBlock - 1)) / ThreadsPerBlock));

__global__ void dot_product(float *a, float *b, float *c)
{
    __shared__ float cache[ThreadsPerBlock];         // shared memory in each block
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // finding global threadID
    int cacheIndex = threadIdx.x;

    float temp = 0; // private variable for each thread to write into
    while (tid < t_elements)
    {
        temp = temp + a[tid] * b[tid];
        tid = tid + blockDim.x * gridDim.x; // increamenting by total number of threads, so no element goes unprocessed or is taken up twice
    }

    cache[cacheIndex] = temp; // storing result of each thread in an array that is shared between threads in a block.

    // waiting for all threads in a block to write to the array before proceeding
    __syncthreads();

    // reduction

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i)
        {
            cache[cacheIndex] += cache[cacheIndex + i]; // offsetting by value of i
        }
        __syncthreads();
        i = i/2;
    }

    // after the above operation, every block will be left with just 1 item
    if (cacheIndex == 0) // making the first thread in every block do this
    {
        c[blockIdx.x] = cache[0];
    }
}

int main()
{
    // pointers to array
    float *a;
    float *b;
    float c;
    float *partial_c;

    // pointers to memory location in gpu
    float *dev_a;
    float *dev_b;
    float *dev_c;

    // dynamically allocating memory for arrays on the host
    a = (float *)malloc(t_elements * sizeof(float));
    b = (float *)malloc(t_elements * sizeof(float));
    partial_c = (float *)malloc(BlocksPerGrid * sizeof(float));

    // populating the arrays on the host
    for (int i = 0; i < t_elements; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    // allocating memory for arrays on the device
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, t_elements * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, t_elements * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, BlocksPerGrid * sizeof(float)));

    //copying array from host to device
    HANDLE_ERROR(cudaMemcpy(dev_a, a, t_elements * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, t_elements * sizeof(float), cudaMemcpyHostToDevice));

    dot_product<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_a, dev_b, dev_c);

    // copying the resultant array back to host to finish addition
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_c, BlocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    c = 0;
    for (int i = 0; i < BlocksPerGrid; i++)
    {
        c = c + partial_c[i];
    }

#define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)
    printf( "Does GPU value %.6g = %.6g?\n", c,2 * sum_squares( (float)(t_elements - 1) ) );
    // free memory on the GPU side
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    // free memory on the CPU side
    free(a);
    free(b);
    free(partial_c);
}
