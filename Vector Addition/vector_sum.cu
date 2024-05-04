#include <stdio.h>
#include "book.h"

#define N (33 * 1024)
__global__ void sum(int*a , int* b, int* c)
{
    // linearize threadID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid = tid + blockDim.x + gridDim.x; //incrementing by total number of threads in grid
    }
}

int main()
{
    int a[N];
    int b[N];
    int c[N];

    int* dev_a;
    int* dev_b;
    int* dev_c;

    //populating the array on the host
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = -i;
    }

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    sum <<<128,128>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    //verify that the gpu did the work successfully
    bool verify = true;
    for(int i = 0; i < N; i++)
    {
        if (a[i] + b[i] != c[i])
        {
            printf("Failed at %d + %d != %d\n",a[i],b[i],c[i]);
            verify = false;
        }
    }
    if (verify)
    {
        printf("Program executed successfully!\n");
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}