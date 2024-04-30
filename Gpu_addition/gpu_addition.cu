#include <stdio.h>
#include "book.h"
#define N 10

__global__ void add(int* a, int* b, int* c)
{
    int tid = blockIdx.x; //block ID
    //printf("%d\n",tid);
    
    if( tid < N )
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main()
{
    int a[N];
    int b[N];
    int c[N];

    //pointers to hold memory address of gpu where arrays will be loaded
    int* dev_memA;
    int* dev_memB;
    int* dev_memC;

    //allocating memory on the device

    HANDLE_ERROR (cudaMalloc((void**)&dev_memA, N* sizeof(int)));
    HANDLE_ERROR (cudaMalloc((void**)&dev_memB, N * sizeof(int)));
    HANDLE_ERROR (cudaMalloc((void**)&dev_memC, N * sizeof(int)));

    //populating arrays on host for convenience

    for(int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    //copying array from host to device

    HANDLE_ERROR (cudaMemcpy(dev_memA, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR (cudaMemcpy(dev_memB, b, N * sizeof(int), cudaMemcpyHostToDevice));

    //calling the function on device with N blocks
    add <<<N,1>>> (dev_memA,dev_memB,dev_memC);

    //copying result from device to host

    HANDLE_ERROR (cudaMemcpy(c, dev_memC, N * sizeof(int), cudaMemcpyDeviceToHost));

    for(int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i],b[i],c[i]);
    }

    cudaFree(dev_memA);
    cudaFree(dev_memB);
    cudaFree(dev_memC);

}