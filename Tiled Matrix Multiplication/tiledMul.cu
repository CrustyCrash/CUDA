#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int width = 1024;
#define Tile_width 16


__global__ void tileMul(int* dev_a, int* dev_b, int* dev_c, int width)
{
    __shared__ int tileA[Tile_width][Tile_width];
    __shared__ int tileB[Tile_width][Tile_width];

    int rows = blockIdx.y * Tile_width + threadIdx.y;
    int cols = blockIdx.x * Tile_width + threadIdx.x;

    int product = 0;
    for(int i = 0; i < width/Tile_width; i++)
    {
        /*
        row*width -> takes you to the row in global memory of the matrix
        i*tile_width -> takes you to the column for that matrix (imagine cropping the matrix to that column)
        +threadIdx.x -> takes you to that element in that element

        i*width - > take you to the column

        */
        tileA[threadIdx.y][threadIdx.x] = dev_a[rows*width + i*Tile_width + threadIdx.x ];
        tileB[threadIdx.y][threadIdx.x] = dev_b[(i*Tile_width + threadIdx.y)*width + cols];
        __syncthreads();

        for(int k = 0; k < Tile_width; k++)
        {
            product += tileA [threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    dev_c[rows * width + cols] = product;
}

int main()
{
    srand(time(0));

    int* a = (int*)malloc(width * width * sizeof(int));
    int* b = (int*)malloc(width * width * sizeof(int));
    int* c = (int*)malloc(width * width * sizeof(int));

    //populating on the host
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < width; j++)
        {
            a[i * width + j] = rand() % 10;
            b[i * width + j] = rand() % 10;
        }
    }

    int* dev_a;
    int* dev_b;
    int* dev_c;

    cudaError_t err = cudaMalloc((void**)&dev_a, width * width * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("Error in cudaMalloc for dev_a\n");

        cudaFree(dev_a);
        free(a);
        free(b);
        free(c);

        return -1;
    }
    err = cudaMalloc((void**)&dev_b, width * width * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("Error in cudaMalloc for dev_b\n");

        cudaFree(dev_a);
        cudaFree(dev_b);
        free(a);
        free(b);
        free(c);

        return -1;
    }
    err = cudaMalloc((void**)&dev_c, width * width * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("Error in cudaMalloc for dev_c\n");

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        free(a);
        free(b);
        free(c);

        return -1;
    }

    err  = cudaMemcpy(dev_a, a, width*width*sizeof(int), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        printf("Error in cudaMemcpy for dev_a\n");
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        free(a);
        free(b);
        free(c);
        return -2;
    }

    err  = cudaMemcpy(dev_b, b, width*width*sizeof(int), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        printf("Error in cudaMemcpy for dev_b\n");
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        free(a);
        free(b);
        free(c);
        return -2;
    }


    dim3 dimBlock (16,16);
    dim3 dimGrid ((width + Tile_width - 1)/Tile_width, (width + Tile_width - 1)/Tile_width);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tileMul<<<dimGrid,dimBlock>>>(dev_a,dev_b,dev_c,width);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    err = cudaMemcpy(c, dev_c, width*width*sizeof(int), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        printf("Error in cudaMemcpy for dev_c to c\n");
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        free(a);
        free(b);
        free(c);
        return -2;
    }

    float ms = 0.0;
    cudaEventElapsedTime(&ms, start,stop);

    printf("Time takes to execute kernel : %.3fms\n",ms);

    // for (int i = 0; i < width; ++i)
    // {
    //     for (int j = 0; j < width; ++j) 
    //         {
    //              printf("%d ", c[i * width + j]);
            
    //         }
    //     printf("\n");
    // }
}