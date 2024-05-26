#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image.h"
#include "../stb/stb_image_write.h"
#include <cuda_runtime.h>
#include <iostream>
#define BLURSIZE 5

__global__ void blur(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels)
{
    int cols = threadIdx.x + blockIdx.x * blockDim.x;
    int rows = threadIdx.y + blockIdx.y * blockDim.y;

    
    if(rows < height && cols < width)
    {
        for(int c = 0; c < channels; c++)
        {
            int pixelVal = 0;
            int pixels = 0;
            for(int i = -BLURSIZE; i < BLURSIZE+1; i++)
            {
                for(int j = -BLURSIZE; j < BLURSIZE+1; j++)
                {
                    int currRow = rows + i;
                    int currCol = cols + j;

                    if(currRow >= 0 && currRow < height && currCol >=0 && currCol < width)
                    {
                        pixelVal += d_in[(currRow * width + currCol) * channels + c];
                        pixels++;
                    }
                    
                }
            
            }
            d_out[(rows * width + cols) * channels + c] = (unsigned char)(pixelVal / pixels);
        }
    }
}

int main()
{
    int width, height, channels;
    unsigned char* image = stbi_load("D:\\Cuda_prac\\Image Processing\\1280x720.jpg",&width, &height, &channels, 0);
    if(image == NULL)
    {
        std::cout<<"Failed to load or find image."<<std::endl;
        return 1;
    }

    size_t size = width * height * channels * sizeof(unsigned char);

    unsigned char* d_in;
    unsigned char* d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, image, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock (16,16);
    dim3 dimGrid ((width+dimBlock.x-1)/dimBlock.x, (height+dimBlock.y-1)/dimBlock.y);
    blur <<<dimGrid,dimBlock>>>(d_in,d_out,width,height,channels);
    cudaDeviceSynchronize();
    
    unsigned char* host_image = (unsigned char*)malloc(size);
    cudaMemcpy(host_image, d_out, size, cudaMemcpyDeviceToHost);

    stbi_write_jpg("Processed_Image.jpg",width,height,channels,host_image,100);

    stbi_image_free(image);
    free(host_image);
    cudaFree(d_in);
    cudaFree(d_out);

    std::cout<<"File successfully processed"<<std::endl;

    return 0;

    
}