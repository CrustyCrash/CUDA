#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image.h"
#include "../stb/stb_image_write.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void brighten(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(y < height && x < width)
    {
        int greyOffset = y * width + x;
        int bgrOffset = greyOffset * channels;

        unsigned char blue = d_in[bgrOffset];
        unsigned char green = d_in[bgrOffset+1];
        unsigned char red = d_in[bgrOffset+2];

        int brightenBlue = min(255, 2*blue);
        int brightenGreen = min(255,2*green);
        int brightenRed = min(255, 2*red);

        d_out[bgrOffset] = brightenBlue;
        d_out[bgrOffset+1] = brightenGreen;
        d_out[bgrOffset+2] = brightenRed;
    }
}

int main()
{
    int width, height, channels;
    const char* path = "D:\\Cuda_prac\\Image Processing\\1280x720.jpg"; // hardcoding the path
    unsigned char* image = stbi_load(path,&width,&height,&channels,0);
    int size = width*height*channels;

    if(image == NULL)
    {
        std::cout << "Failed to load image" << std::endl;
        return 1;
    }
    unsigned char* d_in;
    unsigned char* d_out;

    cudaMalloc((void**)&d_in, size * sizeof(unsigned char));
    cudaMalloc((void**)&d_out, size * sizeof(unsigned char));

    cudaMemcpy(d_in, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 dimBlock(16,16);
    dim3 dimGrid((width+dimBlock.x-1)/dimBlock.x, (height+dimBlock.y-1)/dimBlock.y);

    brighten<<<dimGrid,dimBlock>>>(d_in,d_out,width,height,channels);
    cudaDeviceSynchronize();

    unsigned char* final = (unsigned char*)malloc(size*sizeof(unsigned char));
    cudaMemcpy(final,d_out,size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_jpg("Processed_Image.jpg",width,height,channels,final,100);

    stbi_image_free(image);
    free(final);
    cudaFree(d_in);
    cudaFree(d_out);

    std::cout <<"Image Processed Successfully"<<std::endl;
}