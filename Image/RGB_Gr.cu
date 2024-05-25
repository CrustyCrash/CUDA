#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel(unsigned char* d_image, int width, int height, int channels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x; //rows
    int y = threadIdx.y + blockIdx.y * blockDim.y; //cols

    while(x < height && )
}

int main()
{
    const char* path = "D:\\Cda_prac\\Image\\1280x720.jpg"; //hardcoding the path
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if(image.empty())
    {
        std::cerr<<"Failed to find or read "<<path<<std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();

    //allocating memory on gpu
    unsigned char* d_in;
    cudaMalloc((void**)&d_in, width*height*sizeof(unsigned char));

    //copying image data to gpu
    cudaMemcpy(d_in, image.data, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);

    //setting block and grid dims
    dim3 dimBlock(16,16);
    dim3 dimGrid((width+dimBlock.x-1)/dimBlock.x, (height+dimBlock.y-1)/dimBlock.y);

    //launching kernel
    kernel<<<dimGrid,dimBlock>>>(d_in,width,height,channels);



}