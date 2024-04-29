#include <iostream>

__global__ void add(int a, int b, int *c) //function that will run on gpu
{
    *c = a + b; // int* c is a pointer thsat takes the address of the memory location in device and writes to that address
}

int main()
{
    int host_var; //variable on host that will store the result from the device
    int* address; //pointer on the host that will store the address of memory on the device

    cudaMalloc( (void**)&address,sizeof(int)); //allocating memory of size int on device. 

    add <<<1,1>>> (2,7,address); //calling the function from the host and passing the address in device memory for the function to write on

    cudaMemcpy(&host_var,address,sizeof(int),cudaMemcpyDeviceToHost);
    
    printf("2 + 7 is %d \n",host_var);
    cudaFree(address);
    
    return 0;
}