#include <iostream>
__global__ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
}
int main(void)
{
    int sum;
    int *device1;
    int *device2;
    int *dev_c;

    cudaError_t cudaStatus;

    

   cudaStatus =  cudaMalloc((void **)&device1, sizeof(int));
   if(cudaStatus != cudaSuccess)
   {
    fprintf(stderr,"Error allocating memory: %s\n",cudaGetErrorString(cudaStatus));
    return 1;
   }

    cudaStatus = cudaMalloc((void **)&device2, sizeof(int));
    if(cudaStatus != cudaSuccess){
        fprintf(stderr,"Error allocating memory: %s\n",cudaGetErrorString(cudaStatus));
        cudaFree(device1);
        return 1;
    }

    cudaStatus = cudaMalloc((void **)&dev_c, sizeof(int));
    if(cudaStatus != cudaSuccess){
        fprintf(stderr,"Error allocating memory: %s\n",cudaGetErrorString(cudaStatus));
        cudaFree(device1);
        cudaFree(device2);
        return 1;
    }

    int a = 2;
    int b = 7;

    cudaMemcpy(device1, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device2, &b, sizeof(int), cudaMemcpyHostToDevice); // Corrected line

    add<<<1, 1>>>(device1, device2, dev_c);

    cudaMemcpy(&sum, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(device1);
        cudaFree(device2);
        cudaFree(dev_c);
        return 1;
    }
    
    printf("2 + 7 = %d\n", sum);
    cudaFree(dev_c);
    cudaFree(device1);
    cudaFree(device2);
    return 0;
}