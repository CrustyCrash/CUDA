import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
cuda_code = """
__global__ void add(int* a, int* b, int* c, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        c[idx] = a[idx] + b[idx];
    }
}
"""

#host data
a_host = np.array([1,2,3], dtype=np.int32)
b_host = np.array([4,5,6], dtype=np.int32)
size = len(a_host)

#device data
a_device = cuda.mem_alloc(a_host.nbytes)
cuda.memcpy_htod(a_device, a_host)
b_device = cuda.mem_alloc(b_host.nbytes)
cuda.memcpy_htod(b_device, b_host)

#load cuda module
cuda_module = SourceModule(cuda_code)
add_kernel = cuda_module.get_function("add")

#set up block and grid dimensions
block_dim = (size,1,1)
grid_dim = (1,1)

#launch the cuda kernel
add_kernel(a_device, b_device, cuda.mem_alloc(a_host.nbytes), np.int32(size), block=block_dim, grid=grid_dim)

#copy result back to host
c_host = np.empty(like=a_host)
cuda.memcpy_dtoh(c_host, a_device)
print(c_host)
