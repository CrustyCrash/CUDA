import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from timeit import default_timer as timer

#cpu array addition 
def cpu_add(a,b):
    return a+b

#gpu array addition
def gpu_add(a_gpu, b_gpu, result_gpu, size):
    block_size = 256
    grid_size = (size+block_size-1)//block_size

    add_array_cuda(a_gpu, b_gpu, result_gpu, np.int32(size), block=(block_size,1,1), grid=(grid_size,1))

#generate random arryas
size = 25000 * 25000
a_cpu = np.random.rand(size).astype(np.float32)
b_cpu = np.random.rand(size).astype(np.float32)
result_cpu = np.zeros_like(a_cpu)

#allocate gpu memory
a_gpu = cuda.mem_alloc(a_cpu.nbytes)
b_gpu = cuda.mem_alloc(b_cpu.nbytes)
result_gpu = cuda.mem_alloc(result_cpu.nbytes)

#copy data to gpu
cuda.memcpy_htod(a_gpu, a_cpu)
cuda.memcpy_htod(b_gpu, b_cpu)

#compile cuda code
mod = SourceModule("""
    __global__ void add_array(float *a, float *b, float *result, int size
                   {
                   int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if(idx < size)
                        {
                            result[idx] = a[idx] + b[idx];
                        }
                   }
""")

add_array_cuda = mod.get_function("add_array")

# Record start time
start_time = timer()

# Launch GPU kernel
gpu_add(a_gpu, b_gpu, result_gpu, size)

# Record end time
end_time = timer()

# Copy result back to CPU
result_cpu = np.empty_like(a_cpu)
cuda.memcpy_dtoh(result_cpu, result_gpu)

# Record GPU processing time
gpu_time = (end_time - start_time) * 1000  # convert to milliseconds
print(f"GPU array addition time: {gpu_time:.2f} ms")