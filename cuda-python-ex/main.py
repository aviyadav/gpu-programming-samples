import numpy as np
from numba import cuda


# Kernal function to add two arrays
@cuda.jit
def add_arrays_gpu(a, b, result):
    idx = cuda.grid(1)
    if idx < a.size:
        result[idx] = a[idx] + b[idx]


# Host code
def main():
    # init data (input arrays)
    n = 100
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 12

    # prepare GPU output array
    result = np.zeros(n, dtype=np.float32)

    # Transfer data to GPU
    a_gpu = cuda.to_device(a)
    b_gpu = cuda.to_device(b)
    result_gpu = cuda.to_device(result)

    # Launch GPU kernel
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block
    add_arrays_gpu[blocks_per_grid, threads_per_block](a_gpu, b_gpu, result_gpu)

    # Transfer result back to host CPU
    result_gpu.copy_to_host(result)

    # Clean up
    del a_gpu
    del b_gpu
    del result_gpu

    print("Array a : ", a)
    print("Array b : ", b)
    print("Result a + b : ", result)


if __name__ == "__main__":
    main()
