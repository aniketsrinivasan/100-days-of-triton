import torch
import time

import triton
import triton.language as tl 

from ..utils import benchmark


# get the device 
DEVICE = triton.runtime.driver.active.get_active_torch_device()


# addition kernel
@triton.jit 
def add_kernel(
    x_ptr,  # pointer to first input vector
    y_ptr,  # pointer to second input vector
    output_ptr,  # pointer to first element of pre-allocated output memory
    n_elements,  # size of the FULL vector => we know what elements to mask 
    BLOCK_SIZE: tl.constexpr  # number of elements each program should process
):
    # there are multiple "programs" accessing different data, so we should identify
    #   which program we are and which data we should access
    pid = tl.program_id(axis=0)  # 1-dimensional launch grid => axis is 0
    block_start = pid * BLOCK_SIZE
    
    # define list of pointers for each element in our computation
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # we will compute for this block

    # create mask to guard against out-of-bound accesses
    mask = offsets < n_elements  # notice this is the size of the full vector, so only one program will be masked

    # load x and y from DRAM, mask extra elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # write result back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)


# helper function to wrap the kernel functionality
def add(x: torch.Tensor, y: torch.Tensor):
    # check that both inputs have the same number of total elements
    assert x.numel() == y.numel()
    # pre-allocate the output
    output = torch.empty_like(x)
    assert x.device == y.device and y.device == output.device and output.device == DEVICE

    # get the total number of elements in the vector
    n_elements = output.numel()
    
    # the SPMD launch grid denotes the number of kernel instances that run in parallel 
    #   => same as CUDA launch grids
    # here, we use a 1D grid where the size is the number of blocks
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    # run the kernel on the grid 
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024
    )

    # we return a handle to the output but since torch.cuda.synchronize() hasn't been called,
    #   the kernel is running asynchronously at this point
    return output


def main():
    # create sample input vectors on device
    N = 1024 * 1024 
    x = torch.randn(N, device=DEVICE, dtype=torch.float32)
    y = torch.randn(N, device=DEVICE, dtype=torch.float32)

    # apply Torch function for validation
    o_torch = x + y
    o_triton = add(x, y)
    assert torch.allclose(o_triton, o_torch)

    # benchmark the kernel
    t_triton = benchmark(add, x, y)
    def torch_add(torch.Tensor: x, torch.Tensor: y):
        return x + y
    t_torch = benchmark(torch_add, x, y)

    print(f"Torch time: {t_torch}")
    print(f"Triton time: {t_triton}")
    return 
    

if __name__ == "__main__":
    main()
    