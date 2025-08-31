import torch 
import math

import triton 
import triton.language as tl 


@triton.jit 
def _add(A_ptr, B_ptr, OUT_ptr, NUM_ELEMENTS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE  # offset where our current block starts 
    thread_offsets = block_start + tl.arange(0, BLOCK_SIZE)  # pointers for each individual element
    # need a mask since we don't know that #threads = #data_elements
    mask = thread_offsets < NUM_ELEMENTS  # whether this element gets masked (vector)

    # read values from GPU memory into registers:
    A_ptrs = tl.load(A_ptr + thread_offsets, mask=mask)
    B_ptrs = tl.load(B_ptr + thread_offsets, mask=mask)

    # compute result
    result = A_ptrs + B_ptrs 
    # store result 
    tl.store(OUT_ptr + thread_offsets, result, mask=mask)


def tensor_add(A: torch.tensor, B: torch.tensor) -> torch.tensor:
    assert A.is_cuda and B.is_cuda
    # check shapes before adding (both tensors must have same number of elements)
    assert A.numel() == B.numel()
    num_elements = A.numel()
    # create output (pre-allocate)
    out = torch.empty_like(A)

    BLOCK_SIZE = 128  # size of each block to split the addition into 
    grid_size = math.ceil(num_elements / BLOCK_SIZE)  # how many programs to launch 
    grid = (grid_size, ) 

    ker_out = _add[grid](
        A_ptr=A,
        B_ptr=B, 
        OUT_ptr=out,
        NUM_ELEMENTS=num_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out 


def test_kernel():
    torch.manual_seed(2020)  # seeds both CPU and GPU
    vec_size = 8192

    # initialize random tensors
    a = torch.rand(vec_size, device="cuda")
    b = torch.rand(vec_size, device="cuda")

    torch_result = a + b 
    triton_result = _add(a, b)

    error_threshold = 1e-3
    fidelity = torch.allclose(torch_result, triton_result, atol=error_threshold)
    print(f"{fidelity}")


if __name__ == "__main__":
    test_kernel()

