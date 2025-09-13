import torch 

import triton 
import triton.language as tl 


@triton.jit
def _vector_relu_fwd(
    X_ptr,
    OUT_ptr,
    NUM_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pass


@triton.jit
def _vector_relu_bwd(
    X_ptr, 
    OUT_ptr,
    NUM_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pass
