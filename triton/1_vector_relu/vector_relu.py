import torch 

import triton 
import triton.language as tl 


class VectorReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """"
        Forward-propagation for vector ReLU. 

        x.shape is of the form [D] (vector).
        """
        DIM = x.shape[0]
        num_elements = x.numel()
        # pre-allocate output tensor:
        O = torch.empty_like(x)
        # define launch grid
        BLOCK_SIZE = 128  # size of each block to split the ReLU into
        grid_size = triton.cdiv(DIM, BLOCK_SIZE)  # how many programs to launch 
        grid = (grid_size,) 

        ker_out = _vector_relu_fwd[grid](
            X_ptr=x,
            OUT_ptr=O,
            NUM_ELEMENTS=num_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(x)
        ctx.grid = grid
        ctx.block_size = BLOCK_SIZE
        ctx.numel = num_elements

        return O
    
    def backward(ctx, dO):
        """
        Backward propagation for ReLU. The gradient w.r.t. input x is given by the following simple function: 
            (ReLU(x))' = (0 if x <= 0), (1 if x > 0)
        """
        x = ctx.saved_tensors
        assert dO.is_contiguous()  # should be a contiguous block of memory, already pre-allocated

        # define launch grid
        BLOCK_SIZE = ctx.block_size
        grid = ctx.grid
        num_elements = ctx.numel
        assert x.numel() == num_elements  # check that number of elements match

        ker_out = _vector_relu_bwd[grid](
            X_ptr=x,
            OUT_ptr=dO,
            NUM_ELEMENTS=num_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
