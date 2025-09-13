import torch 

import triton 
import triton.language as tl 

from kernels import _attn_fwd, _attn_bwd_preprocess


# every implementation of a PyTorch operation (e.g. any function) must inherit from 
#   torch.autograd.Function and provide some basic functions (forward and backward)
class TritonAttention(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, Q, K, V, causal, softmax_scale):
        """
        Forward-propagation for FlashAttention.

        We expect Q, K, V to have shape [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM] = [B, N, S, D]

        Q, K, V are expected to already have passed through linear W_Q, W_K, W_V layers since FlashAttention
        does not care for optimizing performance for that linear operation. 

        Args:
            ctx: "context". when training NNs using autograd, when computing the backward pass, we must
                 reuse the activations of each computation node of the forward pass => we save info to ctx
                 ctx is a dictionary-like object made available by PyTorch
            Q, K, V, causal, softmax_scale: self-explanatory attention tensor/float arguments
        """
        # extract shapes and ensure they are what we expect: 
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape 
        assert (HEAD_DIM_Q == HEAD_DIM_K) and (HEAD_DIM_K == HEAD_DIM_V)

        # pre-allocate the output tensor 
        O = torch.empty_like(Q)  # output has the same shape as Q (output of W_Q)

        stage = 3 if causal else 1  # used later to determine which operation to use 

        # launch grid for Triton
        #   we want to launch programs along 2 dimensions 
        #   each sequence in the batch and each head in the sequence should work independently of one another
        #       for each of these programs, we divide SEQ_LEN into blocks of queries (by some BLOCK_SIZE_Q)
        #   cdiv = ceiling division => ceil(SEQ_LEN / BLOCK_SIZE_Q), how many blocks of Q we have 
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),  # which group of queries we are going to work with
            BATCH_SIZE * NUM_HEADS,  # which head of which batch element we are going to work with
            1,  # z-dimension in CUDA launch grid; we don't want another level of parallelism 
        )
        # number of parallel programs: (BATCH_SIZE * NUM_HEADS * NUM_BLOCKS_Q)

        # M is the logsumexp for the backward pass, one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )  # maximum for each row+log(normalization_factor) so we don't re-compute during backward prop.

        # LAUNCH KERNEL:
        #   note: we get a POINTER to the start elements of (flat) Q, K, V
        #         => we must figure out indices ourselves when computing attention 
        #         => we pass in all the necessary strides and shapes to index properly
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,  # logsumexp storage
            O=O,  # pre-allocated output tensor 
            # strides are passed using the tensor strides (managed by PyTorch)
            stride_Q_batch=Q.stride[0],
            stride_Q_head=Q.stride[1],
            stride_Q_seq=Q.stride[2],
            stride_Q_dim=Q.stride[3],
            stride_K_batch=K.stride[0],
            stride_K_head=K.stride[1],
            stride_K_seq=K.stride[2],
            stride_K_dim=K.stride[3],
            stride_V_batch=V.stride[0],
            stride_V_head=V.stride[1],
            stride_V_seq=V.stride[2],
            stride_V_dim=V.stride[3],
            stride_O_batch=O.stride[0],
            stride_O_head=O.stride[1],
            stride_O_seq=O.stride[2],
            stride_O_dim=O.stride[3],
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,  # identify whether to apply causal or non-causal attention
            # we don't pass BLOCK_SIZE_Q and BLOCK_SIZE_KV because these will be passed by Auto Tuning decorator
        )

        # save information for backward pass:
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid 
        ctx.softmax_scale = softmax_scale 
        ctx.HEAD_DIM = HEAD_DIM_K 
        ctx.causal = causal  # must mask out during backward pass based on causal attention 

        return O
    
    @staticmethod
    def backward(ctx, dO):
        # in Triton, we load from HBM to SRAM, and then save from SRAM to HBM
        #   to avoid materializing QK^T from HBM (this is very large), we do not store this to HBM and
        #   thus also do not load it from SRAM
        # it is faster to load such large tensors on-the-fly rather than to be I/O-bound by storing/loading
        Q, K, V, O, M = ctx.saved_tensors  # information we saved during the forward pass to the context

        # check information
        assert dO.is_contiguous()  # assert that dO is a contiguous block of memory
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()

        # initialize empty gradient matrices (where shape of gradient vector is the same shape as the element with
        #   which we compute the gradient with respect to)
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]  # [B, N, S, D]
        NUM_WARPS, NUM_STAGES = 4, 3  # indication on how many threads and how many softer-pipelining stages are used
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        # pre-process kernel launch 
        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM
        )

        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)
        stage = 3 if ctx.causal else 1  # whether causal attention or not (for backward prop.)
        # fix KV and iterate through all the Q blocks 
        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES
        )
