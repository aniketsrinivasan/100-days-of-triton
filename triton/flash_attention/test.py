import torch

import triton
import triton.language as tl 


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    """
    Test FlashAttention implementation using empty matrices of sizes defined (as arguments).

    Args:
        [B, N, S, D] shape of the Q, K, V tensors created. 
        causal: whether to perform causal attention. 
    """
    # Create empty tensors of N(0, 1/2), shape [B, N, S, D] and specified dtype
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal(mean=0.0, std=0.5)
        .requires_grad_()
    )
    
    # (QK^T)/(sqrt(D)) in attention, we test against a naive implementation below: 
    softmax_scale = 1 / (HEAD_DIM ** 0.5)
    d0 = torch.randn_like(Q)  # needed for backward pass, this is where we'll be storing gradients

    # attention mask: upper-triangular matrix of 1s, rest 0s
    attention_mask = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))  # casual mask [S, S]

    # [B, N, S, D] x [B, N, D, S] => [B, N, S, S]
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale  # (QK^T)/(sqrt(D))
    if causal:
        # set the [S, S] dimension of P to -inf wherever the attention mask is 0 (upper-tril)
        P[:, :, attention_mask == 0] = float("-inf")  # [B, N, S, S]
    P = torch.softmax(P.float(), dim=-1).half()  
    ref_O = torch.matmul(P, V)  # end result is P @ V = softmax((QK^T)/sqrt(D)) @ V  (i.e. attention)
    ref_O.backward(d0)
    # ref_dX = reference backpropagation grads from naive algorithm, then set original grads to None 
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # Triton implementation call 
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)  # backward propagation, save in dO
    tri_dV, V.grad = V.grad.clone(), None 
    tri_dK, K.grad = K.grad.clone(), None 
    tri_dQ, Q.grad = Q.grad.clone(), None 

    # compare results 
    rtol = 0.0
    atol = 1e-3  # error margin, absolute distance 
    # asserting elementwise rough equality:
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)

    
