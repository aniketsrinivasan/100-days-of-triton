import torch 

import triton 
import triton.language as tl 


# a Triton kernel can be called from within another Triton kernel (just like CUDA)
@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # note:
    #   stages 1, 2, and 3 are handled separately for optimization purposes.
    #   stage 1: non-causal attention (this function computes attention across all query and key blocks)
    #   stage 2: diagonal-only attention
    #   stage 3: (strictly) lower-triangular attention
    # the separation of causal attention into stage 2 and 3 allows us to optimize using the pipelining that Triton does
    # it also allows the handling of diagonal blocks independently (where some elements need masking and others don't)

    # range of values handled by this stage
    if STAGE == 1:
        # from 0 to the (strict) left of the diagonal (bottom triangular)
        #   i.e. queries only attend to key blocks that come BEFORE it 
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q  
    elif STAGE == 2:
        # used only for the block in which there is transition between non-masked and masked keys 
        #   i.e. queries along the diagonal 
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q 
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)  # let compiler know that lo is a multiple of BLOCK_SIZE_Q (optimizations)
    else:
        # only used for non-causal attention; from 0 to SEQ_LEN (full, no masking)
        lo, hi = 0, SEQ_LEN
    
    # move pointers in corresponding dimensions (to starting locations, based on lo):
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))  # iterate our V block              [HEAD_DIM, SEQ_LEN]
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))  # iterate our K block (transposed) [SEQ_LEN, HEAD_DIM]

    # loop over K, V, and update accumulator 
    for start_kv in range(lo, hi, BLOCK_SIZE_KV): 
        # let the compiler know that start_kv is a multiple of BLOCK_SIZE_KV for compiler optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
        # compute Q_i @ K_i, where Q_i has already been loaded into SRAM, and K_i already transposed 
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)  # dot-product of blocks 

        if STAGE == 2:
            # on the diagonal (so we need to re-mask element-wise in the block)
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])  # causal mask needs to be applied on diagonal 
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))  # avoids exponential overflow (see FlashAttention algorithm)
            QK_block -= m_ij[:, None]
        else:
            # compute the maximum value of QK or keep the old maximum value 
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # compute the exponential of each dot product, so now we are computing exp(QK_ij - m_ij)
        P_block = tl.math.exp(QK_block)

        # compute sum by rows of the attention scores (eventually normalization factor)
        l_ij = tl.sum(P_block, 1)
        alpha = tl.math.exp(m_i - m_ij)  # correction factor exp(prev_max - curr_max)

        # apply correction factor to previous l_i and add the new l_ij 
        l_i = l_i * alpha + l_ij 

        # load the V_block
        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)  # type conversion

        # O_new = P @ V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)  # O_block += P_block @ V_block (O_block is an accumulator)

        # update current iteration maximum estimation
        m_i = m_ij 
        # advance both pointers in HEAD_DIM by BLOCK_SIZE_KV for next iteration (compute next block)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))  # [HEAD_DIM, SEQ_LEN]
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))  # [SEQ_LEN, HEAD_DIM]
    
    return O_block, l_i, m_i 


# a Triton kernel is just a Python method with the triton.jit decorator:
@triton.jit 
def _attn_fwd(
    Q,  # [B, N, S, D], pointer to first element
    K,  # [B, N, S, D], pointer to first element 
    V,  # [B, N, S, D], pointer to first element 
    softmax_scale,
    M,  # [B, N, S],    pointer to first element 
    O,  # [B, N, S, D], pointer to first element 
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    # verify information:
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)  # for auto-tuning

    # launch a grid (series of programs) where each program has an identifier 
    #   we launch(ed):
    #       SEQ_LEN / BLOCK_SIZE_Q  programs in axis=0
    #       BATCH_SIZE * NUM_HEADS  programs in axis=1
    #   of the launch grid, when calling this function. 
    # (1) indicate which block in the sequence length to process (axis=0)
    block_index_q = tl.program_id(0)
    # (2) batch and head that the program is associated with 
    index_batch_head = tl.program_id(1)

    index_batch = index_batch_head // NUM_HEADS 
    index_head = index_batch_head % NUM_HEADS 

    # want to enter tensor at Q[index_batch, index_head, :, :] (right location for this program)
    #   => we have to generate some offset to start our current program at 
    qkv_offset = (
        index_batch.to(tl.int64) * stride_Q_batch,  # batch_idx * batch_stride = start of current batch
        + index_head.to(tl.int64) * stride_Q_head   # head_idx * head_stride = start of current head
    )

    # Triton provides a generalized way to index into tensors (tl.make_block_ptr)
    #   here we treat the parent tensor as Q split into batches and heads 
    #   the offset(s) to the block will be identified by the block index and BLOCK_SIZE_Q 
    # ...
    Q_block_ptr = tl.make_block_ptr(  # Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q :, :]
        base = Q + qkv_offset,  # start pointer of Q[index_batch, index_head, :, :]
        shape = (SEQ_LEN, HEAD_DIM),  # tensor has shape [S, D] as these are selected above (base)
        strides = (stride_Q_seq, stride_Q_dim),  # strides for indexing [S, D] dimensions
        offsets = (block_index_q * BLOCK_SIZE_Q, 0),  # select a block of queries within [S, D]
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0)  # order of the original data format (for compiler optimizations)
    )
    V_block_ptr = tl.make_block_ptr(  # V[index_batch, index_head, :, :]
        base = V + qkv_offset, 
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_V_seq, stride_V_dim), 
        offsets = (0, 0),  # all queries across all [S, D]
        block_shape = (BLOCK_SIZE_KV, HEAD_DIM),
        order = (1, 0)
    )
    K_block_ptr = tl.make_block_ptr(  # K[index_batch, index_head, :, :] transposed 
        base = K + qkv_offset,
        shape = (HEAD_DIM, SEQ_LEN),  # transposed => [B, N, D, S]
        strides = (stride_K_dim, stride_K_seq),  # transposed strides [D, S] dimensions
        offsets = (0, 0),
        block_shape = (HEAD_DIM, BLOCK_SIZE_KV),
        order = (0, 1)  # transposed compiler optimization
    )
    # the above program will generate exactly ONE block of the output matrix, so we must create a pointer that identifies
    # which block of the output matrix this program will create (using a block_ptr):
    #   used later to write to the output O in the correct block 
    O_block_ptr = tl.make_block_ptr(  # O[index_batch, index_head, block_index_q * BLOCK_SIZE_Q :, :]
        base = O + qkv_offset, 
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_O_seq, stride_O_dim),
        offsets = (block_index_q * BLOCK_SIZE_Q, 0),
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0)
    )

    # offs_q: the offsets for the tokens in Q to process
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    # offs_kv: the offsets for the tokens in K and V sequence to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)  # we don't skip anything since we need to iterate through K and V 

    # m_i: the running maximum; one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")  # m_i = -inf  for each query in BLOCK_SIZE_Q

    # l_i: the running sum; one for each query (as we sum the attention scores row-wise)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0  # add 1.0 for logsumexp stability 

    # O_block: the accumulator for the output, which is a group of rows of the O tensor 
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)  # one block of the output tensor (one block of rows)

    # load Q_block from HBM to SRAM (shared memory)
    Q_block = tl.load(Q_block_ptr)

    # we split into two steps:
    #   first, iterate through all keys and values for which index < current query block (causal and non-causal)
    #   (left of diagonal)
    if STAGE == 1 or STAGE == 3:
        # this step runs for non-causal attention OR for blocks (strictly) to the left of the diagonal in causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,  # masking stage (1: causal, 3: non-causal, since this is 4 - STAGE)
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    #   for all the elements on the right (k_index > q_index)
    #   (right of diagonal)
    if STAGE == 3:
        # for causal attention, also compute along the diagonal (independently of previous step)
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,          # masking stage (2: diagonal)
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    # needed to compute the logsumexp for the backwards pass 
    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q  # skip based on what this program is working with
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)  # which batch and head we are working with
    offs_dim = tl.arange(0, HEAD_DIM)  # don't divide on the head_dim dimension; only on seq_len dimension

    # in this function, we don't use make_block_ptr and instead use direct indexing as a programming exercise
    #   load single block of BLOCK_SIZE_Q rows of O
    O_block = tl.load(
        O  # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM  # select row offsets
        + offs_dim[None, :]  # select column offsets
    )  # [BLOCK_SIZE_Q, HEAD_DIM]

    #   load single block of BLOCK_SIZE_Q rows of dO
    dO_block = tl.load(
        dO 
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM  # select row offsets
        + offs_dim[None, :]  # select column offsets
    ).to(tl.float32)  # [BLOCK_SIZE_Q, HEAD_DIM]

    # compute the D block
    D_block = tl.sum(dO_block * O_block, axis=1)  # element-wise multiplication
    # store D block
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q 
    tl.store(D_block_ptrs, D_block)


@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr
):
    # get relevant indices and offset(s):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS 
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    # this offset allows us to select the right sequene given the batch and head
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # adjust the pointers to the right place w.r.t. batch and head 
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # move M and D to the right batch, head, and sequence 
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales 
    offs_dim = tl.arange(0, HEAD_DIM)  # ranges in the 2nd dimension of each K, V to load
    index_block_kv = tl.program_id(0)  # which K, V block this program works with
    start_kv = index_block_kv * BLOCK_KV 
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    # define 2D-tensors stored in SRAM
    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # load K and V. they stay in SRAM throughout the inner loop.
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # [BLOCK_KV, HEAD_DIM]
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # [BLOCK_KV, HEAD_DIM]

    offs_q = tl.arange(0, BLOCK_Q)

    # we access Q as a transposed array since we need (P^T) eventually
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    # iterate over the sequence dimension of the query
    curr_q = 0  # up to SEQ_LEN
    num_steps = SEQ_LEN // BLOCK_Q 
    for block_idx in range(num_steps):
        # load a block of Q 
        qT_block = tl.load(qT_ptrs)
        # load the logsumexp values for the queries in the current block
        offs_q = curr_q + tl.arange(0, BLOCK_Q)  # offsets of elements we need from M tensor
        m = tl.load(M + offs_q)

        # compute (QK^T)^T = (K^T)^T Q^T = KQ^T = S^T
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        # apply the softmax using the logsumexp trick (to get P^T block)
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        if STAGE == 3:
            # autoregressive masking (causal modeling)
            # [BLOCK_KV, BLOCK_Q]
            mask_block = (offs_q[None, :] >= offs_kv[:, None])  # mask is TRUE for all values that don't need masking
            P_T_block = tl.where(mask_block, P_T_block, 0.0)  # all other values set to 0.0
        
        dO_block = tl.load(dO_ptrs)
        # dV_new = dV_old + P^T @ dO according to the paper 
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        # delta = rowsum(O * dO) where * is element-wise product
        Di = tl.load(D + offs_q)

        # dP = dO @ V^T so dP^T = V @ dO^T
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        # dS = P * (dP - delta) so transposing gives dS^T = P^T * (dP^T - delta^T) (element-wise)
        dS_T_block = (P_T_block * (dpT_block - Di[None, :])).to(tl.float16)

        # dK_new = dK_old + dS^T @ Q
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))

        # increment pointers for next block
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq 
        dO_ptrs += BLOCK_Q * stride_seq 
