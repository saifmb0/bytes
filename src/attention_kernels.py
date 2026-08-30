import torch
import triton
import triton.language as tl

# =====================================================================
# 1. BF16 Attention Kernel
# =====================================================================
@triton.jit
def _decode_attn_bf16_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_d,
    num_heads_q, num_heads_k, G,
    context_len, head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // num_heads_q
    h_q = pid % num_heads_q
    h_k = h_q // G
    
    # Load Q: [head_dim]
    q_cols = tl.arange(0, head_dim)
    q_ptr = Q_ptr + b * stride_q_b + h_q * stride_q_h + q_cols * stride_q_d
    q = tl.load(q_ptr).to(tl.float32) # Cast to float32
    
    max_val = -float('inf')
    sum_val = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    for start_n in range(0, context_len, BLOCK_N):
        mask_n = (start_n + tl.arange(0, BLOCK_N)) < context_len
        
        # Load K: [BLOCK_N, head_dim]
        k_offsets = (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_k_s + q_cols[None, :] * stride_k_d
        k_ptr_block = K_ptr + b * stride_k_b + h_k * stride_k_h + k_offsets
        k = tl.load(k_ptr_block, mask=mask_n[:, None], other=0.0).to(tl.float32) # Cast to float32
        
        # Compute dot product in float32
        scores = tl.sum(q[None, :] * k, axis=1)
        scores = tl.where(mask_n, scores, -float('inf'))
        
        # Online softmax
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_val, block_max)
        alpha = tl.math.exp(max_val - new_max)
        beta = tl.math.exp(scores - new_max)
        
        # Load V: [BLOCK_N, head_dim]
        v_offsets = (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_v_s + q_cols[None, :] * stride_v_d
        v_ptr_block = V_ptr + b * stride_v_b + h_k * stride_v_h + v_offsets
        v = tl.load(v_ptr_block, mask=mask_n[:, None], other=0.0).to(tl.float32) # Cast to float32
        
        # Accumulate
        acc = acc * alpha + tl.sum(beta[:, None] * v, axis=0)
        sum_val = sum_val * alpha + tl.sum(beta, axis=0)
        max_val = new_max
        
    acc = acc / sum_val
    out_ptr = O_ptr + b * stride_o_b + h_q * stride_o_h + q_cols * stride_o_d
    tl.store(out_ptr, acc.to(O_ptr.dtype.element_ty))

def decode_attn_bf16(q, k, v):
    """
    q: [B, H_q, 1, D]
    k: [B, H_k, S, D]
    v: [B, H_k, S, D]
    """
    B, H_q, _, D = q.shape
    _, H_k, S, _ = k.shape
    G = H_q // H_k
    
    # Squeeze the singleton query dimension
    q_squeezed = q.squeeze(2) # [B, H_q, D]
    out = torch.empty_like(q_squeezed)
    
    BLOCK_N = 64
    grid = (B * H_q,)
    
    _decode_attn_bf16_kernel[grid](
        q_squeezed, k, v, out,
        q_squeezed.stride(0), q_squeezed.stride(1), q_squeezed.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        H_q, H_k, G, S, D,
        BLOCK_N=BLOCK_N,
        num_warps=4
    )
    return out.unsqueeze(2)

# =====================================================================
# 2. FP8 Attention Kernel
# =====================================================================
@triton.jit
def _decode_attn_fp8_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_d,
    num_heads_q, num_heads_k, G,
    context_len, head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // num_heads_q
    h_q = pid % num_heads_q
    h_k = h_q // G
    
    # Load Q: [head_dim]
    q_cols = tl.arange(0, head_dim)
    q_ptr = Q_ptr + b * stride_q_b + h_q * stride_q_h + q_cols * stride_q_d
    q = tl.load(q_ptr).to(tl.float32) # Cast to float32
    
    max_val = -float('inf')
    sum_val = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    for start_n in range(0, context_len, BLOCK_N):
        mask_n = (start_n + tl.arange(0, BLOCK_N)) < context_len
        
        # Load FP8 K: [BLOCK_N, head_dim] and cast to float32
        k_offsets = (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_k_s + q_cols[None, :] * stride_k_d
        k_ptr_block = K_ptr + b * stride_k_b + h_k * stride_k_h + k_offsets
        k_fp8 = tl.load(k_ptr_block, mask=mask_n[:, None], other=0.0)
        k = k_fp8.to(tl.float32) # Cast in registers
        
        # Compute dot product
        scores = tl.sum(q[None, :] * k, axis=1)
        scores = tl.where(mask_n, scores, -float('inf'))
        
        # Online softmax
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_val, block_max)
        alpha = tl.math.exp(max_val - new_max)
        beta = tl.math.exp(scores - new_max)
        
        # Load FP8 V: [BLOCK_N, head_dim] and cast to float32
        v_offsets = (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_v_s + q_cols[None, :] * stride_v_d
        v_ptr_block = V_ptr + b * stride_v_b + h_k * stride_v_h + v_offsets
        v_fp8 = tl.load(v_ptr_block, mask=mask_n[:, None], other=0.0)
        v = v_fp8.to(tl.float32) # Cast in registers
        
        # Accumulate
        acc = acc * alpha + tl.sum(beta[:, None] * v, axis=0)
        sum_val = sum_val * alpha + tl.sum(beta, axis=0)
        max_val = new_max
        
    acc = acc / sum_val
    out_ptr = O_ptr + b * stride_o_b + h_q * stride_o_h + q_cols * stride_o_d
    tl.store(out_ptr, acc.to(O_ptr.dtype.element_ty))

def decode_attn_fp8(q, k, v):
    """
    q: [B, H_q, 1, D] (bf16)
    k: [B, H_k, S, D] (fp8 - e4m3fn)
    v: [B, H_k, S, D] (fp8 - e4m3fn)
    """
    B, H_q, _, D = q.shape
    _, H_k, S, _ = k.shape
    G = H_q // H_k
    
    q_squeezed = q.squeeze(2) # [B, H_q, D]
    out = torch.empty_like(q_squeezed)
    
    BLOCK_N = 64
    grid = (B * H_q,)
    
    _decode_attn_fp8_kernel[grid](
        q_squeezed, k, v, out,
        q_squeezed.stride(0), q_squeezed.stride(1), q_squeezed.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        H_q, H_k, G, S, D,
        BLOCK_N=BLOCK_N,
        num_warps=4
    )
    return out.unsqueeze(2)

# =====================================================================
# 3. Dequantized low-bit Attention Kernel (MSE style / registers dequant)
# =====================================================================
@triton.jit
def _decode_attn_dequant_kernel(
    Q_ptr, K_idx_ptr, Centroids_ptr, V_ptr, O_ptr,
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_m,
    stride_c_m, stride_c_k, stride_c_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_d,
    num_heads_q, num_heads_k, G,
    context_len, head_dim: tl.constexpr,
    M: tl.constexpr, d_sub: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // num_heads_q
    h_q = pid % num_heads_q
    h_k = h_q // G
    
    # Load Q: [head_dim]
    q_cols = tl.arange(0, head_dim)
    q_ptr = Q_ptr + b * stride_q_b + h_q * stride_q_h + q_cols * stride_q_d
    q = tl.load(q_ptr).to(tl.float32) # Cast to float32
    
    max_val = -float('inf')
    sum_val = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    # Subvector mapping helper
    sub_col_idx = q_cols % d_sub
    
    for start_n in range(0, context_len, BLOCK_N):
        mask_n = (start_n + tl.arange(0, BLOCK_N)) < context_len
        
        # Reconstruct key block: [BLOCK_N, head_dim] in registers
        k = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)
        
        for j in range(M):
            # Load quantized indices for subvector j: [BLOCK_N]
            idx_offsets = (start_n + tl.arange(0, BLOCK_N)) * stride_k_s + j * stride_k_m
            idx_ptr = K_idx_ptr + b * stride_k_b + h_k * stride_k_h + idx_offsets
            idx_j = tl.load(idx_ptr, mask=mask_n, other=0) # [BLOCK_N]
            
            # Formulate vectorized load from Centroids table: [BLOCK_N, head_dim]
            # Select columns corresponding to subvector j
            col_mask = (q_cols >= j * d_sub) & (q_cols < (j + 1) * d_sub)
            
            centroid_offsets = (
                j * stride_c_m + 
                idx_j[:, None] * stride_c_k + 
                sub_col_idx[None, :] * stride_c_d
            )
            
            # Vectorized load: only active columns are read, rest are 0.0
            sub_k = tl.load(Centroids_ptr + centroid_offsets, mask=col_mask[None, :] & mask_n[:, None], other=0.0).to(tl.float32)
            k += sub_k
            
        # Compute dot product
        scores = tl.sum(q[None, :] * k, axis=1)
        scores = tl.where(mask_n, scores, -float('inf'))
        
        # Online softmax
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_val, block_max)
        alpha = tl.math.exp(max_val - new_max)
        beta = tl.math.exp(scores - new_max)
        
        # Load V: [BLOCK_N, head_dim]
        v_offsets = (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_v_s + q_cols[None, :] * stride_v_d
        v_ptr_block = V_ptr + b * stride_v_b + h_k * stride_v_h + v_offsets
        v = tl.load(v_ptr_block, mask=mask_n[:, None], other=0.0).to(tl.float32)
        
        # Accumulate
        acc = acc * alpha + tl.sum(beta[:, None] * v, axis=0)
        sum_val = sum_val * alpha + tl.sum(beta, axis=0)
        max_val = new_max
        
    acc = acc / sum_val
    out_ptr = O_ptr + b * stride_o_b + h_q * stride_o_h + q_cols * stride_o_d
    tl.store(out_ptr, acc.to(O_ptr.dtype.element_ty))

def decode_attn_dequant(q, k_idx, centroids, v, d_sub=8):
    """
    q: [B, H_q, 1, D]
    k_idx: [B, H_k, S, M] (uint8/int64 indices)
    centroids: [M, K, d_sub] (float32 codebooks)
    v: [B, H_k, S, D]
    """
    B, H_q, _, D = q.shape
    _, H_k, S, M = k_idx.shape
    G = H_q // H_k
    
    q_squeezed = q.squeeze(2) # [B, H_q, D]
    out = torch.empty_like(q_squeezed)
    
    BLOCK_N = 64
    grid = (B * H_q,)
    
    _decode_attn_dequant_kernel[grid](
        q_squeezed, k_idx, centroids, v, out,
        q_squeezed.stride(0), q_squeezed.stride(1), q_squeezed.stride(2),
        k_idx.stride(0), k_idx.stride(1), k_idx.stride(2), k_idx.stride(3),
        centroids.stride(0), centroids.stride(1), centroids.stride(2),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        H_q, H_k, G, S, D,
        M=M, d_sub=d_sub,
        BLOCK_N=BLOCK_N,
        num_warps=4
    )
    return out.unsqueeze(2)

# =====================================================================
# 4. PQ-LUT (Query-Rotated Lookup Attention) Kernel
# =====================================================================
@triton.jit
def _decode_attn_pq_lut_kernel(
    LUT_ptr, K_idx_ptr, V_ptr, O_ptr,
    stride_l_b, stride_l_h, stride_l_m, stride_l_k,
    stride_k_b, stride_k_h, stride_k_s, stride_k_m,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_d,
    num_heads_q, num_heads_k, G,
    context_len, head_dim: tl.constexpr,
    M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // num_heads_q
    h_q = pid % num_heads_q
    h_k = h_q // G
    
    # Initialize online softmax accumulators
    max_val = -float('inf')
    sum_val = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    q_cols = tl.arange(0, head_dim)
    
    for start_n in range(0, context_len, BLOCK_N):
        mask_n = (start_n + tl.arange(0, BLOCK_N)) < context_len
        
        # Dequantization-free inner product scoring using lookup table
        scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        
        for j in range(M):
            # Load quantized key indices: [BLOCK_N]
            idx_offsets = (start_n + tl.arange(0, BLOCK_N)) * stride_k_s + j * stride_k_m
            idx_ptr = K_idx_ptr + b * stride_k_b + h_k * stride_k_h + idx_offsets
            idx_j = tl.load(idx_ptr, mask=mask_n, other=0) # [BLOCK_N]
            
            # Lookup attention score values from LUT: [BLOCK_N]
            # LUT is shape [B, num_heads_q, M, K]
            lut_offset = b * stride_l_b + h_q * stride_l_h + j * stride_l_m + idx_j * stride_l_k
            val = tl.load(LUT_ptr + lut_offset, mask=mask_n, other=0.0).to(tl.float32)
            scores += val
            
        scores = tl.where(mask_n, scores, -float('inf'))
        
        # Online softmax
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_val, block_max)
        alpha = tl.math.exp(max_val - new_max)
        beta = tl.math.exp(scores - new_max)
        
        # Load V: [BLOCK_N, head_dim]
        v_offsets = (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_v_s + q_cols[None, :] * stride_v_d
        v_ptr_block = V_ptr + b * stride_v_b + h_k * stride_v_h + v_offsets
        v = tl.load(v_ptr_block, mask=mask_n[:, None], other=0.0).to(tl.float32)
        
        # Accumulate
        acc = acc * alpha + tl.sum(beta[:, None] * v, axis=0)
        sum_val = sum_val * alpha + tl.sum(beta, axis=0)
        max_val = new_max
        
    acc = acc / sum_val
    out_ptr = O_ptr + b * stride_o_b + h_q * stride_o_h + q_cols * stride_o_d
    tl.store(out_ptr, acc.to(O_ptr.dtype.element_ty))

def decode_attn_pq_lut(lut, k_idx, v):
    """
    lut: [B, H_q, M, K] (Precomputed offline: LUT[b, h, j, c] = q_rot[b, h, j]^T * centroids[j, c])
    k_idx: [B, H_k, S, M]
    v: [B, H_k, S, D]
    """
    B, H_q, _, _ = lut.shape
    _, H_k, S, M = k_idx.shape
    _, _, _, D = v.shape
    G = H_q // H_k
    
    out = torch.empty((B, H_q, D), dtype=v.dtype, device=v.device)
    
    BLOCK_N = 64
    grid = (B * H_q,)
    
    _decode_attn_pq_lut_kernel[grid](
        lut, k_idx, v, out,
        lut.stride(0), lut.stride(1), lut.stride(2), lut.stride(3),
        k_idx.stride(0), k_idx.stride(1), k_idx.stride(2), k_idx.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        H_q, H_k, G, S, D,
        M=M,
        BLOCK_N=BLOCK_N,
        num_warps=4
    )
    return out.unsqueeze(2)

# =====================================================================
# 5. PQ-LUT Joint KV Quantized Attention Kernel (Quantized K & Quantized V)
# =====================================================================
@triton.jit
def _decode_attn_pq_lut_kv_kernel(
    LUT_ptr, K_idx_ptr, V_idx_ptr, V_Centroids_ptr, O_ptr,
    stride_l_b, stride_l_h, stride_l_m, stride_l_k,
    stride_k_b, stride_k_h, stride_k_s, stride_k_m,
    stride_v_b, stride_v_h, stride_v_s, stride_v_m,
    stride_vc_m, stride_vc_k, stride_vc_d,
    stride_o_b, stride_o_h, stride_o_d,
    num_heads_q, num_heads_k, G,
    context_len, head_dim: tl.constexpr,
    M_k: tl.constexpr,
    M_v: tl.constexpr, d_sub_v: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // num_heads_q
    h_q = pid % num_heads_q
    h_k = h_q // G
    
    # Initialize online softmax accumulators
    max_val = -float('inf')
    sum_val = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    q_cols = tl.arange(0, head_dim)
    sub_col_idx_v = q_cols % d_sub_v
    
    for start_n in range(0, context_len, BLOCK_N):
        mask_n = (start_n + tl.arange(0, BLOCK_N)) < context_len
        
        # Dequantization-free inner product scoring using lookup table: [BLOCK_N]
        scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for j in range(M_k):
            idx_offsets = (start_n + tl.arange(0, BLOCK_N)) * stride_k_s + j * stride_k_m
            idx_ptr = K_idx_ptr + b * stride_k_b + h_k * stride_k_h + idx_offsets
            idx_j = tl.load(idx_ptr, mask=mask_n, other=0) # [BLOCK_N]
            
            lut_offset = b * stride_l_b + h_q * stride_l_h + j * stride_l_m + idx_j * stride_l_k
            val = tl.load(LUT_ptr + lut_offset, mask=mask_n, other=0.0).to(tl.float32)
            scores += val
            
        scores = tl.where(mask_n, scores, -float('inf'))
        
        # Online softmax
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_val, block_max)
        alpha = tl.math.exp(max_val - new_max)
        beta = tl.math.exp(scores - new_max)
        
        # Reconstruct Value cache block in registers: [BLOCK_N, head_dim]
        v = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)
        for j in range(M_v):
            v_idx_offsets = (start_n + tl.arange(0, BLOCK_N)) * stride_v_s + j * stride_v_m
            v_idx_ptr = V_idx_ptr + b * stride_v_b + h_k * stride_v_h + v_idx_offsets
            v_idx_j = tl.load(v_idx_ptr, mask=mask_n, other=0) # [BLOCK_N]
            
            col_mask_v = (q_cols >= j * d_sub_v) & (q_cols < (j + 1) * d_sub_v)
            v_centroid_offsets = (
                j * stride_vc_m + 
                v_idx_j[:, None] * stride_vc_k + 
                sub_col_idx_v[None, :] * stride_vc_d
            )
            sub_v = tl.load(V_Centroids_ptr + v_centroid_offsets, mask=col_mask_v[None, :] & mask_n[:, None], other=0.0).to(tl.float32)
            v += sub_v
            
        # Accumulate
        acc = acc * alpha + tl.sum(beta[:, None] * v, axis=0)
        sum_val = sum_val * alpha + tl.sum(beta, axis=0)
        max_val = new_max
        
    acc = acc / sum_val
    out_ptr = O_ptr + b * stride_o_b + h_q * stride_o_h + q_cols * stride_o_d
    tl.store(out_ptr, acc.to(O_ptr.dtype.element_ty))

def decode_attn_pq_lut_kv(lut, k_idx, v_idx, v_centroids, d_sub_v=8):
    """
    lut: [B, H_q, M_k, K]
    k_idx: [B, H_k, S, M_k]
    v_idx: [B, H_k, S, M_v]
    v_centroids: [M_v, K_v, d_sub_v]
    """
    B, H_q, M_k, K_k = lut.shape
    _, H_k, S, M_v = v_idx.shape
    M_v, K_v, d_sub_v = v_centroids.shape
    D = M_v * d_sub_v
    G = H_q // H_k
    
    out = torch.empty((B, H_q, D), dtype=v_centroids.dtype, device=v_centroids.device)
    BLOCK_N = 64
    grid = (B * H_q,)
    
    _decode_attn_pq_lut_kv_kernel[grid](
        lut, k_idx, v_idx, v_centroids, out,
        lut.stride(0), lut.stride(1), lut.stride(2), lut.stride(3),
        k_idx.stride(0), k_idx.stride(1), k_idx.stride(2), k_idx.stride(3),
        v_idx.stride(0), v_idx.stride(1), v_idx.stride(2), v_idx.stride(3),
        v_centroids.stride(0), v_centroids.stride(1), v_centroids.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        H_q, H_k, G, S, D,
        M_k=M_k, M_v=M_v, d_sub_v=d_sub_v,
        BLOCK_N=BLOCK_N,
        num_warps=4
    )
    return out.unsqueeze(2)

# =====================================================================
# 6. Paged PQ-LUT Attention Kernel (Block-Based Paged Layout)
# =====================================================================
@triton.jit
def _paged_attn_pq_lut_kv_kernel(
    LUT_ptr, K_idx_ptr, V_idx_ptr, V_Centroids_ptr, Block_Table_ptr, Context_Lens_ptr, O_ptr,
    stride_l_b, stride_l_h, stride_l_m, stride_l_k,
    stride_k_b, stride_k_h, stride_k_s, stride_k_m,
    stride_v_b, stride_v_h, stride_v_s, stride_v_m,
    stride_vc_m, stride_vc_k, stride_vc_d,
    stride_bt_b, stride_bt_s,
    stride_o_b, stride_o_h, stride_o_d,
    num_heads_q, num_heads_k, G,
    head_dim: tl.constexpr,
    M_k: tl.constexpr,
    M_v: tl.constexpr, d_sub_v: tl.constexpr,
    block_size: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // num_heads_q
    h_q = pid % num_heads_q
    h_k = h_q // G
    
    # Load sequence length for this batch element
    context_len = tl.load(Context_Lens_ptr + b)
    
    # Initialize online softmax accumulators
    max_val = -float('inf')
    sum_val = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    q_cols = tl.arange(0, head_dim)
    sub_col_idx_v = q_cols % d_sub_v
    
    # Loop over blocks of block_size
    for start_n in range(0, context_len, block_size):
        # Fetch physical block ID from table
        block_idx = start_n // block_size
        physical_block_id = tl.load(Block_Table_ptr + b * stride_bt_b + block_idx * stride_bt_s)
        
        mask_n = (start_n + tl.arange(0, block_size)) < context_len
        
        # 1. Key lookup scoring (LUT gather-sum)
        scores = tl.zeros([block_size], dtype=tl.float32)
        for j in range(M_k):
            idx_offsets = tl.arange(0, block_size) * stride_k_s + j * stride_k_m
            idx_ptr = K_idx_ptr + physical_block_id * stride_k_b + h_k * stride_k_h + idx_offsets
            idx_j = tl.load(idx_ptr, mask=mask_n, other=0) # [block_size]
            
            lut_offset = b * stride_l_b + h_q * stride_l_h + j * stride_l_m + idx_j * stride_l_k
            val = tl.load(LUT_ptr + lut_offset, mask=mask_n, other=0.0).to(tl.float32)
            scores += val
            
        scores = tl.where(mask_n, scores, -float('inf'))
        
        # Online softmax
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_val, block_max)
        alpha = tl.math.exp(max_val - new_max)
        beta = tl.math.exp(scores - new_max)
        
        # 2. Value reconstruction in registers
        v = tl.zeros([block_size, head_dim], dtype=tl.float32)
        for j in range(M_v):
            v_idx_offsets = tl.arange(0, block_size) * stride_v_s + j * stride_v_m
            v_idx_ptr = V_idx_ptr + physical_block_id * stride_v_b + h_k * stride_v_h + v_idx_offsets
            v_idx_j = tl.load(v_idx_ptr, mask=mask_n, other=0) # [block_size]
            
            col_mask_v = (q_cols >= j * d_sub_v) & (q_cols < (j + 1) * d_sub_v)
            v_centroid_offsets = (
                j * stride_vc_m + 
                v_idx_j[:, None] * stride_vc_k + 
                sub_col_idx_v[None, :] * stride_vc_d
            )
            sub_v = tl.load(V_Centroids_ptr + v_centroid_offsets, mask=col_mask_v[None, :] & mask_n[:, None], other=0.0).to(tl.float32)
            v += sub_v
            
        # Accumulate
        acc = acc * alpha + tl.sum(beta[:, None] * v, axis=0)
        sum_val = sum_val * alpha + tl.sum(beta, axis=0)
        max_val = new_max
        
    acc = acc / sum_val
    out_ptr = O_ptr + b * stride_o_b + h_q * stride_o_h + q_cols * stride_o_d
    tl.store(out_ptr, acc.to(O_ptr.dtype.element_ty))

def paged_attn_pq_lut_kv(lut, k_idx_paged, v_idx_paged, v_centroids, block_table, context_lens, block_size=16):
    """
    lut: [B, H_q, M_k, K_k]
    k_idx_paged: [num_blocks, H_k, block_size, M_k]
    v_idx_paged: [num_blocks, H_k, block_size, M_v]
    v_centroids: [M_v, K_v, d_sub_v]
    block_table: [B, max_blocks]
    context_lens: [B] (int32 sequence lengths)
    """
    B, H_q, M_k, K_k = lut.shape
    _, H_k, _, M_v = v_idx_paged.shape
    M_v, K_v, d_sub_v = v_centroids.shape
    D = M_v * d_sub_v
    G = H_q // H_k
    
    out = torch.empty((B, H_q, D), dtype=v_centroids.dtype, device=v_centroids.device)
    grid = (B * H_q,)
    
    _paged_attn_pq_lut_kv_kernel[grid](
        lut, k_idx_paged, v_idx_paged, v_centroids, block_table, context_lens, out,
        lut.stride(0), lut.stride(1), lut.stride(2), lut.stride(3),
        k_idx_paged.stride(0), k_idx_paged.stride(1), k_idx_paged.stride(2), k_idx_paged.stride(3),
        v_idx_paged.stride(0), v_idx_paged.stride(1), v_idx_paged.stride(2), v_idx_paged.stride(3),
        v_centroids.stride(0), v_centroids.stride(1), v_centroids.stride(2),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        H_q, H_k, G, D,
        M_k=M_k, M_v=M_v, d_sub_v=d_sub_v,
        block_size=block_size,
        num_warps=4
    )
    return out.unsqueeze(2)


# =====================================================================
# 7. Asymmetric PQ-LUT Attention Kernel (Outlier-Aware + Native FP8/BF16 Value Cache)
# =====================================================================
@triton.jit
def _decode_attn_pq_lut_asym_kernel(
    LUT_ptr, K_idx_ptr, K_out_ptr, Q_out_ptr, V_ptr, O_ptr,
    stride_l_b, stride_l_h, stride_l_m, stride_l_k,
    stride_k_b, stride_k_h, stride_k_s, stride_k_m,
    stride_ko_b, stride_ko_h, stride_ko_s, stride_ko_c,
    stride_qo_b, stride_qo_h, stride_qo_c,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_d,
    num_heads_q, num_heads_k, G,
    context_len, head_dim: tl.constexpr,
    M_k: tl.constexpr, C_out: tl.constexpr, C_out_padded: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // num_heads_q
    h_q = pid % num_heads_q
    h_k = h_q // G

    # Initialize online softmax accumulators
    max_val = -float('inf')
    sum_val = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    q_cols = tl.arange(0, head_dim)

    # Precompute pointers and values that do not depend on start_n
    lut_head_ptr = LUT_ptr + b * stride_l_b + h_q * stride_l_h
    
    k_idx_base_ptr = K_idx_ptr + b * stride_k_b + h_k * stride_k_h
    k_idx_sub_offsets = tl.arange(0, M_k)[None, :]
    j_offsets = tl.arange(0, M_k)[None, :] * stride_l_m
    
    v_base_ptr = V_ptr + b * stride_v_b + h_k * stride_v_h
    v_col_offsets = q_cols[None, :]

    if C_out > 0:
        qo_offsets = tl.arange(0, C_out_padded)
        qo_mask = tl.arange(0, C_out_padded) < C_out
        qo_vals = tl.load(Q_out_ptr + b * stride_qo_b + h_q * stride_qo_h + qo_offsets, mask=qo_mask, other=0.0)
        
        k_out_base_ptr = K_out_ptr + b * stride_ko_b + h_k * stride_ko_h
        ko_col_offsets = tl.arange(0, C_out_padded)[None, :]
        c_out_mask = tl.arange(0, C_out_padded)[None, :] < C_out

    for start_n in range(0, context_len, BLOCK_N):
        mask_n = (start_n + tl.arange(0, BLOCK_N)) < context_len

        # 1. Path A: Dense scoring (LUT gathers)
        # Load all indices for the current block in one 2D load!
        k_idx_offsets = (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_k_s + k_idx_sub_offsets
        idx_block = tl.load(k_idx_base_ptr + k_idx_offsets, mask=mask_n[:, None], other=0) # [BLOCK_N, M_k]

        lut_ptrs = lut_head_ptr + j_offsets + idx_block.to(tl.int32)
        val_block = tl.load(lut_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(val_block, axis=1) # [BLOCK_N]

        # 2. Path B: Outlier scoring (Unquantized dot product)
        if C_out > 0:
            # Load all Key outliers for this block in one 2D load!
            ko_offsets = (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_ko_s + ko_col_offsets
            ko_mask = mask_n[:, None] & c_out_mask
            ko_vals = tl.load(k_out_base_ptr + ko_offsets, mask=ko_mask, other=0.0).to(tl.float32)
            
            outlier_scores = tl.sum(qo_vals[None, :] * ko_vals, axis=1)
            scores += outlier_scores

        scores = tl.where(mask_n, scores, -float('inf'))
        
        # Online softmax
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_val, block_max)
        alpha = tl.math.exp(max_val - new_max)
        beta = tl.math.exp(scores - new_max)
        
        # 3. Load V: [BLOCK_N, head_dim]
        v_offsets = (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_v_s + v_col_offsets * stride_v_d
        v = tl.load(v_base_ptr + v_offsets, mask=mask_n[:, None], other=0.0).to(tl.float32)
        
        # Accumulate
        acc = acc * alpha + tl.sum(beta[:, None] * v, axis=0)
        sum_val = sum_val * alpha + tl.sum(beta, axis=0)
        max_val = new_max
        
    acc = acc / sum_val
    out_ptr = O_ptr + b * stride_o_b + h_q * stride_o_h + q_cols * stride_o_d
    tl.store(out_ptr, acc.to(O_ptr.dtype.element_ty))

def decode_attn_pq_lut_asym(lut, k_idx, k_out, q_out, v, BLOCK_N=256):
    """
    lut: [B, H_q, M_k, K_k]
    k_idx: [B, H_k, S, M_k] (dense key indices)
    k_out: [B, H_k, S, C_out] (outlier key values)
    q_out: [B, H_q, C_out] (outlier query values)
    v: [B, H_k, S, D] (FP8 or BF16 values)
    """
    B, H_q, M_k, K_k = lut.shape
    _, H_k, S, _ = k_idx.shape
    _, _, _, D = v.shape
    C_out = k_out.shape[-1]
    G = H_q // H_k
    
    out = torch.empty((B, H_q, D), dtype=v.dtype if v.dtype != torch.float8_e4m3fn else torch.bfloat16, device=v.device)
    
    grid = (B * H_q,)
    
    C_out_padded = 1 << (C_out - 1).bit_length() if C_out > 0 else 0
    if C_out_padded < 1 and C_out > 0:
        C_out_padded = 1
    
    num_warps = 8 if BLOCK_N >= 128 else 4
    
    _decode_attn_pq_lut_asym_kernel[grid](
        lut, k_idx, k_out, q_out, v, out,
        lut.stride(0), lut.stride(1), lut.stride(2), lut.stride(3),
        k_idx.stride(0), k_idx.stride(1), k_idx.stride(2), k_idx.stride(3),
        k_out.stride(0), k_out.stride(1), k_out.stride(2), k_out.stride(3),
        q_out.stride(0), q_out.stride(1), q_out.stride(2),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        H_q, H_k, G, S, D,
        M_k=M_k, C_out=C_out, C_out_padded=C_out_padded,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps
    )
    return out.unsqueeze(2)


# =====================================================================
# 8. Paged Asymmetric PQ-LUT Attention Kernel
# =====================================================================
@triton.jit
def _paged_attn_pq_lut_asym_kernel(
    LUT_ptr, K_idx_ptr, K_out_ptr, Q_out_ptr, V_ptr, Block_Table_ptr, Context_Lens_ptr, O_ptr,
    stride_l_b, stride_l_h, stride_l_m, stride_l_k,
    stride_k_b, stride_k_h, stride_k_s, stride_k_m,
    stride_ko_b, stride_ko_h, stride_ko_s, stride_ko_c,
    stride_qo_b, stride_qo_h, stride_qo_c,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_bt_b, stride_bt_s,
    stride_o_b, stride_o_h, stride_o_d,
    num_heads_q, num_heads_k, G,
    head_dim: tl.constexpr,
    M_k: tl.constexpr, C_out: tl.constexpr, C_out_padded: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // num_heads_q
    h_q = pid % num_heads_q
    h_k = h_q // G
    
    # Load sequence length for this batch element
    context_len = tl.load(Context_Lens_ptr + b)
    
    # Initialize online softmax accumulators
    max_val = -float('inf')
    sum_val = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    q_cols = tl.arange(0, head_dim)

    # Precompute pointers and values that do not depend on start_n
    lut_head_ptr = LUT_ptr + b * stride_l_b + h_q * stride_l_h
    
    k_idx_head_ptr = K_idx_ptr + h_k * stride_k_h
    k_idx_sub_offsets = tl.arange(0, M_k)[None, :]
    j_offsets = tl.arange(0, M_k)[None, :] * stride_l_m
    
    v_head_ptr = V_ptr + h_k * stride_v_h
    v_col_offsets = q_cols[None, :]

    if C_out > 0:
        qo_offsets = tl.arange(0, C_out_padded)
        qo_mask = tl.arange(0, C_out_padded) < C_out
        qo_vals = tl.load(Q_out_ptr + b * stride_qo_b + h_q * stride_qo_h + qo_offsets, mask=qo_mask, other=0.0)
        
        k_out_head_ptr = K_out_ptr + h_k * stride_ko_h
        ko_col_offsets = tl.arange(0, C_out_padded)[None, :]
        c_out_mask = tl.arange(0, C_out_padded)[None, :] < C_out
    
    # Loop over blocks of BLOCK_N
    for start_n in range(0, context_len, BLOCK_N):
        offsets = start_n + tl.arange(0, BLOCK_N)
        mask_n = offsets < context_len
        
        # Determine physical block index for each token
        virtual_block_idx = offsets // block_size
        block_offset = offsets % block_size
        
        # Load physical block IDs
        physical_block_ids = tl.load(Block_Table_ptr + b * stride_bt_b + virtual_block_idx * stride_bt_s, mask=mask_n, other=0)
        
        # 1. Path A: Dense scoring (LUT gathers)
        # Load all indices for the current block in one 2D load!
        k_idx_offsets = physical_block_ids[:, None] * stride_k_b + block_offset[:, None] * stride_k_s + k_idx_sub_offsets
        idx_block = tl.load(k_idx_head_ptr + k_idx_offsets, mask=mask_n[:, None], other=0) # [BLOCK_N, M_k]

        lut_ptrs = lut_head_ptr + j_offsets + idx_block.to(tl.int32)
        val_block = tl.load(lut_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(val_block, axis=1) # [BLOCK_N]

        # 2. Path B: Outliers
        if C_out > 0:
            # Load all Key outliers for this block in one 2D load!
            ko_offsets = physical_block_ids[:, None] * stride_ko_b + block_offset[:, None] * stride_ko_s + ko_col_offsets
            ko_mask = mask_n[:, None] & c_out_mask
            ko_vals = tl.load(k_out_head_ptr + ko_offsets, mask=ko_mask, other=0.0).to(tl.float32)
            
            outlier_scores = tl.sum(qo_vals[None, :] * ko_vals, axis=1)
            scores += outlier_scores
            
        scores = tl.where(mask_n, scores, -float('inf'))
        
        # Online softmax
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_val, block_max)
        alpha = tl.math.exp(max_val - new_max)
        beta = tl.math.exp(scores - new_max)
        
        # 3. Load V from paged layout: [BLOCK_N, head_dim]
        v_ptr = v_head_ptr + physical_block_ids[:, None] * stride_v_b + block_offset[:, None] * stride_v_s + v_col_offsets
        v = tl.load(v_ptr, mask=mask_n[:, None], other=0.0).to(tl.float32)
        
        # Accumulate
        acc = acc * alpha + tl.sum(beta[:, None] * v, axis=0)
        sum_val = sum_val * alpha + tl.sum(beta, axis=0)
        max_val = new_max
        
    acc = acc / sum_val
    out_ptr = O_ptr + b * stride_o_b + h_q * stride_o_h + q_cols * stride_o_d
    tl.store(out_ptr, acc.to(O_ptr.dtype.element_ty))

def paged_attn_pq_lut_asym(lut, k_idx_paged, k_out_paged, q_out, v_paged, block_table, context_lens, block_size=16, BLOCK_N=256):
    """
    lut: [B, H_q, M_k, K_k]
    k_idx_paged: [num_blocks, H_k, block_size, M_k] (dense key indices)
    k_out_paged: [num_blocks, H_k, block_size, C_out] (outlier key values)
    q_out: [B, H_q, C_out] (outlier query values)
    v_paged: [num_blocks, H_k, block_size, D] (FP8 or BF16 values)
    block_table: [B, max_blocks]
    context_lens: [B]
    """
    B, H_q, M_k, K_k = lut.shape
    _, H_k, _, _ = k_idx_paged.shape
    _, _, _, D = v_paged.shape
    C_out = k_out_paged.shape[-1]
    G = H_q // H_k
    
    out = torch.empty((B, H_q, D), dtype=v_paged.dtype if v_paged.dtype != torch.float8_e4m3fn else torch.bfloat16, device=v_paged.device)
    grid = (B * H_q,)
    
    C_out_padded = 1 << (C_out - 1).bit_length() if C_out > 0 else 0
    if C_out_padded < 1 and C_out > 0:
        C_out_padded = 1
        
    num_warps = 8 if BLOCK_N >= 128 else 4
        
    _paged_attn_pq_lut_asym_kernel[grid](
        lut, k_idx_paged, k_out_paged, q_out, v_paged, block_table, context_lens, out,
        lut.stride(0), lut.stride(1), lut.stride(2), lut.stride(3),
        k_idx_paged.stride(0), k_idx_paged.stride(1), k_idx_paged.stride(2), k_idx_paged.stride(3),
        k_out_paged.stride(0), k_out_paged.stride(1), k_out_paged.stride(2), k_out_paged.stride(3),
        q_out.stride(0), q_out.stride(1), q_out.stride(2),
        v_paged.stride(0), v_paged.stride(1), v_paged.stride(2), v_paged.stride(3),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        H_q, H_k, G, D,
        M_k=M_k, C_out=C_out, C_out_padded=C_out_padded,
        block_size=block_size,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps
    )
    return out.unsqueeze(2)


# =====================================================================
# 9. Sparse decode (Option B): two-pass LUT-score -> top-k -> exact gather.
#    Pass 1 reuses the PQ-LUT-Asym LUT scoring primitive but emits the full
#    per-key score vector (no softmax/value). Host does torch.topk. Pass 2
#    gathers full K/V only for the top-k and runs exact online-softmax
#    attention. This is the kernel realization of the sparse-attention thesis:
#    cheap dequant-free scoring over all keys, exact attention over a few.
# =====================================================================
@triton.jit
def _sparse_score_pq_lut_asym_kernel(
    LUT_ptr, K_idx_ptr, K_out_ptr, Q_out_ptr, Scores_ptr,
    stride_l_b, stride_l_h, stride_l_m, stride_l_k,
    stride_k_b, stride_k_h, stride_k_s, stride_k_m,
    stride_ko_b, stride_ko_h, stride_ko_s, stride_ko_c,
    stride_qo_b, stride_qo_h, stride_qo_c,
    stride_sc_b, stride_sc_h, stride_sc_s,
    num_heads_q, num_heads_k, G,
    context_len,
    M_k: tl.constexpr, C_out: tl.constexpr, C_out_padded: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // num_heads_q
    h_q = pid % num_heads_q
    h_k = h_q // G

    lut_head_ptr = LUT_ptr + b * stride_l_b + h_q * stride_l_h
    k_idx_base_ptr = K_idx_ptr + b * stride_k_b + h_k * stride_k_h
    k_idx_sub_offsets = tl.arange(0, M_k)[None, :] * stride_k_m
    j_offsets = tl.arange(0, M_k)[None, :] * stride_l_m

    if C_out > 0:
        qo_offsets = tl.arange(0, C_out_padded) * stride_qo_c
        qo_mask = tl.arange(0, C_out_padded) < C_out
        qo_vals = tl.load(Q_out_ptr + b * stride_qo_b + h_q * stride_qo_h + qo_offsets,
                          mask=qo_mask, other=0.0).to(tl.float32)
        k_out_base_ptr = K_out_ptr + b * stride_ko_b + h_k * stride_ko_h
        ko_col_offsets = tl.arange(0, C_out_padded)[None, :] * stride_ko_c
        c_out_mask = tl.arange(0, C_out_padded)[None, :] < C_out

    for start_n in range(0, context_len, BLOCK_N):
        offs = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs < context_len
        # Path A: dense LUT gather-sum
        k_idx_offsets = offs[:, None] * stride_k_s + k_idx_sub_offsets
        idx_block = tl.load(k_idx_base_ptr + k_idx_offsets, mask=mask_n[:, None], other=0)
        lut_ptrs = lut_head_ptr + j_offsets + idx_block.to(tl.int32) * stride_l_k
        val_block = tl.load(lut_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(val_block, axis=1)
        # Path B: exact outlier dot product
        if C_out > 0:
            ko_offsets = offs[:, None] * stride_ko_s + ko_col_offsets
            ko_vals = tl.load(k_out_base_ptr + ko_offsets,
                              mask=mask_n[:, None] & c_out_mask, other=0.0).to(tl.float32)
            scores += tl.sum(qo_vals[None, :] * ko_vals, axis=1)
        sc_ptr = Scores_ptr + b * stride_sc_b + h_q * stride_sc_h + offs * stride_sc_s
        tl.store(sc_ptr, scores, mask=mask_n)


def sparse_score_pq_lut_asym(lut, k_idx, k_out, q_out, BLOCK_N=256):
    """Pass 1: per-key approximate scores [B, H_q, S] (fp32) via LUT gather + outlier dot.
    Decode-shaped: all S cached keys are causally valid for the single new query."""
    B, H_q, M_k, K_k = lut.shape
    _, H_k, S, _ = k_idx.shape
    C_out = k_out.shape[-1]
    G = H_q // H_k
    scores = torch.empty((B, H_q, S), dtype=torch.float32, device=lut.device)
    C_out_padded = 1 << (C_out - 1).bit_length() if C_out > 0 else 0
    if C_out_padded < 1 and C_out > 0:
        C_out_padded = 1
    num_warps = 8 if BLOCK_N >= 128 else 4
    grid = (B * H_q,)
    _sparse_score_pq_lut_asym_kernel[grid](
        lut, k_idx, k_out, q_out, scores,
        lut.stride(0), lut.stride(1), lut.stride(2), lut.stride(3),
        k_idx.stride(0), k_idx.stride(1), k_idx.stride(2), k_idx.stride(3),
        k_out.stride(0), k_out.stride(1), k_out.stride(2), k_out.stride(3),
        q_out.stride(0), q_out.stride(1), q_out.stride(2),
        scores.stride(0), scores.stride(1), scores.stride(2),
        H_q, H_k, G, S,
        M_k=M_k, C_out=C_out, C_out_padded=C_out_padded,
        BLOCK_N=BLOCK_N, num_warps=num_warps,
    )
    return scores


@triton.jit
def _sparse_gather_attn_kernel(
    Q_ptr, K_ptr, V_ptr, Idx_ptr, O_ptr,
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_i_b, stride_i_h, stride_i_s,
    stride_o_b, stride_o_h, stride_o_d,
    num_heads_q, num_heads_k, G, scaling,
    topk, head_dim: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // num_heads_q
    h_q = pid % num_heads_q
    h_k = h_q // G

    q_cols = tl.arange(0, head_dim)
    q = tl.load(Q_ptr + b * stride_q_b + h_q * stride_q_h + q_cols * stride_q_d).to(tl.float32)

    idx_base = Idx_ptr + b * stride_i_b + h_q * stride_i_h
    k_base = K_ptr + b * stride_k_b + h_k * stride_k_h
    v_base = V_ptr + b * stride_v_b + h_k * stride_v_h

    max_val = -float('inf')
    sum_val = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    for start in range(0, topk, BLOCK_K):
        ar = start + tl.arange(0, BLOCK_K)
        m = ar < topk
        sel = tl.load(idx_base + ar * stride_i_s, mask=m, other=0)   # [BLOCK_K] key positions
        k_off = sel[:, None] * stride_k_s + q_cols[None, :] * stride_k_d
        k = tl.load(k_base + k_off, mask=m[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1) * scaling           # [BLOCK_K]
        scores = tl.where(m, scores, -float('inf'))
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_val, block_max)
        alpha = tl.math.exp(max_val - new_max)
        beta = tl.math.exp(scores - new_max)
        v_off = sel[:, None] * stride_v_s + q_cols[None, :] * stride_v_d
        v = tl.load(v_base + v_off, mask=m[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(beta[:, None] * v, axis=0)
        sum_val = sum_val * alpha + tl.sum(beta, axis=0)
        max_val = new_max

    acc = acc / sum_val
    tl.store(O_ptr + b * stride_o_b + h_q * stride_o_h + q_cols * stride_o_d,
             acc.to(O_ptr.dtype.element_ty))


def sparse_gather_attn(q, k, v, idx, scaling, BLOCK_K=64):
    """Pass 2: exact online-softmax attention over the gathered top-k keys.
    q:[B,H_q,1,D]; k,v:[B,H_k,S,D] (bf16/fp8); idx:[B,H_q,topk] (key positions)."""
    B, H_q, _, D = q.shape
    _, H_k, S, _ = k.shape
    topk = idx.shape[-1]
    G = H_q // H_k
    q_sq = q.squeeze(2)
    out_dtype = torch.bfloat16 if v.dtype == torch.float8_e4m3fn else v.dtype
    out = torch.empty((B, H_q, D), dtype=out_dtype, device=q.device)
    idx = idx.to(torch.int32).contiguous()
    grid = (B * H_q,)
    _sparse_gather_attn_kernel[grid](
        q_sq, k, v, idx, out,
        q_sq.stride(0), q_sq.stride(1), q_sq.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        idx.stride(0), idx.stride(1), idx.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        H_q, H_k, G, scaling,
        topk, head_dim=D, BLOCK_K=BLOCK_K, num_warps=4,
    )
    return out.unsqueeze(2)


def sparse_decode_pq_lut(lut, k_idx, k_out, q_out, q, k, v, frac, scaling,
                       BLOCK_N=256, BLOCK_K=64):
    """End-to-end sparse decode: LUT-score all keys -> top-k -> exact gather attention.
    Returns [B, H_q, 1, D]."""
    scores = sparse_score_pq_lut_asym(lut, k_idx, k_out, q_out, BLOCK_N=BLOCK_N)  # [B,H_q,S]
    S = scores.shape[-1]
    kb = max(1, int(round(frac * S)))
    idx = scores.topk(kb, dim=-1).indices                                       # [B,H_q,kb]
    return sparse_gather_attn(q, k, v, idx, scaling, BLOCK_K=BLOCK_K)


# =====================================================================
# 10. Cheap-scalar Pass-1 score kernels & fused pipeline
# =====================================================================

@triton.jit
def _sparse_score_scalar_kernel(
    Q_ptr, K_ptr, Scores_ptr,
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_sc_b, stride_sc_h, stride_sc_s,
    num_heads_q, num_heads_k, G, scaling,
    context_len, head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // num_heads_q
    h_q = pid % num_heads_q
    h_k = h_q // G
    
    # Load Q: [head_dim]
    q_cols = tl.arange(0, head_dim)
    q_ptr = Q_ptr + b * stride_q_b + h_q * stride_q_h + q_cols * stride_q_d
    q = tl.load(q_ptr).to(tl.float32)
    
    for start_n in range(0, context_len, BLOCK_N):
        mask_n = (start_n + tl.arange(0, BLOCK_N)) < context_len
        row_offsets = start_n + tl.arange(0, BLOCK_N)
        
        # Load K: [BLOCK_N, head_dim]
        k_offsets = row_offsets[:, None] * stride_k_s + q_cols[None, :] * stride_k_d
        k_ptr_block = K_ptr + b * stride_k_b + h_k * stride_k_h + k_offsets
        k = tl.load(k_ptr_block, mask=mask_n[:, None], other=0.0).to(tl.float32)
        
        # Compute dot product
        scores = tl.sum(q[None, :] * k, axis=1) * scaling
        
        sc_ptr = Scores_ptr + b * stride_sc_b + h_q * stride_sc_h + row_offsets * stride_sc_s
        tl.store(sc_ptr, scores, mask=mask_n)


def sparse_score_scalar(q, k_score, scaling, *, score_bits=16, BLOCK_N=128):
    """Pass 1: per-key exact bf16 or fp8 scores [B, H_q, S] (fp32) via dense matrix multiplication."""
    B, H_q, _, D = q.shape
    _, H_k, S, _ = k_score.shape
    G = H_q // H_k
    
    q_squeezed = q.squeeze(2)
    scores = torch.empty((B, H_q, S), dtype=torch.float32, device=q.device)
    
    grid = (B * H_q,)
    _sparse_score_scalar_kernel[grid](
        q_squeezed, k_score, scores,
        q_squeezed.stride(0), q_squeezed.stride(1), q_squeezed.stride(2),
        k_score.stride(0), k_score.stride(1), k_score.stride(2), k_score.stride(3),
        scores.stride(0), scores.stride(1), scores.stride(2),
        H_q, H_k, G, scaling,
        S, D,
        BLOCK_N=BLOCK_N,
        num_warps=4
    )
    return scores


@triton.jit
def _sparse_score_packed_kernel(
    Q_ptr, K_packed_ptr, Scale_ptr, Zero_ptr, Scores_ptr,
    stride_q_b, stride_q_h, stride_q_d,
    stride_kp_b, stride_kp_h, stride_kp_s, stride_kp_d,
    stride_sc_b, stride_sc_h, stride_sc_s,
    stride_s_b, stride_s_h, stride_s_s, stride_s_g,
    stride_z_b, stride_z_h, stride_z_s, stride_z_g,
    num_heads_q, num_heads_k, G, scaling,
    context_len, head_dim: tl.constexpr,
    score_bits: tl.constexpr, group_size: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // num_heads_q
    h_q = pid % num_heads_q
    h_k = h_q // G
    
    # Load Q: [head_dim]
    q_cols = tl.arange(0, head_dim)
    q_ptr = Q_ptr + b * stride_q_b + h_q * stride_q_h + q_cols * stride_q_d
    q = tl.load(q_ptr).to(tl.float32)
    
    # Unpacking parameters
    elems_per_byte = 8 // score_bits
    mask = (1 << score_bits) - 1
    
    # Precompute columns mapping
    byte_cols = q_cols // elems_per_byte
    shifts = (q_cols % elems_per_byte) * score_bits
    group_indices = q_cols // group_size
    
    for start_n in range(0, context_len, BLOCK_N):
        mask_n = (start_n + tl.arange(0, BLOCK_N)) < context_len
        row_offsets = start_n + tl.arange(0, BLOCK_N)
        
        # Load packed bytes
        kp_offsets = row_offsets[:, None] * stride_kp_s + byte_cols[None, :] * stride_kp_d
        kp_ptr = K_packed_ptr + b * stride_kp_b + h_k * stride_kp_h + kp_offsets
        packed_val = tl.load(kp_ptr, mask=mask_n[:, None], other=0)
        
        # In-register dequant
        q_val = (packed_val >> shifts[None, :]) & mask
        
        # Load scale and zero
        s_offsets = row_offsets[:, None] * stride_s_s + group_indices[None, :] * stride_s_g
        s_ptr = Scale_ptr + b * stride_s_b + h_k * stride_s_h + s_offsets
        scale = tl.load(s_ptr, mask=mask_n[:, None], other=1.0).to(tl.float32)
        
        z_offsets = row_offsets[:, None] * stride_z_s + group_indices[None, :] * stride_z_g
        z_ptr = Zero_ptr + b * stride_z_b + h_k * stride_z_h + z_offsets
        zero = tl.load(z_ptr, mask=mask_n[:, None], other=0.0).to(tl.float32)
        
        # Reconstruct key block
        k = q_val.to(tl.float32) * scale + zero
        
        # Compute dot product
        scores = tl.sum(q[None, :] * k, axis=1) * scaling
        
        sc_ptr = Scores_ptr + b * stride_sc_b + h_q * stride_sc_h + row_offsets * stride_sc_s
        tl.store(sc_ptr, scores, mask=mask_n)


def sparse_score_packed(q, k_packed, scale, zero, scaling, *, score_bits=4, group_size=32, BLOCK_N=128):
    """Pass 1: per-key approximate scores [B, H_q, S] (fp32) via packed INT dequantize-on-the-fly."""
    B, H_q, _, D = q.shape
    _, H_k, S, _ = k_packed.shape
    G = H_q // H_k
    
    q_squeezed = q.squeeze(2)
    scores = torch.empty((B, H_q, S), dtype=torch.float32, device=q.device)
    
    grid = (B * H_q,)
    _sparse_score_packed_kernel[grid](
        q_squeezed, k_packed, scale, zero, scores,
        q_squeezed.stride(0), q_squeezed.stride(1), q_squeezed.stride(2),
        k_packed.stride(0), k_packed.stride(1), k_packed.stride(2), k_packed.stride(3),
        scores.stride(0), scores.stride(1), scores.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2), scale.stride(3),
        zero.stride(0), zero.stride(1), zero.stride(2), zero.stride(3),
        H_q, H_k, G, scaling,
        S, D,
        score_bits=score_bits, group_size=group_size,
        BLOCK_N=BLOCK_N,
        num_warps=4
    )
    return scores


def quantize_and_pack_score_keys(k, score_bits, group_size):
    """Helper to quantize and pack keys along head-dim."""
    B, H_k, S, D = k.shape
    num_groups = D // group_size
    levels = 2 ** score_bits
    
    k_reshaped = k.view(B, H_k, S, num_groups, group_size).float()
    xmin = k_reshaped.amin(dim=-1, keepdim=True)
    xmax = k_reshaped.amax(dim=-1, keepdim=True)
    scale = (xmax - xmin).clamp(min=1e-8) / (levels - 1)
    q = torch.clamp(torch.round((k_reshaped - xmin) / scale), 0, levels - 1).to(torch.uint8)
    
    q_flat = q.view(B, H_k, S, D)
    
    if score_bits == 4:
        packed = (q_flat[..., 0::2] & 0x0F) | ((q_flat[..., 1::2] & 0x0F) << 4)
    elif score_bits == 2:
        packed = (q_flat[..., 0::4] & 0x03) | \
                 ((q_flat[..., 1::4] & 0x03) << 2) | \
                 ((q_flat[..., 2::4] & 0x03) << 4) | \
                 ((q_flat[..., 3::4] & 0x03) << 6)
    else:
        raise ValueError("Unsupported score_bits")
        
    return packed, scale.view(B, H_k, S, num_groups).to(k.dtype), xmin.view(B, H_k, S, num_groups).to(k.dtype)


def sparse_decode_scalar(q, k_score_or_packed, scale, zero, k_val, v_val, frac, scaling, *, score_bits, group_size=32, BLOCK_N=128, BLOCK_K=64):
    """End-to-end two-pass sparse decode: score all keys -> top-k -> exact gather.
    Returns [B, H_q, 1, D]."""
    if score_bits in (16, 8):
        scores = sparse_score_scalar(q, k_score_or_packed, scaling, score_bits=score_bits, BLOCK_N=BLOCK_N)
    else:
        scores = sparse_score_packed(q, k_score_or_packed, scale, zero, scaling, score_bits=score_bits, group_size=group_size, BLOCK_N=BLOCK_N)
        
    S = scores.shape[-1]
    kb = max(1, int(round(frac * S)))
    idx = scores.topk(kb, dim=-1).indices
    return sparse_gather_attn(q, k_val, v_val, idx, scaling, BLOCK_K=BLOCK_K)

