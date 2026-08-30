import torch
from src.quantization import get_hadamard_matrix, rotate_vectors, ProductQuantizer, calibrate_codebook
from src.attention_kernels import (
    decode_attn_bf16, decode_attn_fp8, decode_attn_dequant, decode_attn_pq_lut,
    sparse_score_scalar, sparse_score_packed, quantize_and_pack_score_keys,
    sparse_decode_scalar, sparse_gather_attn
)

def test_hadamard_orthogonality():
    print("Running: test_hadamard_orthogonality...")
    d = 128
    device = "cuda"
    H = get_hadamard_matrix(d, device=device)
    
    I = torch.eye(d, device=device)
    prod = torch.matmul(H.t(), H)
    assert torch.allclose(prod, I, atol=1e-5), "Hadamard matrix is not orthogonal"
    
    sign_pattern = torch.randint(0, 2, (d,), device=device).float() * 2.0 - 1.0
    X = torch.randn(10, d, device=device)
    X_rot = rotate_vectors(X, sign_pattern)
    Y = torch.randn(10, d, device=device)
    Y_rot = rotate_vectors(Y, sign_pattern)
    
    ip_orig = torch.matmul(X, Y.t())
    ip_rot = torch.matmul(X_rot, Y_rot.t())
    assert torch.allclose(ip_orig, ip_rot, atol=1e-5), "Orthogonal invariance violated"
    print("Pass: test_hadamard_orthogonality")

def test_product_quantizer():
    print("Running: test_product_quantizer...")
    d = 128
    d_sub = 8
    bits = 3
    device = "cuda"
    
    X = torch.randn(200, d, device=device)
    pq = ProductQuantizer(d, d_sub, bits, device=device)
    pq.fit(X, num_iters=15)
    
    indices = pq.quantize(X)
    X_hat = pq.dequantize(indices)
    
    assert indices.shape == (200, d // d_sub)
    assert X_hat.shape == (200, d)
    mse = torch.mean((X - X_hat) ** 2)
    assert mse < 1.0, f"Reconstruction error too high: {mse.item()}"
    print("Pass: test_product_quantizer")

def test_bf16_attention_correctness():
    print("Running: test_bf16_attention_correctness...")
    B = 2
    H_q = 8
    H_k = 2
    G = H_q // H_k
    S = 128
    D = 64
    device = "cuda"
    
    q = torch.randn(B, H_q, 1, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H_k, S, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H_k, S, D, device=device, dtype=torch.bfloat16)
    
    q_sq = q.squeeze(2)
    ref_out = torch.zeros(B, H_q, D, device=device, dtype=torch.bfloat16)
    for b in range(B):
        for h_q in range(H_q):
            h_k = h_q // G
            q_vec = q_sq[b, h_q]
            k_mat = k[b, h_k]
            v_mat = v[b, h_k]
            
            scores = torch.matmul(k_mat.float(), q_vec.float()) # [S]
            attn = torch.softmax(scores, dim=-1)
            ref_out[b, h_q] = torch.matmul(attn, v_mat.float()).to(torch.bfloat16)
            
    triton_out = decode_attn_bf16(q, k, v).squeeze(2)
    
    assert torch.allclose(ref_out, triton_out, rtol=1e-2, atol=1e-2)
    print("Pass: test_bf16_attention_correctness")

def test_fp8_attention_correctness():
    print("Running: test_fp8_attention_correctness...")
    B = 2
    H_q = 8
    H_k = 2
    G = H_q // H_k
    S = 128
    D = 64
    device = "cuda"
    
    q = torch.randn(B, H_q, 1, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H_k, S, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H_k, S, D, device=device, dtype=torch.bfloat16)
    
    k_fp8 = k.to(torch.float8_e4m3fn)
    v_fp8 = v.to(torch.float8_e4m3fn)
    
    k_dequant = k_fp8.to(torch.bfloat16)
    v_dequant = v_fp8.to(torch.bfloat16)
    
    q_sq = q.squeeze(2)
    ref_out = torch.zeros(B, H_q, D, device=device, dtype=torch.bfloat16)
    for b in range(B):
        for h_q in range(H_q):
            h_k = h_q // G
            q_vec = q_sq[b, h_q]
            scores = torch.matmul(k_dequant[b, h_k].float(), q_vec.float())
            attn = torch.softmax(scores, dim=-1)
            ref_out[b, h_q] = torch.matmul(attn, v_dequant[b, h_k].float()).to(torch.bfloat16)
            
    triton_out = decode_attn_fp8(q, k_fp8, v_fp8).squeeze(2)
    
    assert torch.allclose(ref_out, triton_out, rtol=1e-2, atol=1e-2)
    print("Pass: test_fp8_attention_correctness")

def test_dequant_attention_correctness():
    print("Running: test_dequant_attention_correctness...")
    B = 2
    H_q = 8
    H_k = 2
    G = H_q // H_k
    S = 128
    D = 64
    d_sub = 8
    bits = 3
    device = "cuda"
    
    q = torch.randn(B, H_q, 1, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H_k, S, D, device=device, dtype=torch.bfloat16)
    
    k_raw = torch.randn(B, H_k, S, D, device=device)
    pq = ProductQuantizer(D, d_sub, bits, device=device)
    pq.fit(k_raw.view(-1, D), num_iters=15)
    
    k_idx = pq.quantize(k_raw).to(torch.uint8)
    # Reconstruct keys using centroids for PyTorch reference
    k_hat = pq.dequantize(k_idx) # [B, H_k, S, D]
    
    q_sq = q.squeeze(2)
    ref_out = torch.zeros(B, H_q, D, device=device, dtype=torch.bfloat16)
    for b in range(B):
        for h_q in range(H_q):
            h_k = h_q // G
            q_vec = q_sq[b, h_q]
            scores = torch.matmul(k_hat[b, h_k].float(), q_vec.float())
            attn = torch.softmax(scores, dim=-1)
            ref_out[b, h_q] = torch.matmul(attn, v[b, h_k].float()).to(torch.bfloat16)
            
    triton_out = decode_attn_dequant(q, k_idx, pq.centroids, v, d_sub=d_sub).squeeze(2)
    
    assert torch.allclose(ref_out, triton_out, rtol=1e-2, atol=1e-2)
    print("Pass: test_dequant_attention_correctness")

def test_pq_lut_attention_correctness():
    print("Running: test_pq_lut_attention_correctness...")
    B = 2
    H_q = 8
    H_k = 2
    G = H_q // H_k
    S = 128
    D = 64
    d_sub = 8
    bits = 3
    device = "cuda"
    
    q = torch.randn(B, H_q, 1, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H_k, S, D, device=device, dtype=torch.bfloat16)
    
    sign_pattern = torch.randint(0, 2, (D,), device=device).float() * 2.0 - 1.0
    q_rot = rotate_vectors(q.squeeze(2), sign_pattern)
    
    k_raw = torch.randn(B, H_k, S, D, device=device)
    k_rot = rotate_vectors(k_raw, sign_pattern)
    
    pq = ProductQuantizer(D, d_sub, bits, device=device)
    pq.fit(k_rot.view(-1, D), num_iters=15)
    
    k_idx = pq.quantize(k_rot).to(torch.uint8)
    
    q_rot_split = q_rot.view(B, H_q, D // d_sub, d_sub)
    lut = torch.einsum("bhmd,mkd->bhmk", q_rot_split.float(), pq.centroids.float())
    
    ref_out = torch.zeros(B, H_q, D, device=device, dtype=torch.bfloat16)
    for b in range(B):
        for h_q in range(H_q):
            h_k = h_q // G
            
            scores = torch.zeros(S, device=device)
            for s_idx in range(S):
                score = 0.0
                for j in range(D // d_sub):
                    centroid_idx = k_idx[b, h_k, s_idx, j].item()
                    score += lut[b, h_q, j, centroid_idx].item()
                scores[s_idx] = score
                
            attn = torch.softmax(scores.float(), dim=-1)
            ref_out[b, h_q] = torch.matmul(attn, v[b, h_k].float()).to(torch.bfloat16)
            
    triton_out = decode_attn_pq_lut(lut, k_idx, v).squeeze(2)
    
    assert torch.allclose(ref_out, triton_out, rtol=1e-2, atol=1e-2)
    print("Pass: test_pq_lut_attention_correctness")

def test_codebook_calibration():
    print("Running: test_codebook_calibration...")
    device = "cuda"
    D = 64
    d_sub = 8
    bits = 3
    
    N_cal = 1000
    queries = torch.randn(N_cal, D, device=device)
    keys = queries * 0.5 + torch.randn(N_cal, D, device=device) * 0.1
    
    k_rot = rotate_vectors(keys)
    pq = ProductQuantizer(D, d_sub, bits, device=device)
    pq.fit(k_rot, num_iters=15)
    
    indices = pq.quantize(k_rot)
    q_rot = rotate_vectors(queries)
    
    k_rot_hat_init = pq.dequantize(indices)
    init_ip_error = torch.mean((torch.sum(queries * keys, dim=-1) - torch.sum(q_rot * k_rot_hat_init, dim=-1)) ** 2)
    
    calibrate_codebook(queries, keys, pq, lr=5e-3, steps=100, batch_size=256)
    
    k_rot_hat_cal = pq.dequantize(indices)
    cal_ip_error = torch.mean((torch.sum(queries * keys, dim=-1) - torch.sum(q_rot * k_rot_hat_cal, dim=-1)) ** 2)
    
    print(f"Initial IP Error: {init_ip_error.item():.6f}")
    print(f"Calibrated IP Error: {cal_ip_error.item():.6f}")
    
    assert cal_ip_error < init_ip_error, "Calibration did not reduce inner product error"
    print("Pass: test_codebook_calibration")

def test_pq_lut_kv_attention_correctness():
    print("Running: test_pq_lut_kv_attention_correctness...")
    from src.attention_kernels import decode_attn_pq_lut_kv
    B = 2
    H_q = 8
    H_k = 2
    G = H_q // H_k
    S = 128
    D = 64
    d_sub = 8
    bits = 3
    device = "cuda"
    
    q = torch.randn(B, H_q, 1, D, device=device, dtype=torch.bfloat16)
    v_raw = torch.randn(B, H_k, S, D, device=device)
    
    # Orthogonal rotation setup
    sign_pattern = torch.randint(0, 2, (D,), device=device).float() * 2.0 - 1.0
    q_rot = rotate_vectors(q.squeeze(2), sign_pattern)
    
    k_raw = torch.randn(B, H_k, S, D, device=device)
    k_rot = rotate_vectors(k_raw, sign_pattern)
    
    # Fit keys and values
    pq_k = ProductQuantizer(D, d_sub, bits, device=device)
    pq_k.fit(k_rot.view(-1, D), num_iters=15)
    k_idx = pq_k.quantize(k_rot).to(torch.uint8)
    
    pq_v = ProductQuantizer(D, d_sub, bits, device=device)
    pq_v.fit(v_raw.view(-1, D), num_iters=15)
    v_idx = pq_v.quantize(v_raw).to(torch.uint8)
    
    v_hat = pq_v.dequantize(v_idx).to(torch.bfloat16)
    
    # LUT precompute
    q_rot_split = q_rot.view(B, H_q, D // d_sub, d_sub)
    lut = torch.einsum("bhmd,mkd->bhmk", q_rot_split.float(), pq_k.centroids.float())
    
    # Reference in PyTorch
    q_sq = q.squeeze(2)
    ref_out = torch.zeros(B, H_q, D, device=device, dtype=torch.bfloat16)
    for b in range(B):
        for h_q in range(H_q):
            h_k = h_q // G
            
            scores = torch.zeros(S, device=device)
            for s_idx in range(S):
                score = 0.0
                for j in range(D // d_sub):
                    centroid_idx = k_idx[b, h_k, s_idx, j].item()
                    score += lut[b, h_q, j, centroid_idx].item()
                scores[s_idx] = score
                
            attn = torch.softmax(scores.float(), dim=-1)
            ref_out[b, h_q] = torch.matmul(attn, v_hat[b, h_k].float()).to(torch.bfloat16)
            
    # Triton PQ-LUT KV Attention
    triton_out = decode_attn_pq_lut_kv(lut, k_idx, v_idx, pq_v.centroids, d_sub_v=d_sub).squeeze(2)
    
    assert torch.allclose(ref_out.float(), triton_out.float(), rtol=1e-2, atol=1e-2)
    print("Pass: test_pq_lut_kv_attention_correctness")

def test_paged_pq_lut_kv_attention_correctness():
    print("Running: test_paged_pq_lut_kv_attention_correctness...")
    from src.attention_kernels import paged_attn_pq_lut_kv
    B = 2
    H_q = 8
    H_k = 2
    G = H_q // H_k
    D = 64
    d_sub = 8
    bits = 3
    block_size = 16
    device = "cuda"
    
    # Define sequence lengths
    context_lens = torch.tensor([45, 78], dtype=torch.int32, device=device)
    max_len = context_lens.max().item()
    
    q = torch.randn(B, H_q, 1, D, device=device, dtype=torch.bfloat16)
    
    # We will generate contiguous keys/values up to max_len
    k_raw = torch.randn(B, H_k, max_len, D, device=device)
    v_raw = torch.randn(B, H_k, max_len, D, device=device)
    
    sign_pattern = torch.randint(0, 2, (D,), device=device).float() * 2.0 - 1.0
    q_rot = rotate_vectors(q.squeeze(2), sign_pattern)
    k_rot = rotate_vectors(k_raw, sign_pattern)
    
    # Fit
    pq_k = ProductQuantizer(D, d_sub, bits, device=device)
    pq_k.fit(k_rot.view(-1, D), num_iters=15)
    k_idx = pq_k.quantize(k_rot).to(torch.uint8) # [B, H_k, max_len, M_k]
    
    pq_v = ProductQuantizer(D, d_sub, bits, device=device)
    pq_v.fit(v_raw.view(-1, D), num_iters=15)
    v_idx = pq_v.quantize(v_raw).to(torch.uint8) # [B, H_k, max_len, M_v]
    
    v_hat = pq_v.dequantize(v_idx).to(torch.bfloat16)
    
    # LUT
    q_rot_split = q_rot.view(B, H_q, D // d_sub, d_sub)
    lut = torch.einsum("bhmd,mkd->bhmk", q_rot_split.float(), pq_k.centroids.float())
    
    # Reference (unpaged, but masked to sequence length)
    q_sq = q.squeeze(2)
    ref_out = torch.zeros(B, H_q, D, device=device, dtype=torch.bfloat16)
    for b in range(B):
        cur_len = context_lens[b].item()
        for h_q in range(H_q):
            h_k = h_q // G
            
            scores = torch.zeros(cur_len, device=device)
            for s_idx in range(cur_len):
                score = 0.0
                for j in range(D // d_sub):
                    centroid_idx = k_idx[b, h_k, s_idx, j].item()
                    score += lut[b, h_q, j, centroid_idx].item()
                scores[s_idx] = score
                
            attn = torch.softmax(scores.float(), dim=-1)
            ref_out[b, h_q] = torch.matmul(attn, v_hat[b, h_k, :cur_len].float()).to(torch.bfloat16)
            
    # Setup paging structure
    # Seq 0: 45 tokens -> 3 blocks of size 16 (ceil(45/16) = 3)
    # Seq 1: 78 tokens -> 5 blocks of size 16 (ceil(78/16) = 5)
    # Total blocks = 8
    num_blocks = 8
    k_idx_paged = torch.zeros(num_blocks, H_k, block_size, D // d_sub, dtype=torch.uint8, device=device)
    v_idx_paged = torch.zeros(num_blocks, H_k, block_size, D // d_sub, dtype=torch.uint8, device=device)
    
    # block table
    # Seq 0 blocks: 0, 1, 2. Seq 1 blocks: 3, 4, 5, 6, 7.
    block_table = torch.tensor([
        [0, 1, 2, -1, -1],
        [3, 4, 5, 6, 7]
    ], dtype=torch.int32, device=device)
    
    # Map flat sequence to paged sequence
    for b in range(B):
        cur_len = context_lens[b].item()
        for t in range(cur_len):
            block_idx = t // block_size
            offset = t % block_size
            physical_block = block_table[b, block_idx].item()
            k_idx_paged[physical_block, :, offset] = k_idx[b, :, t]
            v_idx_paged[physical_block, :, offset] = v_idx[b, :, t]
            
    # Run Paged PQ-LUT Triton kernel
    triton_out = paged_attn_pq_lut_kv(
        lut, k_idx_paged, v_idx_paged, pq_v.centroids, block_table, context_lens, block_size=block_size
    ).squeeze(2)
    
    assert torch.allclose(ref_out.float(), triton_out.float(), rtol=1e-2, atol=1e-2)
    print("Pass: test_paged_pq_lut_kv_attention_correctness")

def test_pq_lut_asym_attention_correctness():
    print("Running: test_pq_lut_asym_attention_correctness...")
    from src.attention_kernels import decode_attn_pq_lut_asym
    import math
    
    # Try different parameter sets to stress test correctness
    test_cases = [
        # B, H_q, H_k, S, D, C_out, v_dtype
        (2, 8, 2, 128, 64, 2, torch.bfloat16),
        (2, 8, 2, 128, 64, 2, torch.float8_e4m3fn),
        (1, 4, 1, 64, 128, 1, torch.float8_e4m3fn),
        (2, 16, 1, 256, 128, 2, torch.bfloat16),
    ]
    
    device = "cuda"
    bits = 3
    d_sub = 8
    
    for B, H_q, H_k, S, D, C_out, v_dtype in test_cases:
        G = H_q // H_k
        M_k = D // d_sub
        
        q = torch.randn(B, H_q, 1, D, device=device, dtype=torch.bfloat16)
        k_raw = torch.randn(B, H_k, S, D, device=device)
        outlier_channels = [12, 45][:C_out]
        for ch in outlier_channels:
            k_raw[:, :, :, ch] = (torch.randn(B, H_k, S, device=device) + 3.0) * 50.0
            
        v_raw = torch.randn(B, H_k, S, D, device=device)
        
        variances = torch.var(k_raw.view(-1, D), dim=0)
        outlier_indices = torch.topk(variances, k=C_out).indices
        
        k_out = k_raw[:, :, :, outlier_indices].to(torch.bfloat16)
        q_out = q.squeeze(2)[:, :, outlier_indices].to(torch.bfloat16)
        
        k_dense = k_raw.clone()
        k_dense[:, :, :, outlier_indices] = 0.0
        q_dense = q.squeeze(2).clone()
        q_dense[:, :, outlier_indices] = 0.0
        
        sign_pattern = torch.randint(0, 2, (D,), device=device).float() * 2.0 - 1.0
        q_rot = rotate_vectors(q_dense, sign_pattern)
        k_rot = rotate_vectors(k_dense, sign_pattern)
        
        pq_k = ProductQuantizer(D, d_sub, bits, device=device)
        pq_k.fit(k_rot.view(-1, D), num_iters=10)
        k_idx = pq_k.quantize(k_rot).to(torch.uint8)
        
        q_rot_split = q_rot.view(B, H_q, M_k, d_sub)
        lut = torch.einsum("bhmd,mkd->bhmk", q_rot_split.float(), pq_k.centroids.float())
        
        if v_dtype == torch.float8_e4m3fn:
            v_cache = v_raw.to(torch.float8_e4m3fn)
            v_ref = v_cache.to(torch.bfloat16)
        else:
            v_cache = v_raw.to(torch.bfloat16)
            v_ref = v_cache
            
        ref_out = torch.zeros(B, H_q, D, device=device, dtype=torch.bfloat16)
        for b in range(B):
            for h_q in range(H_q):
                h_k = h_q // G
                
                scores = torch.zeros(S, device=device)
                for s_idx in range(S):
                    res_score = 0.0
                    for j in range(M_k):
                        centroid_idx = k_idx[b, h_k, s_idx, j].item()
                        res_score += lut[b, h_q, j, centroid_idx].item()
                    
                    out_score = 0.0
                    for c in range(C_out):
                        out_score += q_out[b, h_q, c].item() * k_out[b, h_k, s_idx, c].item()
                        
                    scores[s_idx] = res_score + out_score
                    
                attn = torch.softmax(scores.float(), dim=-1)
                ref_out[b, h_q] = torch.matmul(attn, v_ref[b, h_k].float()).to(torch.bfloat16)
                
        triton_out = decode_attn_pq_lut_asym(lut, k_idx, k_out, q_out, v_cache).squeeze(2)
        
        assert torch.allclose(ref_out.float(), triton_out.float(), rtol=1e-2, atol=1e-2), f"Failed test case: {B, H_q, H_k, S, D, C_out}"
        
    print("Pass: test_pq_lut_asym_attention_correctness")

def test_paged_pq_lut_asym_attention_correctness():
    print("Running: test_paged_pq_lut_asym_attention_correctness...")
    from src.attention_kernels import paged_attn_pq_lut_asym
    import math
    
    test_cases = [
        # B, H_q, H_k, D, C_out, block_size, v_dtype
        (2, 8, 2, 64, 2, 16, torch.bfloat16),
        (2, 8, 2, 64, 2, 16, torch.float8_e4m3fn),
        (1, 4, 1, 128, 1, 32, torch.float8_e4m3fn),
    ]
    
    device = "cuda"
    bits = 3
    d_sub = 8
    
    for B, H_q, H_k, D, C_out, block_size, v_dtype in test_cases:
        G = H_q // H_k
        M_k = D // d_sub
        
        context_lens = torch.tensor([45, 78] if B == 2 else [55], dtype=torch.int32, device=device)
        max_len = context_lens.max().item()
        
        q = torch.randn(B, H_q, 1, D, device=device, dtype=torch.bfloat16)
        k_raw = torch.randn(B, H_k, max_len, D, device=device)
        outlier_channels = [12, 45][:C_out]
        for ch in outlier_channels:
            k_raw[:, :, :, ch] = (torch.randn(B, H_k, max_len, device=device) + 3.0) * 50.0
            
        v_raw = torch.randn(B, H_k, max_len, D, device=device)
        
        variances = torch.var(k_raw.view(-1, D), dim=0)
        outlier_indices = torch.topk(variances, k=C_out).indices
        
        k_out = k_raw[:, :, :, outlier_indices].to(torch.bfloat16)
        q_out = q.squeeze(2)[:, :, outlier_indices].to(torch.bfloat16)
        
        k_dense = k_raw.clone()
        k_dense[:, :, :, outlier_indices] = 0.0
        q_dense = q.squeeze(2).clone()
        q_dense[:, :, outlier_indices] = 0.0
        
        sign_pattern = torch.randint(0, 2, (D,), device=device).float() * 2.0 - 1.0
        q_rot = rotate_vectors(q_dense, sign_pattern)
        k_rot = rotate_vectors(k_dense, sign_pattern)
        
        pq_k = ProductQuantizer(D, d_sub, bits, device=device)
        pq_k.fit(k_rot.view(-1, D), num_iters=10)
        k_idx = pq_k.quantize(k_rot).to(torch.uint8)
        
        q_rot_split = q_rot.view(B, H_q, M_k, d_sub)
        lut = torch.einsum("bhmd,mkd->bhmk", q_rot_split.float(), pq_k.centroids.float())
        
        num_blocks_per_seq = [math.ceil(l.item() / block_size) for l in context_lens]
        total_blocks = sum(num_blocks_per_seq)
        
        k_idx_paged = torch.zeros(total_blocks, H_k, block_size, M_k, dtype=torch.uint8, device=device)
        k_out_paged = torch.zeros(total_blocks, H_k, block_size, C_out, dtype=torch.bfloat16, device=device)
        
        if v_dtype == torch.float8_e4m3fn:
            v_paged = torch.zeros(total_blocks, H_k, block_size, D, dtype=torch.float8_e4m3fn, device=device)
            v_ref = v_raw.to(torch.float8_e4m3fn).to(torch.bfloat16)
        else:
            v_paged = torch.zeros(total_blocks, H_k, block_size, D, dtype=torch.bfloat16, device=device)
            v_ref = v_raw.to(torch.bfloat16)
            
        block_table = torch.full((B, max(num_blocks_per_seq)), -1, dtype=torch.int32, device=device)
        
        block_idx_counter = 0
        for b in range(B):
            for block_seq_idx in range(num_blocks_per_seq[b]):
                block_table[b, block_seq_idx] = block_idx_counter
                block_idx_counter += 1
                
        for b in range(B):
            cur_len = context_lens[b].item()
            for t in range(cur_len):
                block_idx = t // block_size
                offset = t % block_size
                physical_block = block_table[b, block_idx].item()
                k_idx_paged[physical_block, :, offset] = k_idx[b, :, t]
                k_out_paged[physical_block, :, offset] = k_out[b, :, t]
                if v_dtype == torch.float8_e4m3fn:
                    v_paged[physical_block, :, offset] = v_raw.to(torch.float8_e4m3fn)[b, :, t]
                else:
                    v_paged[physical_block, :, offset] = v_raw.to(torch.bfloat16)[b, :, t]
                    
        ref_out = torch.zeros(B, H_q, D, device=device, dtype=torch.bfloat16)
        for b in range(B):
            cur_len = context_lens[b].item()
            for h_q in range(H_q):
                h_k = h_q // G
                
                scores = torch.zeros(cur_len, device=device)
                for s_idx in range(cur_len):
                    res_score = 0.0
                    for j in range(M_k):
                        centroid_idx = k_idx[b, h_k, s_idx, j].item()
                        res_score += lut[b, h_q, j, centroid_idx].item()
                    
                    out_score = 0.0
                    for c in range(C_out):
                        out_score += q_out[b, h_q, c].item() * k_out[b, h_k, s_idx, c].item()
                        
                    scores[s_idx] = res_score + out_score
                    
                attn = torch.softmax(scores.float(), dim=-1)
                ref_out[b, h_q] = torch.matmul(attn, v_ref[b, h_k, :cur_len].float()).to(torch.bfloat16)
                
        triton_out = paged_attn_pq_lut_asym(
            lut, k_idx_paged, k_out_paged, q_out, v_paged, block_table, context_lens, block_size=block_size
        ).squeeze(2)
        
        assert torch.allclose(ref_out.float(), triton_out.float(), rtol=1e-2, atol=1e-2), f"Failed test case: {B, H_q, H_k, D, C_out}"
        
    print("Pass: test_paged_pq_lut_asym_attention_correctness")

def test_turbo_quant_qjl_correctness():
    print("Running: test_turbo_quant_qjl_correctness...")
    import math
    from src.quantization import generate_random_orthogonal, PerChannelScalarQuantizer, generate_jl_matrix
    
    B = 2
    H_q = 8
    H_k = 8
    G = H_q // H_k
    S = 64
    D = 64
    m_jl = 64
    bits_k = 4
    device = "cuda"
    
    # Inputs
    q = torch.randn(B, H_q, 1, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H_k, S, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H_k, S, D, device=device, dtype=torch.bfloat16)
    
    R = generate_random_orthogonal(D, device=device)
    P_jl = generate_jl_matrix(m_jl, D, device=device)
    jl_scale = math.sqrt(math.pi / (2 * m_jl))
    scale_attn = 1.0 / math.sqrt(D)
    
    # Fit scalar quantizer on rotated keys
    k_rot = torch.matmul(k.float(), R) # [B, H_k, S, D]
    scalar_q = PerChannelScalarQuantizer(D, bits_k, device=device)
    scalar_q.fit(k_rot.view(-1, D))
    
    # 1. KEY RECONSTRUCTION PATH (eval_ppl.py logic)
    k_idx = scalar_q.quantize(k_rot)
    k_hat_rot = scalar_q.dequantize(k_idx)
    
    r = k_rot - k_hat_rot
    r_norm = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    r_hat = r / r_norm
    
    # 1-bit JL sketch
    b = torch.sign(torch.einsum('md,bhtd->bhtm', P_jl, r_hat)) # [B, H_k, S, m]
    
    # Reconstructed residual and keys
    r_approx = torch.einsum('md,bhtm->bhtd', P_jl, b) * r_norm * jl_scale
    k_quant = torch.matmul(k_hat_rot + r_approx, R.T)
    
    # Compute attention using reconstructed keys
    k_quant_rep = k_quant[:, :, None, :, :].expand(-1, -1, G, -1, -1).reshape(B, H_q, S, D)
    v_rep = v[:, :, None, :, :].expand(-1, -1, G, -1, -1).reshape(B, H_q, S, D)
    
    scores_ref = torch.matmul(q.float(), k_quant_rep.float().transpose(-1, -2)) * scale_attn # [B, H_q, 1, S]
    attn_ref = torch.softmax(scores_ref.squeeze(2), dim=-1) # [B, H_q, S]
    out_ref = torch.matmul(attn_ref.unsqueeze(2), v_rep.float()).squeeze(2)
    
    # 2. QUERY-SIDE CORRECTION PATH (optimized decode step logic)
    q_rot = torch.matmul(q.float(), R) # [B, H_q, 1, D]
    q_grouped = q_rot.squeeze(2).view(B, H_k, G, D)[:, :, 0, :] # [B, H_k, D]
    
    # Base logits
    base = torch.einsum('bhd,bhtd->bht', q_grouped, k_hat_rot) # [B, H_k, S]
    
    # JL Correction
    q_jl = torch.einsum('md,bhd->bhm', P_jl, q_grouped) # [B, H_k, m]
    correction = torch.einsum('bhm,bhtm->bht', q_jl, b) * (r_norm.squeeze(-1) * jl_scale)
    
    logits_corr = (base + correction) * scale_attn # [B, H_k, S]
    
    # Softmax and reduction per KV head
    attn_corr = torch.softmax(logits_corr, dim=-1)
    out_corr = torch.einsum('bht,bhtd->bhd', attn_corr, v.float()) # [B, H_k, D]
    
    # Repeat for GQA heads to match reference
    out_corr_rep = out_corr.unsqueeze(2).expand(-1, -1, G, -1).reshape(B, H_q, D)
    
    assert torch.allclose(out_ref, out_corr_rep, rtol=1e-5, atol=1e-5), "Numerical equivalence check failed!"
    print("Pass: test_turbo_quant_qjl_correctness")


def test_sparse_scalar_score_correctness():
    print("Running: test_sparse_scalar_score_correctness...")
    device = "cuda"
    B = 2
    H_q = 8
    H_k = 2
    G = H_q // H_k
    S = 128
    scaling = 0.125
    
    def repeat_kv_local(x, G):
        if G == 1:
            return x
        B, H_k, S, D = x.shape
        return x[:, :, None, :, :].expand(B, H_k, G, S, D).reshape(B, H_k * G, S, D)
        
    from src.quantization import _asym_uniform_fakequant
    
    for D in [64, 128]:
        for bits in [16, 8, 4, 2]:
            q = torch.randn(B, H_q, 1, D, device=device, dtype=torch.bfloat16)
            k = torch.randn(B, H_k, S, D, device=device, dtype=torch.bfloat16)
            q_squeezed = q.squeeze(2)
            
            if bits == 16:
                ref = torch.einsum('bhd,bhsd->bhs', q_squeezed.float(), repeat_kv_local(k, G).float()) * scaling
                out = sparse_score_scalar(q, k, scaling, score_bits=16)
            elif bits == 8:
                k_fp8 = k.to(torch.float8_e4m3fn)
                ref = torch.einsum('bhd,bhsd->bhs', q_squeezed.float(), repeat_kv_local(k_fp8.to(torch.float32), G)) * scaling
                out = sparse_score_scalar(q, k_fp8, scaling, score_bits=8)
            elif bits in (4, 2):
                levels = 2 ** bits
                group_size = 32
                k_packed, scale, zero = quantize_and_pack_score_keys(k, bits, group_size)
                feat_per_int = 8 // bits
                unpacked_q = torch.zeros(B, H_k, S, D, dtype=torch.float32, device=k.device)
                mask = (1 << bits) - 1
                for c in range(D):
                    byte_col = c // feat_per_int
                    shift = (c % feat_per_int) * bits
                    val = (k_packed[..., byte_col] >> shift) & mask
                    unpacked_q[..., c] = val
                scale_expanded = scale.unsqueeze(-1).expand(-1, -1, -1, -1, group_size).reshape(B, H_k, S, D)
                zero_expanded = zero.unsqueeze(-1).expand(-1, -1, -1, -1, group_size).reshape(B, H_k, S, D)
                k_dq = (unpacked_q * scale_expanded + zero_expanded).to(k.dtype)
                ref = torch.einsum('bhd,bhsd->bhs', q_squeezed.float(), repeat_kv_local(k_dq, G).float()) * scaling
                out = sparse_score_packed(q, k_packed, scale, zero, scaling, score_bits=bits, group_size=group_size)
                
            assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2), f"Failed score check for D={D}, bits={bits}"
    print("Pass: test_sparse_scalar_score_correctness")


def test_sparse_pipeline_correctness():
    print("Running: test_sparse_pipeline_correctness...")
    device = "cuda"
    B = 2
    H_q = 8
    H_k = 2
    G = H_q // H_k
    S = 256
    scaling = 0.125
    frac = 0.05
    
    def repeat_kv_local(x, G):
        if G == 1:
            return x
        B, H_k, S, D = x.shape
        return x[:, :, None, :, :].expand(B, H_k, G, S, D).reshape(B, H_k * G, S, D)
        
    from src.quantization import _asym_uniform_fakequant
    
    for D in [64, 128]:
        for bits in [16, 8, 4, 2]:
            q = torch.randn(B, H_q, 1, D, device=device, dtype=torch.bfloat16)
            k = torch.randn(B, H_k, S, D, device=device, dtype=torch.bfloat16)
            v = torch.randn(B, H_k, S, D, device=device, dtype=torch.bfloat16)
            
            # Formulate inputs & reference selection
            if bits == 16:
                k_score_input = k
                scale_input, zero_input = None, None
                k_dq = k
            elif bits == 8:
                k_fp8 = k.to(torch.float8_e4m3fn)
                k_score_input = k_fp8
                scale_input, zero_input = None, None
                k_dq = k_fp8.to(torch.float32).to(torch.bfloat16)
            elif bits in (4, 2):
                group_size = 32
                k_packed, scale, zero = quantize_and_pack_score_keys(k, bits, group_size)
                k_score_input = k_packed
                scale_input, zero_input = scale, zero
                levels = 2 ** bits
                feat_per_int = 8 // bits
                unpacked_q = torch.zeros(B, H_k, S, D, dtype=torch.float32, device=k.device)
                mask = (1 << bits) - 1
                for c in range(D):
                    byte_col = c // feat_per_int
                    shift = (c % feat_per_int) * bits
                    val = (k_packed[..., byte_col] >> shift) & mask
                    unpacked_q[..., c] = val
                scale_expanded = scale.unsqueeze(-1).expand(-1, -1, -1, -1, group_size).reshape(B, H_k, S, D)
                zero_expanded = zero.unsqueeze(-1).expand(-1, -1, -1, -1, group_size).reshape(B, H_k, S, D)
                k_dq = (unpacked_q * scale_expanded + zero_expanded).to(k.dtype)
            
            # Reference topk and attention
            q_squeezed = q.squeeze(2)
            if bits == 16:
                scores = sparse_score_scalar(q, k, scaling, score_bits=16)
            elif bits == 8:
                scores = sparse_score_scalar(q, k_fp8, scaling, score_bits=8)
            elif bits in (4, 2):
                scores = sparse_score_packed(q, k_packed, scale, zero, scaling, score_bits=bits, group_size=32)
                
            kb = max(1, int(round(frac * S)))
            kernel_idx = scores.topk(kb, dim=-1).indices
            
            # Reference exact gather attention using the kernel's indices
            k_rep = repeat_kv_local(k, G)
            v_rep = repeat_kv_local(v, G)
            
            gidx = kernel_idx.unsqueeze(-1).expand(B, H_q, kb, D)
            k_sel = torch.gather(k_rep, 2, gidx)
            v_sel = torch.gather(v_rep, 2, gidx)
            
            sc = (q_squeezed.unsqueeze(2) * k_sel).sum(-1) * scaling
            w = torch.softmax(sc, dim=-1)
            ref_out = (w.unsqueeze(-1) * v_sel).sum(2).unsqueeze(2)
            
            # Kernel run
            out = sparse_decode_scalar(
                q, k_score_input, scale_input, zero_input, k, v, frac, scaling,
                score_bits=bits, group_size=32, BLOCK_N=128, BLOCK_K=64
            )
            
            assert torch.allclose(out.float(), ref_out.float(), rtol=2e-2, atol=2e-2), f"Failed pipeline check for D={D}, bits={bits}"
    print("Pass: test_sparse_pipeline_correctness")


if __name__ == "__main__":
    print("=== STARTING CORRECTNESS TESTS ===")
    test_hadamard_orthogonality()
    test_product_quantizer()
    test_bf16_attention_correctness()
    test_fp8_attention_correctness()
    test_dequant_attention_correctness()
    test_pq_lut_attention_correctness()
    test_codebook_calibration()
    test_pq_lut_kv_attention_correctness()
    test_paged_pq_lut_kv_attention_correctness()
    test_pq_lut_asym_attention_correctness()
    test_paged_pq_lut_asym_attention_correctness()
    test_turbo_quant_qjl_correctness()
    test_sparse_scalar_score_correctness()
    test_sparse_pipeline_correctness()
    print("=== ALL CORRECTNESS TESTS PASSED ===")

