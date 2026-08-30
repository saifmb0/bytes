import sys
import os
import argparse
import math
import json
import gc
import subprocess
import torch
import torch.nn as nn

# Official KIVI checkout is an optional external dependency.
KIVI_REPO = os.environ.get("KIVI_REPO")
if not KIVI_REPO:
    raise RuntimeError("set KIVI_REPO to the official KIVI checkout")
sys.path.append(KIVI_REPO)
sys.path.append(os.path.join(KIVI_REPO, "quant"))
from quant.matmul import cuda_bmm_fA_qB_outer
from quant.new_pack import triton_quantize_and_pack_along_last_dim

def repeat_kv(x, G):
    if G == 1:
        return x
    B, H_k, S, D = x.shape
    return x[:, :, None, :, :].expand(B, H_k, G, S, D).reshape(B, H_k * G, S, D)

def unpack_tensor_last_dim(code, bits):
    shape = code.shape
    feat_per_int = 32 // bits
    S_unpacked = shape[-1] * feat_per_int
    unpacked = torch.zeros(shape[:-1] + (S_unpacked,), dtype=torch.float16, device=code.device)
    num = (1 << bits) - 1
    for i in range(feat_per_int):
        val = (code >> (i * bits)) & num
        unpacked[..., i::feat_per_int] = val
    return unpacked

def dequant_k(Kq, K_scale, K_mn, group_size, bits):
    B, H_k, D, Sq_packed = Kq.shape
    feat_per_int = 32 // bits
    Sq = Sq_packed * feat_per_int
    num_groups = Sq // group_size
    
    K_unpacked = unpack_tensor_last_dim(Kq, bits)
    K_unpacked = K_unpacked.view(B, H_k, D, num_groups, group_size)
    K_dq = K_unpacked * K_scale.unsqueeze(-1) + K_mn.unsqueeze(-1)
    return K_dq.view(B, H_k, D, Sq).transpose(2, 3).contiguous()

def dequant_v(Vq, V_scale, V_mn, group_size, bits):
    B, H_k, Sq, D_packed = Vq.shape
    feat_per_int = 32 // bits
    D = D_packed * feat_per_int
    num_groups = D // group_size
    
    V_unpacked = unpack_tensor_last_dim(Vq, bits)
    V_unpacked = V_unpacked.view(B, H_k, Sq, num_groups, group_size)
    V_dq = V_unpacked * V_scale.unsqueeze(-1) + V_mn.unsqueeze(-1)
    return V_dq.view(B, H_k, Sq, D)

def kivi_decode_attn(Q, Kq, K_scale, K_mn, K_residual, Vq, V_scale, V_mn, V_residual, bits, group_size=32):
    B, H_q, _, D = Q.shape
    H_k = K_residual.shape[1]
    G = H_q // H_k
    S_residual = K_residual.shape[2]
    
    # 1. Compute Q @ K_quant
    Q_f16 = Q.to(torch.float16).contiguous()
    att_qkquant = cuda_bmm_fA_qB_outer(
        group_size, Q_f16, Kq, K_scale, K_mn, bits
    ).to(Q.dtype)
    
    # 2. Compute Q @ K_residual
    K_residual_rep = repeat_kv(K_residual, G)
    att_qkfull = torch.matmul(Q, K_residual_rep.transpose(2, 3))
    
    # 3. Combine and Softmax
    attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(D)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    
    # 4. Compute P @ V_quant and P @ V_residual
    attn_weights_quant = attn_weights[:, :, :, :-S_residual].to(torch.float16).contiguous()
    attn_weights_full = attn_weights[:, :, :, -S_residual:].to(Q.dtype)
    
    attn_output = cuda_bmm_fA_qB_outer(
        group_size, attn_weights_quant, Vq, V_scale, V_mn, bits
    ).to(Q.dtype)
    
    V_residual_rep = repeat_kv(V_residual, G)
    attn_output += torch.matmul(attn_weights_full, V_residual_rep)
    
    return attn_output

def kivi_decode_attn_ref(Q, K_dq, K_residual, V_dq, V_residual, G):
    D = Q.shape[-1]
    K_full = torch.cat([K_dq, K_residual], dim=2)
    V_full = torch.cat([V_dq, V_residual], dim=2)
    K_rep = repeat_kv(K_full, G)
    V_rep = repeat_kv(V_full, G)
    attn_scores = torch.matmul(Q, K_rep.transpose(2, 3)) / math.sqrt(D)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_weights, V_rep)

def run_integrity_gate():
    print("=== RUNNING KIVI INTEGRITY GATE ===")
    B = 2
    H_q = 8
    H_k = 2
    G = H_q // H_k
    D = 128
    Sq = 256
    S_residual = 128
    
    torch.manual_seed(42)
    Q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
    K_quant = torch.randn(B, H_k, Sq, D, device="cuda", dtype=torch.bfloat16)
    K_residual = torch.randn(B, H_k, S_residual, D, device="cuda", dtype=torch.bfloat16)
    V_quant = torch.randn(B, H_k, Sq, D, device="cuda", dtype=torch.bfloat16)
    V_residual = torch.randn(B, H_k, S_residual, D, device="cuda", dtype=torch.bfloat16)
    
    errors = {}
    for bits in [4, 2]:
        # Pack K
        K_quant_trans = K_quant.transpose(2, 3).contiguous().to(torch.float16)
        Kq, K_scale, K_mn = triton_quantize_and_pack_along_last_dim(K_quant_trans, 32, bits)
        
        # Pack V
        V_quant_f16 = V_quant.contiguous().to(torch.float16)
        Vq, V_scale, V_mn = triton_quantize_and_pack_along_last_dim(V_quant_f16, 32, bits)
        
        # Dequantize references
        K_dq = dequant_k(Kq, K_scale, K_mn, 32, bits).to(torch.bfloat16)
        V_dq = dequant_v(Vq, V_scale, V_mn, 32, bits).to(torch.bfloat16)
        
        # Run ref and kernel
        out_ref = kivi_decode_attn_ref(Q, K_dq, K_residual, V_dq, V_residual, G)
        out_kernel = kivi_decode_attn(Q, Kq, K_scale, K_mn, K_residual, Vq, V_scale, V_mn, V_residual, bits, 32)
        
        rel_err = torch.norm(out_kernel - out_ref) / torch.norm(out_ref)
        print(f"{bits}-bit relative error: {rel_err:.2e}")
        errors[bits] = rel_err.item()
        
        if rel_err >= 0.01:
            print(f"ABORT: {bits}-bit relative error {rel_err:.2e} >= 1%!")
            sys.exit(1)
            
    print("KIVI integrity gate: PASS")
    return errors

def benchmark_with_cuda_graphs(attn_func, *args, num_warmups=20, num_runs=200):
    for _ in range(num_warmups):
        _ = attn_func(*args)
    torch.cuda.synchronize()
    
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = attn_func(*args)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        g.replay()
    end_event.record()
    
    torch.cuda.synchronize()
    latency_ms = start_event.elapsed_time(end_event) / num_runs
    return latency_ms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="results/paper/kivi_latency.json")
    args = parser.parse_args()
    errors = run_integrity_gate()
    
    # Benchmarking shape
    B = 32
    H_q = 64
    H_k = 8
    D = 128
    S = 16384
    S_residual = 128
    Sq = S - S_residual
    
    assert (S - 128) % 32 == 0
    
    torch.manual_seed(42)
    Q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
    K_quant = torch.randn(B, H_k, Sq, D, device="cuda", dtype=torch.bfloat16)
    K_residual = torch.randn(B, H_k, S_residual, D, device="cuda", dtype=torch.bfloat16)
    V_quant = torch.randn(B, H_k, Sq, D, device="cuda", dtype=torch.bfloat16)
    V_residual = torch.randn(B, H_k, S_residual, D, device="cuda", dtype=torch.bfloat16)
    
    kivi_revision = subprocess.check_output(["git", "-C", KIVI_REPO, "rev-parse", "HEAD"], text=True).strip()
    results = {
        "_meta": {
            "description": "KIVI decode-attention latency using KIVI's official CUDA kernel (kivi_gemv.gemv_forward_cuda_outer_dim via quant.matmul.cuda_bmm_fA_qB_outer), built for sm89 (RTX 4000 Ada). Measured operating point B=32, S=16384, D=128, H_q=64, H_k=8, fp16, CUDA graphs (20 warmup / 200 runs). Complete measured attention step: Q@Kq (KIVI kernel) + Q@Kfull (fp16 residual window) + softmax + P@Vq (KIVI kernel) + P@Vfull (fp16 residual). KIVI key=per-channel INT (quantized along token axis, group_size groups); value=per-token INT; full-precision residual window of the most recent `residual_length` tokens. All compared methods time attention over an already-populated cache; the O(1) per-step cache append is excluded for every method.",
            "integrity_gate": f"PASS — cuda_bmm_fA_qB_outer validated against a full fp16-dequant matmul reference at a small GQA shape: 4-bit rel-err {errors[4]:.2e}, 2-bit rel-err {errors[2]:.2e} (both < 1%).",
            "harness": "benchmarks/kivi_latency.py",
            "kivi_repo": f"https://github.com/jy-yuan/KIVI.git @ {kivi_revision}",
            "bpw_conventions": "eff_key_bpw_full counts per-group scale+mn (2*16/group_size) + residual FP16 share (16*W/S), including all storage overhead. eff_key_bpw_harness omits scale+mn (bits*(S-W)/S + 16*W/S), matching the convention KIVIQuantizer.bpw() uses on the PPL panel. Both are reported; quality and latency panels must state which convention they use.",
        }
    }
    
    for bits in [4, 2]:
        print(f"Benchmarking KIVI-{bits}...")
        K_quant_trans = K_quant.transpose(2, 3).contiguous().to(torch.float16)
        Kq, K_scale, K_mn = triton_quantize_and_pack_along_last_dim(K_quant_trans, 32, bits)
        V_quant_f16 = V_quant.contiguous().to(torch.float16)
        Vq, V_scale, V_mn = triton_quantize_and_pack_along_last_dim(V_quant_f16, 32, bits)
        
        # Measure peak memory
        torch.cuda.reset_peak_memory_stats()
        lat_ms = benchmark_with_cuda_graphs(
            kivi_decode_attn, Q, Kq, K_scale, K_mn, K_residual, Vq, V_scale, V_mn, V_residual, bits, 32
        )
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        # Calculate bpw conventions
        W = S_residual
        bpw_harness = bits * (S - W) / S + 16 * W / S
        bpw_full = bits * (S - W) / S + (32 / 32) * (S - W) / S + 16 * W / S
        
        results[f"KIVI-{bits}"] = {
            "k_bits": bits, "v_bits": bits, "group_size": 32, "residual_length": S_residual,
            "eff_key_bpw_full": round(bpw_full, 3), "eff_key_bpw_harness": round(bpw_harness, 3),
            "lat_ms": round(lat_ms, 3), "peak_gb": round(peak_gb, 2),
            "B": B, "S": S, "D": D, "H_q": H_q, "H_k": H_k
        }
        print(f"KIVI-{bits} Latency: {lat_ms:.3f} ms, Peak Memory: {peak_gb:.2f} GB")
        
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output_json}")

if __name__ == "__main__":
    main()
