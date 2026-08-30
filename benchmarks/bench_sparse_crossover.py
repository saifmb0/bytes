import sys
import argparse
import os
import gc
import json
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# KIVI is optional and configured explicitly for reproducibility.
KIVI_REPO = os.environ.get("KIVI_REPO")
if not KIVI_REPO:
    raise RuntimeError("set KIVI_REPO to the official KIVI checkout")
sys.path.append(KIVI_REPO)
sys.path.append(os.path.join(KIVI_REPO, "quant"))

from src.attention_kernels import (
    decode_attn_bf16,
    decode_attn_fp8,
    sparse_score_scalar,
    sparse_score_packed,
    quantize_and_pack_score_keys,
    sparse_gather_attn,
    sparse_decode_scalar
)
from benchmarks.kivi_latency import kivi_decode_attn, repeat_kv
from quant.new_pack import triton_quantize_and_pack_along_last_dim

def benchmark_with_cuda_graphs(attn_func, *args, num_warmups=20, num_runs=200):
    """Isolate pure GPU runtime using CUDA Graphs."""
    try:
        # Warm up
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
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Clear graph/cache on OOM
            gc.collect()
            torch.cuda.empty_cache()
            return float('nan')
        else:
            raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="results/paper/sparse_crossover.json")
    args = parser.parse_args()
    # Grid parameters
    S_grid = [4096, 8192, 16384, 32768, 65536]
    frac_grid = [0.01, 0.02, 0.05, 0.10, 0.25]
    
    # Headline shape
    B = 32
    D = 128
    H_q = 64
    H_k = 8
    G = H_q // H_k
    scaling = 1.0 / math.sqrt(D)
    
    results = []
    
    print("=== STARTING DECISIVE CROSSOVER SWEEP ===")
    print(f"Shape: B={B}, D={D}, H_q={H_q}, H_k={H_k}")
    
    for S in S_grid:
        for frac in frac_grid:
            print(f"\nEvaluating cell S={S}, frac={frac}...")
            cell_data = {
                "S": S, "frac": frac,
                "lat_ms": {}, "roofline": {}
            }
            
            # Setup inputs
            kb = max(1, int(round(frac * S)))
            
            # Helper to run benchmarks cleanly and free memory immediately
            # 1. BF16 native timing
            try:
                q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(B, H_k, S, D, device="cuda", dtype=torch.bfloat16)
                v = torch.randn(B, H_k, S, D, device="cuda", dtype=torch.bfloat16)
                
                lat_bf16 = benchmark_with_cuda_graphs(decode_attn_bf16, q, k, v)
                cell_data["lat_ms"]["bf16"] = lat_bf16
            except Exception as e:
                print(f"  bf16 failed: {e}")
                cell_data["lat_ms"]["bf16"] = float('nan')
                
            # 2. FP8 native timing
            try:
                k_fp8 = k.to(torch.float8_e4m3fn)
                v_fp8 = v.to(torch.float8_e4m3fn)
                lat_fp8 = benchmark_with_cuda_graphs(decode_attn_fp8, q, k_fp8, v_fp8)
                cell_data["lat_ms"]["fp8"] = lat_fp8
            except Exception as e:
                print(f"  fp8 failed: {e}")
                cell_data["lat_ms"]["fp8"] = float('nan')
            
            # 3. KIVI-4 timing
            try:
                S_res = 128
                Sq = S - S_res
                K_quant = k[:, :, :Sq, :]
                K_res = k[:, :, Sq:, :]
                V_quant = v[:, :, :Sq, :]
                V_res = v[:, :, Sq:, :]
                
                K_quant_trans = K_quant.transpose(2, 3).contiguous().to(torch.float16)
                V_quant_f16 = V_quant.contiguous().to(torch.float16)
                
                Kq_4, K_scale_4, K_mn_4 = triton_quantize_and_pack_along_last_dim(K_quant_trans, 32, 4)
                Vq_4, V_scale_4, V_mn_4 = triton_quantize_and_pack_along_last_dim(V_quant_f16, 32, 4)
                
                lat_kivi4 = benchmark_with_cuda_graphs(
                    lambda: kivi_decode_attn(q, Kq_4, K_scale_4, K_mn_4, K_res, Vq_4, V_scale_4, V_mn_4, V_res, 4, 32)
                )
                cell_data["lat_ms"]["kivi4"] = lat_kivi4
                
                del Kq_4, K_scale_4, K_mn_4, Vq_4, V_scale_4, V_mn_4
            except Exception as e:
                print(f"  kivi4 failed: {e}")
                cell_data["lat_ms"]["kivi4"] = float('nan')
                
            # 4. KIVI-2 timing
            try:
                K_quant_trans = K_quant.transpose(2, 3).contiguous().to(torch.float16)
                V_quant_f16 = V_quant.contiguous().to(torch.float16)
                
                Kq_2, K_scale_2, K_mn_2 = triton_quantize_and_pack_along_last_dim(K_quant_trans, 32, 2)
                Vq_2, V_scale_2, V_mn_2 = triton_quantize_and_pack_along_last_dim(V_quant_f16, 32, 2)
                
                lat_kivi2 = benchmark_with_cuda_graphs(
                    lambda: kivi_decode_attn(q, Kq_2, K_scale_2, K_mn_2, K_res, Vq_2, V_scale_2, V_mn_2, V_res, 2, 32)
                )
                cell_data["lat_ms"]["kivi2"] = lat_kivi2
                
                del Kq_2, K_scale_2, K_mn_2, Vq_2, V_scale_2, V_mn_2
            except Exception as e:
                print(f"  kivi2 failed: {e}")
                cell_data["lat_ms"]["kivi2"] = float('nan')
                
            # 5. Sparse BF16 timing
            try:
                lat_sparse_bf16 = benchmark_with_cuda_graphs(
                    lambda: sparse_decode_scalar(q, k, None, None, k, v, frac, scaling,
                                                 score_bits=16, group_size=32, BLOCK_N=128, BLOCK_K=64)
                )
                cell_data["lat_ms"]["sparse_bf16"] = lat_sparse_bf16
            except Exception as e:
                print(f"  sparse_bf16 failed: {e}")
                cell_data["lat_ms"]["sparse_bf16"] = float('nan')
                
            # 6. Sparse FP8 timing
            try:
                lat_sparse_fp8 = benchmark_with_cuda_graphs(
                    lambda: sparse_decode_scalar(q, k_fp8, None, None, k, v, frac, scaling,
                                                 score_bits=8, group_size=32, BLOCK_N=128, BLOCK_K=64)
                )
                cell_data["lat_ms"]["sparse_fp8"] = lat_sparse_fp8
            except Exception as e:
                print(f"  sparse_fp8 failed: {e}")
                cell_data["lat_ms"]["sparse_fp8"] = float('nan')
                
            # 7. Sparse INT4 timing
            try:
                k_packed_4, scale_4, zero_4 = quantize_and_pack_score_keys(k, 4, 32)
                lat_sparse_int4 = benchmark_with_cuda_graphs(
                    lambda: sparse_decode_scalar(q, k_packed_4, scale_4, zero_4, k, v, frac, scaling,
                                                 score_bits=4, group_size=32, BLOCK_N=128, BLOCK_K=64)
                )
                cell_data["lat_ms"]["sparse_int4"] = lat_sparse_int4
                del k_packed_4, scale_4, zero_4
            except Exception as e:
                print(f"  sparse_int4 failed: {e}")
                cell_data["lat_ms"]["sparse_int4"] = float('nan')
                
            # 8. Sparse INT2 timing + sub-stage timing
            try:
                k_packed_2, scale_2, zero_2 = quantize_and_pack_score_keys(k, 2, 32)
                lat_sparse_int2 = benchmark_with_cuda_graphs(
                    lambda: sparse_decode_scalar(q, k_packed_2, scale_2, zero_2, k, v, frac, scaling,
                                                 score_bits=2, group_size=32, BLOCK_N=128, BLOCK_K=64)
                )
                cell_data["lat_ms"]["sparse_int2"] = lat_sparse_int2
                
                # Sub-stages timings
                # Score stage
                score_pass_ms = benchmark_with_cuda_graphs(
                    lambda: sparse_score_packed(q, k_packed_2, scale_2, zero_2, scaling, score_bits=2, group_size=32, BLOCK_N=128)
                )
                cell_data["roofline"]["score_pass_ms"] = score_pass_ms
                
                # Top-k stage (on warm precomputed scores tensor)
                scores = sparse_score_packed(q, k_packed_2, scale_2, zero_2, scaling, score_bits=2, group_size=32, BLOCK_N=128)
                topk_only_ms = benchmark_with_cuda_graphs(
                    lambda: scores.topk(kb, dim=-1).indices
                )
                cell_data["roofline"]["topk_only_ms"] = topk_only_ms
                
                # Gather stage (on warm precomputed indices)
                idx = scores.topk(kb, dim=-1).indices
                gather_only_ms = benchmark_with_cuda_graphs(
                    lambda: sparse_gather_attn(q, k, v, idx, scaling, BLOCK_K=64)
                )
                cell_data["roofline"]["gather_only_ms"] = gather_only_ms
                
                del k_packed_2, scale_2, zero_2, scores, idx
            except Exception as e:
                print(f"  sparse_int2/roofline failed: {e}")
                cell_data["lat_ms"]["sparse_int2"] = float('nan')
                cell_data["roofline"] = {
                    "score_pass_ms": float('nan'),
                    "topk_only_ms": float('nan'),
                    "gather_only_ms": float('nan')
                }
                
            # Determine Winner
            lat_dict = cell_data["lat_ms"]
            valid_methods = {k: v for k, v in lat_dict.items() if not math.isnan(v)}
            if valid_methods:
                winner = min(valid_methods, key=valid_methods.get)
                cell_data["winner"] = winner
            else:
                cell_data["winner"] = "None"
                
            results.append(cell_data)
            
            # Print latency breakdown
            print(f"  bf16:        {lat_dict.get('bf16', float('nan')):.3f} ms")
            print(f"  fp8:         {lat_dict.get('fp8', float('nan')):.3f} ms")
            print(f"  kivi4:       {lat_dict.get('kivi4', float('nan')):.3f} ms")
            print(f"  kivi2:       {lat_dict.get('kivi2', float('nan')):.3f} ms")
            print(f"  sparse_int2: {lat_dict.get('sparse_int2', float('nan')):.3f} ms")
            if "score_pass_ms" in cell_data["roofline"]:
                print(f"    Sub-stages: score={cell_data['roofline']['score_pass_ms']:.3f} ms, "
                      f"topk={cell_data['roofline']['topk_only_ms']:.3f} ms, "
                      f"gather={cell_data['roofline']['gather_only_ms']:.3f} ms")
            
            # Memory cleaning between cells
            q = k = v = k_fp8 = v_fp8 = None
            gc.collect()
            torch.cuda.empty_cache()
            
    # Save results to JSON
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved crossover sweep results to {args.output_json}")
    
    # Plot Crossover Heatmap
    # X-axis: Context Length S, Y-axis: Sparsity Fraction frac
    # Values: Speedup or ratio of sparse_int2 / kivi4
    grid_ratio = np.zeros((len(frac_grid), len(S_grid)))
    for cell in results:
        s_idx = S_grid.index(cell["S"])
        f_idx = frac_grid.index(cell["frac"])
        int2_lat = cell["lat_ms"].get("sparse_int2", float('nan'))
        kivi4_lat = cell["lat_ms"].get("kivi4", float('nan'))
        
        if math.isnan(int2_lat) or math.isnan(kivi4_lat):
            grid_ratio[f_idx, s_idx] = np.nan
        else:
            grid_ratio[f_idx, s_idx] = int2_lat / kivi4_lat
            
    plt.figure(figsize=(8, 6))
    im = plt.imshow(grid_ratio, cmap="coolwarm", aspect="auto", origin="lower", vmin=0.5, vmax=1.5)
    plt.colorbar(im, label="Latency Ratio (Sparse INT2 / KIVI-4)\n< 1 means Sparse INT2 is faster")
    
    plt.xticks(range(len(S_grid)), [str(s) for s in S_grid])
    plt.yticks(range(len(frac_grid)), [f"{int(f*100)}%" for f in frac_grid])
    plt.xlabel("Context Length (S)")
    plt.ylabel("Sparsity Fraction (frac)")
    plt.title("Cheap-Scalar Sparse Decode vs KIVI-4 Crossover Heatmap\n(RTX 4000 Ada, B=32, D=128)")
    
    # Annotate ratios in cells
    for i in range(len(frac_grid)):
        for j in range(len(S_grid)):
            val = grid_ratio[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}x", ha="center", va="center", 
                         color="black" if 0.8 < val < 1.2 else "white")
            else:
                plt.text(j, i, "OOM", ha="center", va="center", color="white")
                
    plt.tight_layout()
    plt.savefig("results/plot_sparse_crossover.png", dpi=150)
    print("Generated crossover heatmap: results/plot_sparse_crossover.png")

if __name__ == "__main__":
    main()
