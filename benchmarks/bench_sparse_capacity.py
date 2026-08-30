import sys
import os
import gc
import json
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="results/paper/sparse_capacity.json")
    args = parser.parse_args()
    device = "cuda"
    D = 128
    H_q = 64
    H_k = 8
    S_grid = [16384, 32768]
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512]
    
    results = {}
    
    print("=== STARTING STRESS TEST: MAXIMUM BATCH CAPACITY ===")
    print(f"Shape parameters: D={D}, H_q={H_q}, H_k={H_k}")
    
    for S in S_grid:
        print(f"\nEvaluating Context Length S = {S}...")
        results[str(S)] = {}
        
        for method_name in ["BF16", "FP8", "KIVI-4", "Sparse-INT2"]:
            max_b = 0
            max_mem_gb = 0.0
            
            for B in batch_sizes:
                # Clean baseline
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Allocation pointers
                tensors = []
                
                try:
                    if method_name == "BF16":
                        k = torch.zeros(B, H_k, S, D, device=device, dtype=torch.bfloat16)
                        v = torch.zeros(B, H_k, S, D, device=device, dtype=torch.bfloat16)
                        tensors.extend([k, v])
                    elif method_name == "FP8":
                        k = torch.zeros(B, H_k, S, D, device=device, dtype=torch.float8_e4m3fn)
                        v = torch.zeros(B, H_k, S, D, device=device, dtype=torch.float8_e4m3fn)
                        tensors.extend([k, v])
                    elif method_name == "KIVI-4":
                        S_res = 128
                        Sq = S - S_res
                        
                        # Keys: per-channel 4-bit, grouped along tokens (group_size=32)
                        # Packed shape: [B, H_k, D, Sq // 8] int32
                        Kq = torch.zeros(B, H_k, D, Sq // 8, device=device, dtype=torch.int32)
                        K_scale = torch.zeros(B, H_k, D, Sq // 32, device=device, dtype=torch.float16)
                        K_mn = torch.zeros(B, H_k, D, Sq // 32, device=device, dtype=torch.float16)
                        
                        # Values: per-token 4-bit, grouped along channels (group_size=32)
                        # Packed shape: [B, H_k, Sq, D // 8] int32
                        Vq = torch.zeros(B, H_k, Sq, D // 8, device=device, dtype=torch.int32)
                        V_scale = torch.zeros(B, H_k, Sq, D // 32, device=device, dtype=torch.float16)
                        V_mn = torch.zeros(B, H_k, Sq, D // 32, device=device, dtype=torch.float16)
                        
                        # Residual window: 128 tokens in FP16/BF16
                        K_res = torch.zeros(B, H_k, S_res, D, device=device, dtype=torch.bfloat16)
                        V_res = torch.zeros(B, H_k, S_res, D, device=device, dtype=torch.bfloat16)
                        
                        tensors.extend([Kq, K_scale, K_mn, Vq, V_scale, V_mn, K_res, V_res])
                    elif method_name == "Sparse-INT2":
                        # Cheap INT2 score keys: packed shape [B, H_k, S, D // 4] uint8
                        k_packed = torch.zeros(B, H_k, S, D // 4, device=device, dtype=torch.uint8)
                        scale = torch.zeros(B, H_k, S, D // 32, device=device, dtype=torch.bfloat16)
                        zero = torch.zeros(B, H_k, S, D // 32, device=device, dtype=torch.bfloat16)
                        
                        # Gathered cache: BF16 keys/values
                        k_val = torch.zeros(B, H_k, S, D, device=device, dtype=torch.bfloat16)
                        v_val = torch.zeros(B, H_k, S, D, device=device, dtype=torch.bfloat16)
                        
                        tensors.extend([k_packed, scale, zero, k_val, v_val])
                        
                    torch.cuda.synchronize()
                    max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    max_b = B
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        break
                    else:
                        raise e
                finally:
                    # Explicit cleanup
                    del tensors
                    gc.collect()
                    torch.cuda.empty_cache()
            
            results[str(S)][method_name] = {
                "max_batch": max_b,
                "peak_mem_gb": round(max_mem_gb, 2)
            }
            print(f"  {method_name:12}: max_batch = {max_b:3}, peak_mem = {max_mem_gb:.2f} GB")
            
    # Save results to JSON
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved capacity results to {args.output_json}")

if __name__ == "__main__":
    main()
