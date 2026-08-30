import argparse
import hashlib
import sys
import torch
import torch.nn as nn
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from src.quantization import get_hadamard_matrix, rotate_vectors, ProductQuantizer, calibrate_codebook, detect_outlier_channels, generate_random_orthogonal, PerChannelScalarQuantizer, generate_jl_matrix, KIVIQuantizer, KVQuantNUQ
from src.stability import compute_sign_error, search_best_sign_pattern
from benchmarks.sparse_attention import sparse_attention_chunked, reconstruct_approx_keys, make_selection_keys, SOFT_SELECTORS, dense_attention

# Global state for hooking and calibration
CALIBRATION_DATA = {}
QUANT_CONFIGS = {}
ATTN_KL_DIVS = {}
MODE = "baseline" # "baseline", "collect", "quantize"

# Pilot B1: selection-fidelity (top-k recall of approx LUT scores vs exact scores).
# Computed inside the quantize hook from attn_weights (approx, post-RoPE, from
# quantized keys) vs attn_weights_ref (exact). top-k by softmax weight == top-k by
# score (monotonic), so this is exactly the LUT-oracle ranking fidelity.
COMPUTE_RECALL = False
RECALL_FRACS = [0.02, 0.05, 0.1, 0.2]
RECALL_CTXS = [128, 256, 384, 510]   # query positions (context lengths) to probe
RECALL_STATS = {}                    # (ctx, frac) -> list of mean recalls

# Pilot B2: fixed-budget top-k sparse attention. MODE="sparse" replaces dense
# attention with attend-to-top-k, where the top-k is chosen by SPARSE_SELECTOR:
#   "exact"  -> select by exact post-RoPE scores            (oracle upper bound)
#   "approx" -> select by LUT-approx scores (quantized keys) (OURS)
#   "quest"  -> select top pages by Quest channel-bound      (baseline)
# In all cases the attention OUTPUT uses exact scores+values over the selected set,
# so PPL isolates selection quality. Budget k = round(SPARSE_FRAC * context).
SPARSE_SELECTOR = "exact"
SPARSE_FRAC = 0.1
SPARSE_PAGE = 16
Q_CHUNK = 256          # query-chunk for the allocation-bounded sparse path
K_CHUNK = 2048         # key-chunk for streaming top-k
LOCAL_W = 0            # recent-window tokens forced into every selector's top-k (fairness)
ATTEND_FP8 = False     # legacy alias for ATTEND_MODE == "fp8" (kept for back-compat)
ATTEND_MODE = "bf16"   # precision of the attend cache after selection (composition axis):
                       # "bf16" (exact), "fp8" (e4m3 roundtrip), "int4" (asym fakequant).
                       # Lets us measure selection-loss and attend-quant-loss together.


def _topk_mask(sel_scores, k):
    """Return additive mask (0 keep / -inf drop) of top-k per query row over last dim."""
    idx = sel_scores.topk(k, dim=-1).indices
    keep = torch.zeros_like(sel_scores, dtype=torch.bool).scatter_(-1, idx, True)
    return torch.where(keep, 0.0, float("-inf"))


def _quest_mask(q_r, k_r, attn_mask, k_budget, page):
    """Quest-style page selection: top pages by channel min/max upper bound."""
    B, H, S, D = k_r.shape
    np_ = (S + page - 1) // page
    pad = np_ * page - S
    kp = k_r
    if pad:
        kp = torch.cat([k_r, k_r[:, :, -1:, :].expand(B, H, pad, D)], dim=2)
    kp = kp.view(B, H, np_, page, D)
    kmin = kp.min(dim=3).values            # [B,H,np,D]
    kmax = kp.max(dim=3).values
    # upper bound on q·k for any key in page: sum_d max(q*kmin, q*kmax)
    qq = q_r.unsqueeze(3)                   # [B,H,S,1,D]
    bound = torch.maximum(qq * kmin.unsqueeze(2), qq * kmax.unsqueeze(2)).sum(-1)  # [B,H,S,np]
    # causal: page p valid for query i if p*page <= i
    pos = torch.arange(S, device=k_r.device)
    pstart = torch.arange(np_, device=k_r.device) * page
    valid = pstart.view(1, 1, 1, np_) <= pos.view(1, 1, S, 1)
    bound = torch.where(valid, bound, float("-inf"))
    npick = max(1, (k_budget + page - 1) // page)
    npick = min(npick, np_)
    pidx = bound.topk(npick, dim=-1).indices           # [B,H,S,npick]
    pkeep = torch.zeros_like(bound, dtype=torch.bool).scatter_(-1, pidx, True)  # [B,H,S,np]
    # expand page-keep to key-keep
    keep = pkeep.repeat_interleave(page, dim=-1)[:, :, :, :S]  # [B,H,S,S]
    return torch.where(keep, 0.0, float("-inf"))


def patch_qwen_attention(model):
    """
    Patches attention modules (Qwen2 and Llama, see PATCHABLE_ATTN_CLASSES) to
    intercept and quantize the KV cache during forward passes.
    """
    for name, module in model.named_modules():
        if module.__class__.__name__ in PATCHABLE_ATTN_CLASSES:
            layer_idx = module.layer_idx
            
            # Save original forward
            module.orig_forward = module.forward
            
            # Override forward
            def make_new_forward(mod, idx):
                def new_forward(
                    hidden_states,
                    position_embeddings,
                    attention_mask=None,
                    past_key_values=None,
                    **kwargs
                ):
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, mod.head_dim)
                    
                    q = mod.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    k = mod.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    v = mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    
                    k_orig = k.clone()
                    
                    # 2. Intercept for calibration or quantization (before RoPE)
                    if MODE == "collect":
                        if idx not in CALIBRATION_DATA:
                            CALIBRATION_DATA[idx] = {"q": [], "k": [], "v": []}
                        # Store in CPU/float32 to avoid VRAM bloat
                        CALIBRATION_DATA[idx]["q"].append(q.detach().float().cpu())
                        CALIBRATION_DATA[idx]["k"].append(k.detach().float().cpu())
                        CALIBRATION_DATA[idx]["v"].append(v.detach().float().cpu())
                        
                    elif MODE == "quantize" and idx in QUANT_CONFIGS:
                        cfg = QUANT_CONFIGS[idx]
                        
                        # A. Quantize Keys
                        if cfg["method"] == "norot":
                            # Standard PQ
                            pq_k = cfg["pq_k"]
                            k_idx = pq_k.quantize(k.float())
                            k_quant = pq_k.dequantize(k_idx).to(k.dtype)
                        elif cfg["method"] in ["rotated_std", "rotated_cal", "rotated_cal_sign"]:
                            pq_k = cfg["pq_k"]
                            sign_pattern = cfg["sign_pattern"]
                            
                            # Rotate -> Quantize -> Dequantize -> Inverse Rotate
                            k_rot = rotate_vectors(k.float(), sign_pattern)
                            k_idx = pq_k.quantize(k_rot)
                            k_rot_hat = pq_k.dequantize(k_idx)
                            # Inverse rotation: X = (X_rot * diag(s)) * H
                            k_quant = rotate_vectors(k_rot_hat * sign_pattern, None).to(k.dtype)
                        elif cfg["method"] == "pq_outlier":
                            pq_k = cfg["pq_k"]
                            sign_pattern = cfg["sign_pattern"]
                            outlier_indices = cfg["outlier_indices"]
                            H_k = k.shape[1]
                            
                            # Dense part (head-aware)
                            k_dense = k.float().clone()
                            if outlier_indices is not None and outlier_indices.shape[1] > 0:
                                for h in range(H_k):
                                    k_dense[:, h, :, outlier_indices[h]] = 0.0
                            
                            if isinstance(sign_pattern, str) and sign_pattern == "no_rot":
                                k_idx = pq_k.quantize(k_dense)
                                k_dense_recon = pq_k.dequantize(k_idx).to(k.dtype)
                            else:
                                k_rot = rotate_vectors(k_dense, sign_pattern)
                                k_idx = pq_k.quantize(k_rot)
                                k_rot_hat = pq_k.dequantize(k_idx)
                                k_dense_recon = rotate_vectors(k_rot_hat * sign_pattern, None).to(k.dtype)
                            
                            # Reinject unquantized outliers (head-aware)
                            k_quant = k_dense_recon.clone()
                            if outlier_indices is not None and outlier_indices.shape[1] > 0:
                                for h in range(H_k):
                                    k_quant[:, h, :, outlier_indices[h]] = k[:, h, :, outlier_indices[h]]
                        elif cfg["method"] == "turbo_quant":
                            # TurboQuant: random-orthogonal rotation + per-channel scalar dequant
                            R = cfg["R"].to(k.device)
                            scalar_q = cfg["scalar_q"]
                            k_rot = torch.matmul(k.float(), R)          # [..., D]
                            k_idx = scalar_q.quantize(k_rot)
                            k_rot_hat = scalar_q.dequantize(k_idx)
                            k_quant = torch.matmul(k_rot_hat, R.T).to(k.dtype)
                        elif cfg["method"] == "turbo_quant_qjl":
                            R = cfg["R"].to(k.device)
                            scalar_q = cfg["scalar_q"]
                            P_jl = cfg["P_jl"].to(k.device)   # [m, D]
                            m_jl = P_jl.shape[0]
                            jl_scale = math.sqrt(math.pi / (2 * m_jl))
                            
                            # Rotate -> scalar quantize
                            k_rot = torch.matmul(k.float(), R)     # [B, H, T, D]
                            k_idx = scalar_q.quantize(k_rot)
                            k_hat_rot = scalar_q.dequantize(k_idx)
                            
                            # Residual; separate direction from magnitude
                            r = k_rot - k_hat_rot                  # [B, H, T, D]
                            r_norm = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B,H,T,1]
                            r_hat  = r / r_norm                    # unit-length direction
                            
                            # 1-bit JL sketch of unit residual (stored at cache-write)
                            b = torch.sign(
                                torch.einsum('md,bhtd->bhtm', P_jl, r_hat)
                            )                                       # [B, H, T, m]
                            
                            # Reconstruct residual: E[P.T @ b * r_norm * jl_scale] = r  (exact)
                            r_approx = (
                                torch.einsum('md,bhtm->bhtd', P_jl, b)
                                * r_norm * jl_scale
                            )                                       # [B, H, T, D]
                            
                            # Corrected key in original space
                            k_quant = torch.matmul(k_hat_rot + r_approx, R.T).to(k.dtype)
                        elif cfg["method"] == "kvquant":
                            # KVQuant: pre-RoPE per-channel non-uniform quant +
                            # dense-and-sparse outliers (external baseline).
                            k_quant = cfg["kvq"].quantize_key(k)
                        elif cfg["method"] == "kivi":
                            # KIVI quantizes the key cache post-RoPE; defer to the
                            # post-RoPE block below. Leave keys untouched here.
                            k_quant = k
                        else:
                            k_quant = k

                        # B. Quantize Values. Default: each method's native value
                        # cache (pq_outlier=FP8, PQ-family=PQ, KIVI/KVQuant=own).
                        # An explicit cfg["value_mode"] overrides the default so any
                        # key method can be paired with either value cache. The
                        # two-value-cache ablation (Table 1) uses this to separate
                        # the key-method axis from the (dominant) value-cache axis.
                        vmode = cfg.get("value_mode")
                        if vmode is None:
                            if cfg["method"] == "pq_outlier":
                                vmode = "fp8"
                            elif cfg["method"] == "kivi":
                                vmode = "kivi"
                            elif cfg["method"] == "kvquant":
                                vmode = "kvquant"
                            elif "pq_v" in cfg:
                                vmode = "pq"
                            else:
                                vmode = "none"

                        if vmode == "fp8":
                            # Simulated FP8 (e4m3fn) Value cache
                            v_quant = v.to(torch.float8_e4m3fn).to(v.dtype)
                        elif vmode == "kivi":
                            v_quant = cfg["kivi_q"].quantize_value(v)
                        elif vmode == "kvquant":
                            v_quant = cfg["kvq"].quantize_value(v)
                        elif vmode == "pq":
                            pq_v = cfg["pq_v"]
                            v_idx = pq_v.quantize(v.float())
                            v_quant = pq_v.dequantize(v_idx).to(v.dtype)
                        else:
                            v_quant = v
                            
                        k = k_quant
                        v = v_quant
 
                    cos, sin = position_embeddings
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)

                    # KIVI quantizes the KEY cache as stored, i.e. post-RoPE — its
                    # per-channel outlier structure is defined on the rotated keys.
                    if (MODE == "quantize" and idx in QUANT_CONFIGS
                            and QUANT_CONFIGS[idx].get("method") == "kivi"):
                        k = QUANT_CONFIGS[idx]["kivi_q"].quantize_key(k)

                    # Update cache
                    if past_key_values is not None:
                        k, v = past_key_values.update(k, v, idx)
                        
                    # B2: fixed-budget top-k sparse attention (exact values+scores
                    # over a top-k chosen by the configured selector).
                    # KIVI quality baseline (near-lossless full attention, the iso-quality
                    # target for the sparse latency win): per-channel INT key + per-token INT
                    # value + 128-token FP16 residual, then exact dense attention.
                    if MODE == "sparse" and SPARSE_SELECTOR in ("kivi4", "kivi2"):
                        from src.quantization import KIVIQuantizer
                        kqz = KIVIQuantizer(bits=(4 if SPARSE_SELECTOR == "kivi4" else 2),
                                            group_size=32, residual_length=128)
                        k_dq = kqz.quantize_key(k)
                        v_dq = kqz.quantize_value(v)
                        ao = dense_attention(q, k_dq, v_dq, mod.num_key_value_groups, mod.scaling)
                        ao = ao.transpose(1, 2).contiguous().reshape(*input_shape, -1)
                        return mod.o_proj(ao), None

                    if MODE == "sparse" and idx in QUANT_CONFIGS:
                        # Allocation-bounded chunked sparse attention
                        # (benchmarks/sparse_attention.py). q, k are already RoPE'd
                        # here; v is exact. Numerically identical to the old dense
                        # [S,S] path (test_sparse_eval.py) but O(Q_CHUNK*K_CHUNK)
                        # memory, so it scales to long context. Assumes pure causal
                        # masking (single sequence, no padding) as in PPL eval.
                        sel_override = None
                        if SPARSE_SELECTOR in SOFT_SELECTORS:
                            sel_override = make_selection_keys(
                                SPARSE_SELECTOR, k_orig, QUANT_CONFIGS[idx], cos, sin,
                                rotate_vectors, apply_rotary_pos_emb)
                        k_att, v_att = k, v
                        att_mode = "fp8" if ATTEND_FP8 else ATTEND_MODE
                        if att_mode == "fp8":
                            # fp8-roundtrip the attend cache (selection still uses sel_override);
                            # mirrors the fp8 gather kernel: gather fp8 K/V, exact softmax over them.
                            k_att = k.to(torch.float8_e4m3fn).to(k.dtype)
                            v_att = v.to(torch.float8_e4m3fn).to(v.dtype)
                        elif att_mode == "int4":
                            # int4 asym fakequant of the attend cache (composition axis):
                            # selection picks keys, then attend over a 4-bit-quantized K/V.
                            # Same fakequant semantics as the int4 selection key code.
                            from src.quantization import _asym_uniform_fakequant
                            k_att = _asym_uniform_fakequant(k.float(), levels=16, dim=-1).to(k.dtype)
                            v_att = _asym_uniform_fakequant(v.float(), levels=16, dim=-1).to(v.dtype)
                        ao = sparse_attention_chunked(
                            q, k_att, v_att, sel_override, SPARSE_SELECTOR, SPARSE_FRAC,
                            mod.num_key_value_groups, mod.scaling,
                            page=SPARSE_PAGE, q_chunk=Q_CHUNK, k_chunk=K_CHUNK,
                            local_w=LOCAL_W)
                        ao = ao.transpose(1, 2).contiguous().reshape(*input_shape, -1)
                        return mod.o_proj(ao), None

                    # 4. Standard Attention compute
                    key_states = repeat_kv(k, mod.num_key_value_groups)
                    value_states = repeat_kv(v, mod.num_key_value_groups)
                    
                    q_for_score = q
                    if MODE == "quantize" and idx in QUANT_CONFIGS:
                        cfg = QUANT_CONFIGS[idx]
                        if cfg.get("share_lut", False):
                            q_for_score = q[:, :1].expand(-1, q.shape[1], -1, -1)
                    
                    attn_weights = torch.matmul(q_for_score, key_states.transpose(-1, -2)) * mod.scaling
                    
                    if attention_mask is not None:
                        attn_weights = attn_weights + attention_mask
                        
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
                    
                    # Compute Reference Attention (FP16) for KL Divergence
                    if MODE == "quantize":
                        q_ref, k_ref = apply_rotary_pos_emb(q_for_score, k_orig, cos, sin)
                        # We reconstruct the reference key cache by replacing the current token key with k_ref
                        k_ref_full = k.clone()
                        k_ref_full[:, :, -k_ref.shape[2]:] = k_ref
                        key_states_ref = repeat_kv(k_ref_full, mod.num_key_value_groups)
                        
                        attn_weights_ref = torch.matmul(q_for_score, key_states_ref.transpose(-1, -2)) * mod.scaling
                        if attention_mask is not None:
                            attn_weights_ref = attn_weights_ref + attention_mask
                        attn_weights_ref = nn.functional.softmax(attn_weights_ref, dim=-1, dtype=torch.float32)
                        
                        # Calculate KL Divergence: ref log(ref/quant)
                        kl = attn_weights_ref * (torch.log(attn_weights_ref + 1e-12) - torch.log(attn_weights + 1e-12))
                        kl_val = kl.sum(dim=-1).mean().item()
                        if idx not in ATTN_KL_DIVS:
                            ATTN_KL_DIVS[idx] = []
                        ATTN_KL_DIVS[idx].append(kl_val)

                        # Pilot B1: top-k selection recall (approx vs exact) at fixed
                        # context positions. Recall = |top-k(approx) ∩ top-k(exact)|/k.
                        if COMPUTE_RECALL:
                            Bsz, Hh, Sq, Sk = attn_weights_ref.shape
                            for ctx in RECALL_CTXS:
                                if ctx >= Sq:
                                    continue
                                ref_row = attn_weights_ref[:, :, ctx, :ctx + 1]   # [B,H,ctx+1]
                                apx_row = attn_weights[:,     :, ctx, :ctx + 1]
                                for frac in RECALL_FRACS:
                                    kk = max(1, int(round(frac * (ctx + 1))))
                                    t_idx = ref_row.topk(kk, dim=-1).indices
                                    a_idx = apx_row.topk(kk, dim=-1).indices
                                    tm = torch.zeros_like(ref_row).scatter_(-1, t_idx, 1.0)
                                    am = torch.zeros_like(apx_row).scatter_(-1, a_idx, 1.0)
                                    rec = ((tm * am).sum(-1) / kk).mean().item()
                                    RECALL_STATS.setdefault((ctx, frac), []).append(rec)
                    
                    attn_output = torch.matmul(attn_weights.to(q.dtype), value_states)
                    
                    # Reshape back and project
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(*input_shape, -1)
                    attn_output = mod.o_proj(attn_output)
                    
                    return attn_output, attn_weights
                
                return new_forward
            
            module.forward = make_new_forward(module, layer_idx)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Architecture-agnostic RoPE; identical math in Llama and Qwen2 modeling code."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Attention module class names this hook knows how to patch (Qwen2 + Llama share
# the same projection layout, GQA scheme, and RoPE math).
PATCHABLE_ATTN_CLASSES = ("Qwen2Attention", "LlamaAttention")

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats Key/Value heads for Grouped Query Attention (GQA).
    """
    if n_rep == 1:
        return hidden_states
    bs, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bs, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bs, num_key_value_heads * n_rep, seq_len, head_dim)

def evaluate_perplexity(model, tokenizer, dataset, max_samples=30, seq_len=512):
    """
    Calculates Perplexity (PPL) on the dataset and return (PPL, mean_kl).
    """
    global ATTN_KL_DIVS
    ATTN_KL_DIVS.clear()
    model.eval()
    nlls = []
    
    # Tokenize dataset using space separation
    encodings = tokenizer(" ".join(dataset["text"]), return_tensors="pt")
    input_ids = encodings.input_ids[0]
    
    num_tokens = input_ids.shape[0]
    num_eval_tokens = min(num_tokens, max_samples * seq_len)
    
    print(f"Evaluating perplexity on {num_eval_tokens} tokens...")
    
    with torch.no_grad():
        for i in range(0, num_eval_tokens - seq_len, seq_len):
            inputs = input_ids[i : i + seq_len].unsqueeze(0).to(model.device)
            targets = inputs.clone()
            
            outputs = model(inputs, labels=targets)
            neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood * seq_len)
            
    if len(nlls) == 0:
        return float('nan'), 0.0
        
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seq_len)).item()
    
    # Calculate mean KL divergence
    if len(ATTN_KL_DIVS) > 0:
        all_kls = [sum(v)/len(v) for v in ATTN_KL_DIVS.values()]
        mean_kl = sum(all_kls) / len(all_kls)
    else:
        mean_kl = 0.0
        
    return ppl, mean_kl

def run_calibration_and_training(model, tokenizer, calibration_text, args):
    global MODE, CALIBRATION_DATA, QUANT_CONFIGS
    bits_k = args.bits_k
    bits_v = args.bits_v
    d_sub = args.d_sub
    outliers = args.outliers
    no_rot = args.no_rot
    share_lut = args.share_lut
    selection_only = getattr(args, "selection_only", False)
    
    # Step 1: Collect calibration data
    print("Collecting calibration activations...")
    MODE = "collect"
    CALIBRATION_DATA.clear()
    
    encodings = tokenizer(calibration_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    
    with torch.no_grad():
        _ = model(input_ids)
        
    # Step 2: Fit and calibrate quantizers for each layer
    print("Fitting and calibrating quantizers layer-by-layer...")
    MODE = "baseline"
    
    device = model.device
    num_layers = len(CALIBRATION_DATA)
    D = model.config.hidden_size // model.config.num_attention_heads
    
    for idx in range(num_layers):
        print(f"--- Layer {idx+1}/{num_layers} ---")
        
        q_collected = torch.cat(CALIBRATION_DATA[idx]["q"], dim=2)
        k_collected = torch.cat(CALIBRATION_DATA[idx]["k"], dim=2)
        v_collected = torch.cat(CALIBRATION_DATA[idx]["v"], dim=2)
        
        G = model.config.num_attention_heads // model.config.num_key_value_heads
        
        # Repeat key/value along head dimension (dim 1) to match GQA head counts
        k_repeated = k_collected[:, :, None, :, :].expand(-1, -1, G, -1, -1).reshape(
            k_collected.shape[0], k_collected.shape[1] * G, k_collected.shape[2], k_collected.shape[3]
        )
        v_repeated = v_collected[:, :, None, :, :].expand(-1, -1, G, -1, -1).reshape(
            v_collected.shape[0], v_collected.shape[1] * G, v_collected.shape[2], v_collected.shape[3]
        )
        
        q_all = q_collected.transpose(0, 1).contiguous().view(-1, D).to(device)
        k_all = k_repeated.transpose(0, 1).contiguous().view(-1, D).to(device)
        v_all = v_repeated.transpose(0, 1).contiguous().view(-1, D).to(device)
        
        M = D // d_sub

        # Long-context selector studies only consume pq_outlier. Avoid fitting the
        # value cache and the full compression-ablation baselines at every layer.
        # This preserves the pq_outlier recipe: calibrated default-sign seed PQ ->
        # sign-selection error sign search -> refit/calibrate after outlier removal.
        if selection_only:
            sign_ones = torch.ones(D, device=device)
            k_rot_ones = rotate_vectors(k_all, sign_ones)
            pq_seed = ProductQuantizer(D, d_sub, bits_k, device=device)
            pq_seed.fit(k_rot_ones, num_iters=10)
            calibrate_codebook(q_all, k_all, pq_seed, lr=5e-3, steps=150, batch_size=512)

            H_k = model.config.num_key_value_heads
            variances = torch.var(k_collected[0].to(device), dim=1)
            if outliers > 0:
                outlier_indices = torch.stack(
                    [torch.topk(variances[h], k=outliers).indices for h in range(H_k)])
            else:
                outlier_indices = torch.zeros((H_k, 0), dtype=torch.long, device=device)
            q_dense_h = q_all.view(model.config.num_attention_heads, -1, D).clone()
            k_dense_h = k_all.view(model.config.num_attention_heads, -1, D).clone()
            if outliers > 0:
                for h_q in range(model.config.num_attention_heads):
                    h_k = h_q // G
                    q_dense_h[h_q, :, outlier_indices[h_k]] = 0.0
                    k_dense_h[h_q, :, outlier_indices[h_k]] = 0.0
            q_dense, k_dense = q_dense_h.view(-1, D), k_dense_h.view(-1, D)
            if no_rot:
                best_sign_asym, k_rot_asym = "no_rot", k_dense
            else:
                best_sign_asym = search_best_sign_pattern(
                    q_dense, k_dense, pq_seed, num_candidates=32)
                k_rot_asym = rotate_vectors(k_dense, best_sign_asym)
            pq_k_asym = ProductQuantizer(D, d_sub, bits_k, device=device)
            pq_k_asym.fit(k_rot_asym, num_iters=10)
            calibrate_codebook(q_dense, k_dense, pq_k_asym,
                               sign_pattern=best_sign_asym, outlier_indices=None,
                               lr=5e-3, steps=150, batch_size=512)
            QUANT_CONFIGS[idx] = {"pq_outlier": {
                "pq_k": pq_k_asym, "sign_pattern": best_sign_asym,
                "outlier_indices": outlier_indices, "share_lut": share_lut,
                "method": "pq_outlier"}}
            continue
        
        # 1. Config A: Standard PQ (No rotation)
        pq_k_norot = ProductQuantizer(D, d_sub, bits_k, device=device)
        pq_k_norot.fit(k_all, num_iters=10)
        
        pq_v = ProductQuantizer(D, d_sub, bits_v, device=device)
        pq_v.fit(v_all, num_iters=10)
        
        # 2. Config B: Hadamard Rotation + Default Sign
        sign_ones = torch.ones(D, device=device)
        k_rot_ones = rotate_vectors(k_all, sign_ones)
        
        pq_k_rot_std = ProductQuantizer(D, d_sub, bits_k, device=device)
        pq_k_rot_std.fit(k_rot_ones, num_iters=10)
        
        # 3. Config C: Calibrated Codebook + Default Sign
        pq_k_rot_cal = ProductQuantizer(D, d_sub, bits_k, device=device)
        pq_k_rot_cal.fit(k_rot_ones, num_iters=10)
        # Calibrate inner product
        calibrate_codebook(q_all, k_all, pq_k_rot_cal, lr=5e-3, steps=150, batch_size=512)
        
        # 4. Config D: Calibrated Codebook + sign-selection error optimal sign search
        best_sign = search_best_sign_pattern(q_all, k_all, pq_k_rot_cal, num_candidates=32)
        k_rot_best = rotate_vectors(k_all, best_sign)
        
        pq_k_rot_best = ProductQuantizer(D, d_sub, bits_k, device=device)
        pq_k_rot_best.fit(k_rot_best, num_iters=10)
        # Note: pass best_sign to calibrate_codebook to ensure correct calibration!
        calibrate_codebook(q_all, k_all, pq_k_rot_best, sign_pattern=best_sign, lr=5e-3, steps=150, batch_size=512)
        
        # 5. Config E: Split-Domain Asymmetric PQ-LUT (3-bit/4-bit PQ-LUT Keys + FP8 Values + Outliers)
        H_k = model.config.num_key_value_heads
        k_collected_flat = k_collected[0].to(device)
        variances = torch.var(k_collected_flat, dim=1) # [H_k, D]
        
        if outliers > 0:
            outlier_indices = torch.zeros((H_k, outliers), dtype=torch.long, device=device)
            for h in range(H_k):
                outlier_indices[h] = torch.topk(variances[h], k=outliers).indices
        else:
            outlier_indices = torch.zeros((H_k, 0), dtype=torch.long, device=device)
            
        # Zero out outlier channels (head-aware)
        q_dense_reshaped = q_all.view(model.config.num_attention_heads, -1, D).clone()
        k_dense_reshaped = k_all.view(model.config.num_attention_heads, -1, D).clone()
        
        if outliers > 0:
            for h_q in range(model.config.num_attention_heads):
                h_k = h_q // G
                q_dense_reshaped[h_q, :, outlier_indices[h_k]] = 0.0
                k_dense_reshaped[h_q, :, outlier_indices[h_k]] = 0.0
            
        q_dense = q_dense_reshaped.view(-1, D)
        k_dense = k_dense_reshaped.view(-1, D)
        
        if no_rot:
            best_sign_asym = "no_rot"
            k_rot_asym = k_dense
        else:
            best_sign_asym = search_best_sign_pattern(q_dense, k_dense, pq_k_rot_cal, num_candidates=32)
            k_rot_asym = rotate_vectors(k_dense, best_sign_asym)
        
        pq_k_asym = ProductQuantizer(D, d_sub, bits_k, device=device)
        pq_k_asym.fit(k_rot_asym, num_iters=10)
        
        calibrate_codebook(q_dense, k_dense, pq_k_asym, sign_pattern=best_sign_asym, outlier_indices=None, lr=5e-3, steps=150, batch_size=512)
        
        # 6. Config F: TurboQuant baseline
        R_turbo = generate_random_orthogonal(D, device=device)
        k_rot_turbo = torch.matmul(k_all, R_turbo)          # [N, D]
        scalar_q_k = PerChannelScalarQuantizer(D, bits_k, device=device)
        scalar_q_k.fit(k_rot_turbo)
        
        # 7. Config G: TurboQuant + 1-bit QJL residual correction
        m_jl = D      # m == D => +1 bpw over plain TurboQuant
        P_jl = generate_jl_matrix(m_jl, D, device=device)   # [m, D]

        # 8. External baselines: KIVI (tuning-free) and KVQuant (calibrated nuq).
        #    KIVI needs no calibration — scales are computed at decode time. The
        #    short PPL eval (512 tokens) uses a one-group (32-token) residual so
        #    the residual FP16 tokens don't dominate effective bpw; the long-
        #    context serving/memory analysis uses KIVI's full 128-token default.
        kivi_4 = KIVIQuantizer(bits=4, group_size=32, residual_length=32)
        kivi_2 = KIVIQuantizer(bits=2, group_size=32, residual_length=32)

        # KVQuant fits per-channel non-uniform signposts from pre-RoPE keys.
        k_cal = k_collected[0].to(device)   # [H_k, N, D]
        kvq_4 = KVQuantNUQ(bits=4, sparse_frac=0.01, device=device); kvq_4.fit(k_cal)
        kvq_3 = KVQuantNUQ(bits=3, sparse_frac=0.01, device=device); kvq_3.fit(k_cal)
        kvq_2 = KVQuantNUQ(bits=2, sparse_frac=0.01, device=device); kvq_2.fit(k_cal)

        # Store all configs
        QUANT_CONFIGS[idx] = {
            "norot":           {"pq_k": pq_k_norot,    "pq_v": pq_v, "method": "norot"},
            "rotated_std":     {"pq_k": pq_k_rot_std,  "pq_v": pq_v, "sign_pattern": sign_ones, "method": "rotated_std"},
            "rotated_cal":     {"pq_k": pq_k_rot_cal,  "pq_v": pq_v, "sign_pattern": sign_ones, "method": "rotated_cal"},
            "rotated_cal_sign":{"pq_k": pq_k_rot_best, "pq_v": pq_v, "sign_pattern": best_sign, "method": "rotated_cal_sign"},
            "pq_outlier":    {"pq_k": pq_k_asym,
                                "sign_pattern": best_sign_asym,
                                "outlier_indices": outlier_indices,
                                "share_lut": share_lut,
                                "method": "pq_outlier"},
            "turbo_quant":     {"R": R_turbo, "scalar_q": scalar_q_k, "method": "turbo_quant"},
            "turbo_quant_qjl": {"R": R_turbo, "scalar_q": scalar_q_k,
                                "P_jl": P_jl,
                                "method": "turbo_quant_qjl"},
            "kivi":            {"kivi_q": kivi_4, "method": "kivi"},
            "kivi_2bit":       {"kivi_q": kivi_2, "method": "kivi"},
            "kvquant":         {"kvq": kvq_4, "method": "kvquant"},
            "kvquant_3bit":    {"kvq": kvq_3, "method": "kvquant"},
            "kvquant_2bit":    {"kvq": kvq_2, "method": "kvquant"},
        }
        
    CALIBRATION_DATA.clear()
    print("Calibration complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--d-sub", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--outliers", type=int, default=2)
    parser.add_argument("--share-lut", action="store_true")
    parser.add_argument("--no-rot", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated subset of methods to evaluate (default: all).")
    parser.add_argument("--value-mode", type=str, default="native",
                        choices=["native", "fp8", "pq"],
                        help="Force the value cache for every method ('native' keeps "
                             "each method's own value quant). 'fp8' isolates the KEY "
                             "axis by holding all methods to a common FP8 value cache.")
    args = parser.parse_args()

    # Seed every randomness source (k-means init, random-orthogonal rotation, JL
    # sketch) so each run is exactly reproducible and baseline numbers trace to a
    # fixed seed. Vary --seed to characterise run-to-run variance.
    import random as _random
    import numpy as _np
    _random.seed(args.seed)
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {args.model}... (seed={args.seed})")
    
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Patch the attention layer
    patch_qwen_attention(model)
    
    # Load Wikitext-2
    print("Loading Wikitext-2 dataset...")
    dataset_train_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset_test_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset_fingerprint = hashlib.sha256(
        f"{dataset_train_raw._fingerprint}:{dataset_test_raw._fingerprint}".encode()
    ).hexdigest()

    # Filter empty or short lines to keep actual paragraphs. Publication runs fail
    # loudly on dataset errors; synthetic fallback text would invalidate PPL.
    train_lines = [line.strip() for line in dataset_train_raw["text"] if len(line.strip()) > 80]
    test_lines = [line.strip() for line in dataset_test_raw["text"] if len(line.strip()) > 80]
    dataset_train = {"text": train_lines}
    dataset_test = {"text": test_lines}
        
    # Get calibration text from filtered paragraphs
    calibration_text = " ".join(dataset_train["text"][:10])
    calibration_text = calibration_text[:6000]
    
    # Run baseline perplexity
    global MODE, QUANT_CONFIGS
    MODE = "baseline"
    print("=========================================================")
    ppl_base, kl_base = evaluate_perplexity(model, tokenizer, dataset_test, args.max_samples)
    print(f"FP16 Baseline Perplexity: {ppl_base:.4f} | KL: {kl_base:.6f}")
    print("=========================================================")
    
    # Run calibration
    run_calibration_and_training(model, tokenizer, calibration_text, args)
    
    # Evaluate each method
    # turbo_quant / turbo_quant_qjl: TurboQuant comparators (ICLR 2026)
    # kivi* / kvquant*: external SotA KV-quant baselines (KIVI ICML'24, KVQuant NeurIPS'24)
    # norot / rotated_*: internal ablation baselines
    methods = ["turbo_quant", "turbo_quant_qjl",
               "kivi", "kivi_2bit", "kvquant", "kvquant_3bit", "kvquant_2bit",
               "norot", "rotated_std", "rotated_cal", "rotated_cal_sign", "pq_outlier"]
    if args.methods is not None:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results = {}
    kl_results = {}
    
    for method in methods:
        print(f"\n=========================================================")
        print(f"Evaluating Perplexity for Method: {method.upper()}")
        print("=========================================================")
        
        # Promote this method's fields to the top-level config dict.
        # Done field-by-field so methods that lack pq_k / R / etc. don't crash.
        for idx in QUANT_CONFIGS:
            method_cfg = QUANT_CONFIGS[idx][method]
            QUANT_CONFIGS[idx]["method"] = method_cfg["method"]
            for field in ["pq_k", "pq_v", "sign_pattern", "outlier_indices",
                          "R", "scalar_q", "P_jl", "share_lut", "kivi_q", "kvq"]:
                if field in method_cfg:
                    QUANT_CONFIGS[idx][field] = method_cfg[field]
                elif field in QUANT_CONFIGS[idx]:
                    del QUANT_CONFIGS[idx][field]

            # Optionally force a common value cache so the comparison isolates the
            # KEY method (--value-mode fp8 -> all methods share an FP8 value cache).
            if args.value_mode != "native":
                QUANT_CONFIGS[idx]["value_mode"] = args.value_mode
                if args.value_mode == "pq" and "pq_v" not in QUANT_CONFIGS[idx]:
                    QUANT_CONFIGS[idx]["pq_v"] = QUANT_CONFIGS[idx]["norot"]["pq_v"]
            elif "value_mode" in QUANT_CONFIGS[idx]:
                del QUANT_CONFIGS[idx]["value_mode"]

        MODE = "quantize"
        ppl, kl = evaluate_perplexity(model, tokenizer, dataset_test, args.max_samples)
        results[method] = ppl
        kl_results[method] = kl
        print(f"Method {method.upper()} Perplexity: {ppl:.4f} | KL: {kl:.6f}")

    # Allow a method subset (--methods): backfill un-evaluated methods with nan so
    # the fixed summary table below doesn't KeyError.
    for _m in ["turbo_quant", "turbo_quant_qjl", "kivi", "kivi_2bit", "kvquant",
               "kvquant_3bit", "kvquant_2bit", "norot", "rotated_std",
               "rotated_cal", "rotated_cal_sign", "pq_outlier"]:
        results.setdefault(_m, float('nan'))
        kl_results.setdefault(_m, float('nan'))

    # bpw: turbo_quant_qjl stores m bits (JL sketch) + 16 bits (float16 r_norm) per key
    # => bpw = bits_k + (m_jl + 16) / D  = 4 + (64+16)/64 = 5.25 for D=64, m=64
    any_idx = next(iter(QUANT_CONFIGS))
    m_jl   = QUANT_CONFIGS[any_idx]["turbo_quant_qjl"]["P_jl"].shape[0]
    D_head = QUANT_CONFIGS[any_idx]["turbo_quant_qjl"]["P_jl"].shape[1]
    bpw_qjl = args.bits_k + (m_jl + 16) / D_head   # +16 for float16 per-token r_norm

    # Effective bpw for the external KV-quant baselines. KIVI's depends on the
    # eval context length (residual FP16 window); KVQuant's adds sparse overhead.
    seq_len_eval = 512   # evaluate_perplexity() default seq_len
    bpw_map = {
        "kivi":         QUANT_CONFIGS[any_idx]["kivi"]["kivi_q"].bpw(seq_len_eval),
        "kivi_2bit":    QUANT_CONFIGS[any_idx]["kivi_2bit"]["kivi_q"].bpw(seq_len_eval),
        "kvquant":      QUANT_CONFIGS[any_idx]["kvquant"]["kvq"].bpw(D_head),
        "kvquant_3bit": QUANT_CONFIGS[any_idx]["kvquant_3bit"]["kvq"].bpw(D_head),
        "kvquant_2bit": QUANT_CONFIGS[any_idx]["kvquant_2bit"]["kvq"].bpw(D_head),
    }

    print("\n=========================================================")
    print("Final Perplexity Summary Table:")
    print("=========================================================")
    print(f"| {'Method':<48} | {'PPL':>7} | {'bpw':>5} | {'KL Div':>8} | Category       |")
    print(f"|{'-'*50}|{'-'*9}|{'-'*7}|{'-'*10}|{'-'*16}|")
    print(f"| {'FP16 Baseline':<48} | {ppl_base:>7.4f} | {'16.0':>5} | {kl_base:>8.6f} | Reference      |")
    print(f"| {'TurboQuant (Rand-Orth + Scalar, no QJL)':<48} | {results['turbo_quant']:>7.4f} | {args.bits_k:>5.1f} | {kl_results['turbo_quant']:>8.6f} | External SotA  |")
    print(f"| {f'TurboQuant-QJL (Scalar + {m_jl}-bit JL residual)':<48} | {results['turbo_quant_qjl']:>7.4f} | {bpw_qjl:>5.1f} | {kl_results['turbo_quant_qjl']:>8.6f} | External SotA+ |")
    print(f"| {'KIVI-4 (per-channel INT + residual)':<48} | {results['kivi']:>7.4f} | {bpw_map['kivi']:>5.2f} | {kl_results['kivi']:>8.6f} | External SotA  |")
    print(f"| {'KIVI-2 (per-channel INT + residual)':<48} | {results['kivi_2bit']:>7.4f} | {bpw_map['kivi_2bit']:>5.2f} | {kl_results['kivi_2bit']:>8.6f} | External SotA  |")
    print(f"| {'KVQuant-4 (per-channel nuq + sparse)':<48} | {results['kvquant']:>7.4f} | {bpw_map['kvquant']:>5.2f} | {kl_results['kvquant']:>8.6f} | External SotA  |")
    print(f"| {'KVQuant-3 (per-channel nuq + sparse)':<48} | {results['kvquant_3bit']:>7.4f} | {bpw_map['kvquant_3bit']:>5.2f} | {kl_results['kvquant_3bit']:>8.6f} | External SotA  |")
    print(f"| {'KVQuant-2 (per-channel nuq + sparse)':<48} | {results['kvquant_2bit']:>7.4f} | {bpw_map['kvquant_2bit']:>5.2f} | {kl_results['kvquant_2bit']:>8.6f} | External SotA  |")
    print(f"| {f'{args.bits_k}-bit PQ (No rotation)':<48} | {results['norot']:>7.4f} | {args.bits_k:>5.1f} | {kl_results['norot']:>8.6f} | Ablation       |")
    print(f"| {f'{args.bits_k}-bit Rotated PQ (Default Sign)':<48} | {results['rotated_std']:>7.4f} | {args.bits_k:>5.1f} | {kl_results['rotated_std']:>8.6f} | Ablation       |")
    print(f"| {f'{args.bits_k}-bit rotated PQ + calibration':<48} | {results['rotated_cal']:>7.4f} | {args.bits_k:>5.1f} | {kl_results['rotated_cal']:>8.6f} | Ablation       |")
    print(f"| {f'{args.bits_k}-bit calibrated PQ + sign selection':<48} | {results['rotated_cal_sign']:>7.4f} | {args.bits_k:>5.1f} | {kl_results['rotated_cal_sign']:>8.6f} | Ablation       |")
    print(f"| {'PQ-LUT + outlier isolation':<48} | {results['pq_outlier']:>7.4f} | {args.bits_k:>5.1f} | {kl_results['pq_outlier']:>8.6f} | Ablation       |")
    print("=========================================================")
    
    tq_ppl      = results['turbo_quant']
    tq_qjl_ppl  = results['turbo_quant_qjl']
    ours_ppl    = results['pq_outlier']
    
    def rel(a, b): return f"{'+' if a>=b else ''}{100*(a-b)/b:.1f}%"
    print(f"\nKey comparisons (PPL delta vs baseline)")
    print(f"  TurboQuant (no QJL):        {tq_ppl:.4f}  ({rel(tq_ppl, ppl_base)} vs FP16)")
    print(f"  TurboQuant-QJL ({m_jl}-bit):   {tq_qjl_ppl:.4f}  ({rel(tq_qjl_ppl, ppl_base)} vs FP16)")
    print(f"  PQ-LUT + outlier isolation:   {ours_ppl:.4f}  ({rel(ours_ppl, ppl_base)} vs FP16)")
    print("=========================================================")

    if args.output_json is not None:
        import json
        output_data = {
            "schema": "compression_seed_v1",
            "model": args.model,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "dataset": "wikitext/wikitext-2-raw-v1",
            "dataset_fingerprint": dataset_fingerprint,
            "command": " ".join(sys.argv),
            "fp16_ppl": ppl_base,
            "fp16_kl": kl_base,
            "results": results,
            "kl_results": kl_results,
            "bpw_qjl": bpw_qjl,
            "bpw_baselines": bpw_map,
            "bits_k": args.bits_k,
            "d_sub": args.d_sub,
            "outliers": args.outliers,
            "share_lut": args.share_lut,
            "no_rot": args.no_rot,
            "seed": args.seed
            ,"max_samples": args.max_samples,
            "value_mode": args.value_mode
        }
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Saved perplexity results to {args.output_json}")

if __name__ == "__main__":
    main()
