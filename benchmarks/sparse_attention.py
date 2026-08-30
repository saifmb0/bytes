"""Allocation-bounded primitives for dynamic sparse attention (Option B).

The dense `MODE=="sparse"` path in eval_ppl.py materializes the full [B,H,S,S]
score matrix and OOMs past ~8K context. These primitives compute the same thing
in key-chunks with a streaming top-k, so peak memory is O(Q_CHUNK * K_CHUNK)
rather than O(S^2), letting the long-context selection experiments (Phase A2)
run to 16K-128K on real models.

Semantics are kept IDENTICAL to the dense branch so test_sparse_eval can assert
equivalence at small S:
  - fixed budget kb = round(frac * S) applied to every query row,
  - top-kb keys by the selector's score among causal-valid positions (<= query pos),
  - OUTPUT computed with EXACT scores+values over the selected set (selection and
    output score spaces are decoupled: we may select by approx but attend by exact).

Selectors:
  exact   - top-k by exact post-RoPE q·k                       (oracle upper bound)
  approx  - top-k by LUT-equivalent q·k_quant (PQ-reconstructed) (OURS)
  quest   - top pages by channel min/max upper bound            (baseline)
  recent  - most-recent kb keys                                 (cheap baseline / C3)
  random  - random kb keys                                      (floor / C3)

H2O and SnapKV need attention-mass accumulation across the prefill; they are added
in the long-context driver step (A2) on top of these primitives.
"""
import torch
import torch.nn.functional as F
from src.quantization import _asym_uniform_fakequant

NEG_INF = float("-inf")

# Selectors whose ranking uses an APPROXIMATE key (a "scoring oracle"); the
# selection-key tensor is passed via `sel_override`. All other selectors rank in
# the exact-key space (or ignore keys). Used by the A3 oracle bake-off (C3).
SOFT_SELECTORS = ("approx", "int4", "int2", "fp8", "signvq")

# SparQ Attention: score keys using only the top-r highest-|q| query channels
# (reads r/D of each key -> cheap scan). Set by the driver. The closest per-key peer.
SPARQ_R = 16

# Self-Indexing KVCache (arXiv 2603.14224): sign-based 1-bit vector quantization on
# d-dim subvectors (2^SIGNVQ_SUB sign clusters) + a low-bit per-subvector magnitude.
# Faithful reimplementation as a scoring oracle for the matched-byte selection study.
# Scoring cost/key (byte-aligned, D dims): SIGNVQ_SUB sign bits + log2(levels) mag
# bits per subvector -> (D/SIGNVQ_SUB)*(SIGNVQ_SUB + log2(levels))/8 bytes.
SIGNVQ_SUB = 4
SIGNVQ_MAG_LEVELS = 4    # magnitude quant levels (4 = 2-bit, 2 = 1-bit)
SIGNVQ_MAG_PERDIM = True  # True: 2-bit magnitude per element (faithful "abs-value quant");
#                           False: one magnitude per SIGNVQ_SUB-dim subvector (cheaper VQ reading)


def signvq_bytes_per_key(D, perdim=None, levels=None):
    """Byte-aligned scoring cost of the sign-VQ key code at head-dim D.
    sign: 1 bit/dim. magnitude: log2(levels) bits per element (perdim) or per subvector."""
    import math
    perdim = SIGNVQ_MAG_PERDIM if perdim is None else perdim
    levels = SIGNVQ_MAG_LEVELS if levels is None else levels
    mag_bits = int(math.log2(levels))
    sign_bits = D
    mag = D * mag_bits if perdim else (D // SIGNVQ_SUB) * mag_bits
    return (sign_bits + mag) / 8.0


def _signvq_reconstruct(k):
    """Faithful Self-Indexing sign-VQ key reconstruction (pre-RoPE).

    Each SIGNVQ_SUB-dim subvector picks a sign pattern (one of 2^SIGNVQ_SUB sign
    clusters); magnitudes are quantized to SIGNVQ_MAG_LEVELS. Per-element magnitude
    (SIGNVQ_MAG_PERDIM=True, the faithful "2-bit quant of absolute values" reading)
    or one magnitude per subvector (False, the cheaper VQ reading). k:[B,H,S,D]->same.
    """
    B, H, S, D = k.shape
    sub = SIGNVQ_SUB
    assert D % sub == 0, f"D={D} not divisible by SIGNVQ_SUB={sub}"
    sign = torch.sign(k.float())
    if SIGNVQ_MAG_PERDIM:
        mag_q = _asym_uniform_fakequant(k.float().abs(), levels=SIGNVQ_MAG_LEVELS, dim=-1)
        return (sign * mag_q).to(k.dtype)
    kr = k.float().reshape(B, H, S, D // sub, sub)
    mag = kr.abs().mean(dim=-1, keepdim=True)                   # [B,H,S,G,1]
    mag_q = _asym_uniform_fakequant(mag, levels=SIGNVQ_MAG_LEVELS, dim=3)
    return (torch.sign(kr) * mag_q).reshape(B, H, S, D).to(k.dtype)


def repeat_kv(x, n_rep):
    """[B,H_k,S,D] -> [B,H_k*n_rep,S,D] (GQA expansion), matching eval_ppl.repeat_kv."""
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)


def reconstruct_approx_keys(k_orig, cfg, cos, sin, rotate_vectors, apply_rope):
    """PQ-reconstruct the (post-RoPE) approximate keys used for `approx` selection.

    Mirrors the pq_outlier path in eval_ppl.py: zero outliers -> Hadamard rotate
    -> PQ quantize+dequantize -> inverse-rotate -> reinject exact outliers -> RoPE.
    Returns kq_roped [B, H_k, S, D] (same size as the key cache, NOT [S,S]).

    NOTE (kernel faithfulness, Phase B): this reconstructs in pre-RoPE space then
    applies RoPE, i.e. the score is post-RoPE q·k_quant. A static-LUT kernel computes
    pre-RoPE q_rot·k_hat; making the two identical requires post-RoPE PQ keys + a
    per-step LUT. That fork is resolved in Phase B; here we keep the post-RoPE approx
    that B1/B2 validated.
    """
    pqk = cfg["pq_k"]
    sp = cfg["sign_pattern"]
    oi = cfg["outlier_indices"]
    H_k = k_orig.shape[1]
    kd = k_orig.float().clone()
    if oi.shape[1] > 0:
        for h in range(H_k):
            kd[:, h, :, oi[h]] = 0.0
    if isinstance(sp, str) and sp == "no_rot":
        kq = pqk.dequantize(pqk.quantize(kd)).to(k_orig.dtype)
    else:
        kr = rotate_vectors(kd, sp)
        kq = rotate_vectors(pqk.dequantize(pqk.quantize(kr)) * sp, None).to(k_orig.dtype)
    if oi.shape[1] > 0:
        for h in range(H_k):
            kq[:, h, :, oi[h]] = k_orig[:, h, :, oi[h]]
    _, kq_r = apply_rope(torch.zeros_like(kq), kq, cos, sin)  # RoPE the keys only
    return kq_r


def make_selection_keys(selector, k_orig, cfg, cos, sin, rotate_vectors, apply_rope):
    """Post-RoPE approximate keys for a 'scoring oracle' selector (A3 / C3 bake-off).
    Returns None for selectors that rank in exact-key space or ignore keys.

    Scoring cost per key at D (the byte budget we hold matched in the bake-off):
      approx (PQ cb256/d_sub=2): M = D/d_sub uint8 indices  -> D/2 bytes
      int4 (per-token scalar):   D * 4 bits                  -> D/2 bytes  (matched!)
      fp8:                       D * 8 bits                  -> D bytes
      exact (reference ceiling): D * 16 bits                 -> 2D bytes
    """
    if selector == "approx":
        return reconstruct_approx_keys(k_orig, cfg, cos, sin, rotate_vectors, apply_rope)
    if selector == "int4":
        kq = _asym_uniform_fakequant(k_orig.float(), levels=16, dim=-1).to(k_orig.dtype)
        _, kq_r = apply_rope(torch.zeros_like(kq), kq, cos, sin)
        return kq_r
    if selector == "int2":  # 2-bit scalar (matches PQ d_sub=4/cb256 at 16 B/key, D=64)
        kq = _asym_uniform_fakequant(k_orig.float(), levels=4, dim=-1).to(k_orig.dtype)
        _, kq_r = apply_rope(torch.zeros_like(kq), kq, cos, sin)
        return kq_r
    if selector == "fp8":
        kq = k_orig.to(torch.float8_e4m3fn).to(k_orig.dtype)
        _, kq_r = apply_rope(torch.zeros_like(kq), kq, cos, sin)
        return kq_r
    if selector == "signvq":  # Self-Indexing sign-VQ (sign clusters + 2-bit magnitude)
        kq = _signvq_reconstruct(k_orig)
        _, kq_r = apply_rope(torch.zeros_like(kq), kq, cos, sin)
        return kq_r
    return None


# Selection/attend SCORES are accumulated in fp32. Bf16 q,k can mis-rank keys on some
# models, so a bf16 q.k "exact" oracle is not necessarily an upper bound. A controlled
# key-channel clipping intervention did not by itself repair the observed failure;
# higher-precision scoring removes the confound without asserting a single mechanism.
# SCORE_DTYPE selects the accumulation precision: "fp32" (default, true oracle), "fp16"
# (10 mantissa bits -- tests whether mantissa, not dynamic range, is the fix), or "bf16"
# (7 mantissa bits -- exposes the pitfall). SCORE_FP32 is a legacy alias: setting it
# False forces bf16 (the old --score-bf16 path) regardless of SCORE_DTYPE.
SCORE_DTYPE = "fp32"
SCORE_FP32 = True
# Optional controlled intervention for the numerical-pitfall study.  This affects
# only the keys used to *rank* candidates, never the exact keys/values used by
# attend_topk.  None preserves the publication default.
SCORE_KEY_CLIP = None

_DIAG_ENABLED = False
_DIAG_GUARD = False
_DIAG = {}


def reset_score_diagnostics(enabled=False):
    global _DIAG_ENABLED, _DIAG
    _DIAG_ENABLED = enabled
    _DIAG = {"score_tiles": 0, "topk_calls": 0, "topk_items": 0,
             "topk_matches": 0, "nonfinite_scores": 0,
             "max_abs_q": 0.0, "max_abs_k": 0.0,
             "max_abs_element_product": 0.0, "max_abs_score": 0.0}


def get_score_diagnostics():
    out = dict(_DIAG)
    n = out.get("topk_items", 0)
    out["topk_agreement_with_fp32"] = out.get("topk_matches", 0) / n if n else None
    return out

_SCORE_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _score_dtype():
    """Resolve the score-accumulation torch dtype, honoring the legacy SCORE_FP32 flag."""
    if not SCORE_FP32:
        return torch.bfloat16
    return _SCORE_DTYPES[SCORE_DTYPE]


def _scast(t):
    """Cast a tensor to the active score-accumulation dtype (no-op if already there)."""
    d = _score_dtype()
    return t if t.dtype == d else t.to(d)


def _score_tile(q_blk, k_tile, scaling):
    """q_blk [B,Hq,Qc,D], k_tile [B,Hq,Kc,D] -> scores [B,Hq,Qc,Kc] (score-dtype accum)."""
    sc = torch.matmul(_scast(q_blk), _scast(k_tile).transpose(-1, -2)) * scaling
    if _DIAG_ENABLED and not _DIAG_GUARD:
        qf, kf = q_blk.float(), k_tile.float()
        _DIAG["score_tiles"] += 1
        _DIAG["max_abs_q"] = max(_DIAG["max_abs_q"], qf.abs().max().item())
        _DIAG["max_abs_k"] = max(_DIAG["max_abs_k"], kf.abs().max().item())
        _DIAG["max_abs_element_product"] = max(
            _DIAG["max_abs_element_product"],
            (qf.abs().amax(dim=-2, keepdim=True) * kf.abs().amax(dim=-2).unsqueeze(-2)).max().item())
        finite = torch.isfinite(sc)
        _DIAG["nonfinite_scores"] += int((~finite).sum().item())
        if finite.any():
            _DIAG["max_abs_score"] = max(_DIAG["max_abs_score"], sc[finite].abs().max().item())
    return sc


def select_topk(selector, q_blk, sel_keys, q_pos, S, kb, k_chunk, scaling,
                page=16, generator=None, local_w=0):
    """Return top-kb causal key indices per query row.

    q_blk:    [B,Hq,Qc,D] post-RoPE queries for this query chunk
    sel_keys: [B,Hq,S,D]  keys in the SELECTOR's score space (exact->k_r, approx->kq_r);
              ignored for 'recent'/'random'; for 'quest' it is k_r (exact keys).
    q_pos:    [Qc] global positions of the query rows
    local_w:  reserve the most-recent `local_w` causal tokens for EVERY selector
              (forced into the top-k), matching Quest/H2O/SnapKV which always keep a
              recent window. Applied uniformly so the selector comparison is fair.
    returns idx [B,Hq,Qc,kb] (long); invalid/padding slots filled with 0 (they get
    -inf exact score at attend time and contribute no weight).
    """
    global _DIAG_GUARD, SCORE_DTYPE, SCORE_FP32
    B, Hq, Qc, D = q_blk.shape
    dev = q_blk.device
    pos = q_pos.to(dev)

    # A causal numerical intervention, not a production optimization: cap key
    # channels before score accumulation and ask whether bf16 ranking recovers.
    # It is deliberately restricted to exact/sparse score selectors, and leaves
    # the post-selection attention computation exact.
    if SCORE_KEY_CLIP is not None and selector in ("exact", "approx", "int4", "int2", "fp8", "signvq", "sparq"):
        sel_keys = sel_keys.clamp(min=-SCORE_KEY_CLIP, max=SCORE_KEY_CLIP)

    def _force_recent(sc, kpos):
        # set score = +inf for keys in the recent window (p-local_w, p] so topk
        # always keeps them; positions are distinct from the older method picks.
        if local_w <= 0:
            return sc
        rec = (kpos[None, None, None, :] > (pos[None, None, :, None] - local_w)) & \
              (kpos[None, None, None, :] <= pos[None, None, :, None])
        return torch.where(rec, float("inf"), sc)

    if selector == "recent":
        # kb most recent causal keys per row: positions [p-kb+1 .. p]
        ar = torch.arange(kb, device=dev)
        idx = (pos[:, None] - ar[None, :]).clamp(min=0)          # [Qc,kb]
        return idx[None, None].expand(B, Hq, Qc, kb).contiguous()

    if selector == "random":
        g = generator
        # sample kb of [0..p] per row (with clamp); cheap floor baseline
        rnd = torch.rand(B, Hq, Qc, kb, device=dev, generator=g)
        idx = (rnd * (pos[None, None, :, None].float() + 1)).long().clamp(min=0)
        return idx

    if selector == "quest":
        return _quest_select(q_blk, sel_keys, pos, S, kb, page, scaling, local_w)

    if selector == "sparq":
        # SparQ: zero all but the top-r highest-|q| channels per query row, then
        # score exactly with the masked query (identical to SparQ's r-channel dot).
        r = min(SPARQ_R, D)
        topr = q_blk.abs().topk(r, dim=-1).indices                # [B,Hq,Qc,r]
        m = torch.zeros_like(q_blk)
        m.scatter_(-1, topr, 1.0)
        q_blk = q_blk * m                                          # masked query
        # falls through to the streaming top-k below (sel_keys = exact keys)

    # exact / approx / sparq: streaming top-kb over key chunks.
    # Placeholder slots use the sentinel idx = S (out of range): when a row has
    # fewer valid causal keys than kb, these fillers must NOT alias a real key
    # (esp. position 0). attend_topk masks any idx >= S. Without this, -inf ties
    # resolve to the placeholder and double-count it (CUDA topk tie-break).
    run_vals = torch.full((B, Hq, Qc, kb), NEG_INF, device=dev)
    run_idx = torch.full((B, Hq, Qc, kb), S, dtype=torch.long, device=dev)
    max_key = int(pos.max().item()) + 1
    for kc0 in range(0, max_key, k_chunk):
        kc1 = min(kc0 + k_chunk, max_key)
        ktile = sel_keys[:, :, kc0:kc1, :]
        sc = _score_tile(q_blk, ktile, scaling)                  # [B,Hq,Qc,Kc]
        kpos = torch.arange(kc0, kc1, device=dev)
        causal = kpos[None, None, None, :] <= pos[None, None, :, None]
        sc = torch.where(causal, sc, NEG_INF)
        sc = _force_recent(sc, kpos)
        cand_vals = torch.cat([run_vals, sc], dim=-1)
        cand_idx = torch.cat(
            [run_idx, kpos[None, None, None, :].expand(B, Hq, Qc, kc1 - kc0)], dim=-1)
        kk = min(kb, cand_vals.shape[-1])
        tv, ti = cand_vals.topk(kk, dim=-1)
        run_vals[..., :kk] = tv
        run_idx[..., :kk] = torch.gather(cand_idx, -1, ti)
    if _DIAG_ENABLED and not _DIAG_GUARD and _score_dtype() != torch.float32:
        _DIAG_GUARD = True
        old_dtype, old_fp32 = SCORE_DTYPE, SCORE_FP32
        try:
            SCORE_DTYPE, SCORE_FP32 = "fp32", True
            ref_idx = select_topk(selector, q_blk, sel_keys, q_pos, S, kb, k_chunk,
                                  scaling, page=page, generator=generator, local_w=local_w)
        finally:
            SCORE_DTYPE, SCORE_FP32 = old_dtype, old_fp32
            _DIAG_GUARD = False
        # Set overlap, not positional equality: topk ordering is irrelevant to attention.
        matches = (run_idx.unsqueeze(-1) == ref_idx.unsqueeze(-2)).any(-1).sum().item()
        _DIAG["topk_calls"] += 1
        _DIAG["topk_matches"] += int(matches)
        _DIAG["topk_items"] += run_idx.numel()
    return run_idx


def _quest_select(q_blk, k_r, pos, S, kb, page, scaling, local_w=0):
    """Quest: select keys in the top pages by channel-bound, then take their indices.
    Bound for a page = sum_d max(q_d*kmin_d, q_d*kmax_d). Causal at page granularity;
    final attend re-applies exact causal so partial-future pages are harmless."""
    B, Hq, Qc, D = q_blk.shape
    dev = q_blk.device
    np_ = (S + page - 1) // page
    padS = np_ * page
    kp = k_r
    if padS > S:
        kp = torch.cat([k_r, k_r[:, :, -1:, :].expand(B, Hq, padS - S, D)], dim=2)
    kp = kp.view(B, Hq, np_, page, D)
    kmin = kp.min(dim=3).values                                   # [B,Hq,np,D]
    kmax = kp.max(dim=3).values
    kmin, kmax, q_blk = _scast(kmin), _scast(kmax), _scast(q_blk)
    qq = q_blk.unsqueeze(3)                                       # [B,Hq,Qc,1,D]
    bound = torch.maximum(qq * kmin.unsqueeze(2), qq * kmax.unsqueeze(2)).sum(-1)  # [B,Hq,Qc,np]
    bound = bound * scaling
    pstart = torch.arange(np_, device=dev) * page
    valid = pstart[None, None, None, :] <= pos[None, None, :, None]
    bound = torch.where(valid, bound, NEG_INF)
    if local_w > 0:
        # force pages overlapping the recent window (p-local_w, p] to be selected
        pg = torch.arange(np_, device=dev)
        lo = ((pos - local_w + 1).clamp(min=0) // page)
        hi = (pos // page)
        rec_pg = (pg[None, :] >= lo[:, None]) & (pg[None, :] <= hi[:, None])  # [Qc,np]
        bound = torch.where(rec_pg[None, None], float("inf"), bound)
    npick = max(1, (kb + page - 1) // page)
    npick = min(npick, np_)
    pidx = bound.topk(npick, dim=-1).indices                     # [B,Hq,Qc,npick]
    # expand selected pages to key indices
    base = pidx * page                                           # [B,Hq,Qc,npick]
    off = torch.arange(page, device=dev)
    idx = (base[..., None] + off[None, None, None, None, :]).reshape(B, Hq, Qc, npick * page)
    idx = idx.clamp(max=S - 1)
    return idx


def attend_topk(q_blk, k_r, v, idx, q_pos, scaling):
    """Exact softmax attention over the selected keys.
    q_blk [B,Hq,Qc,D]; k_r,v [B,Hq,S,D]; idx [B,Hq,Qc,ksel]; q_pos [Qc].
    Returns [B,Hq,Qc,D]. Re-applies causal so any invalid/duplicate gathered index
    (past-the-end or future) is masked out -> matches the dense reference exactly.
    """
    B, Hq, Qc, D = q_blk.shape
    S = k_r.shape[2]
    ksel = idx.shape[-1]
    dev = q_blk.device
    # valid = causally allowed AND not a sentinel/out-of-range placeholder.
    valid = (idx <= q_pos.to(dev)[None, None, :, None]) & (idx < S)
    gidx_flat = idx.clamp(max=S - 1)                            # safe index for gather
    gidx = gidx_flat.unsqueeze(-1).expand(B, Hq, Qc, ksel, D)
    k_sel = torch.gather(k_r.unsqueeze(2).expand(B, Hq, Qc, S, D), 3, gidx)
    v_sel = torch.gather(v.unsqueeze(2).expand(B, Hq, Qc, S, D), 3, gidx)
    qf, ksf = _scast(q_blk), _scast(k_sel)
    sc = (qf.unsqueeze(3) * ksf).sum(-1) * scaling              # [B,Hq,Qc,ksel] (score-dtype accum)
    sc = torch.where(valid, sc, NEG_INF)
    w = F.softmax(sc, dim=-1, dtype=torch.float32).to(v_sel.dtype)
    return (w.unsqueeze(-1) * v_sel).sum(3)                      # [B,Hq,Qc,D]


def _attention_importance(q_r, kfull, scaling, mode, obs_window=32, pool=7, q_chunk=256):
    """Global per-key importance for the eviction/observation baselines (H2O, SnapKV).

    q_r, kfull: [B,Hq,S,D] post-RoPE queries and GQA-expanded exact keys.
    Returns imp [B,Hq,S] (fp32).
      h2o    - accumulated causal attention each key RECEIVES (column sum of the
               causal-softmax attention over all queries) = the heavy-hitter score.
      snapkv - attention from the last `obs_window` queries only, summed over that
               window, then 1D max-pooled over the key axis (SnapKV's clustering).
    Computed in query-chunks so peak memory is O(q_chunk * S), not O(S^2).
    """
    B, Hq, S, D = q_r.shape
    dev = q_r.device
    kpos = torch.arange(S, device=dev)
    if mode == "snapkv":
        q0 = max(0, S - obs_window)
        qb = q_r[:, :, q0:S, :]
        sc = torch.matmul(_scast(qb), _scast(kfull).transpose(-1, -2)) * scaling          # [B,Hq,W,S]
        qpos = torch.arange(q0, S, device=dev)
        causal = kpos[None, None, None, :] <= qpos[None, None, :, None]
        sc = torch.where(causal, sc, NEG_INF)
        A = torch.softmax(sc, dim=-1, dtype=torch.float32)
        imp = A.sum(2)                                                     # [B,Hq,S]
        if pool > 1:
            imp = F.max_pool1d(imp, kernel_size=pool, stride=1, padding=pool // 2)
        return imp
    # h2o: accumulate over all queries, chunked
    imp = torch.zeros(B, Hq, S, device=dev, dtype=torch.float32)
    for q0 in range(0, S, q_chunk):
        q1 = min(q0 + q_chunk, S)
        qb = q_r[:, :, q0:q1, :]
        sc = torch.matmul(_scast(qb), _scast(kfull).transpose(-1, -2)) * scaling
        qpos = torch.arange(q0, q1, device=dev)
        causal = kpos[None, None, None, :] <= qpos[None, None, :, None]
        sc = torch.where(causal, sc, NEG_INF)
        A = torch.softmax(sc, dim=-1, dtype=torch.float32)
        imp += A.sum(2)
    return imp


def _global_keepset_attend(q_r, kfull, vfull, imp, kb, scaling, q_chunk, local_w):
    """Attend each query to a GLOBAL heavy-hitter keep-set (top kb-local_w by imp)
    plus a per-query recent window of local_w. Used by h2o/snapkv. attend_topk
    re-applies causal masking so global-keep keys ahead of a query contribute 0."""
    B, Hq, S, D = q_r.shape
    dev = q_r.device
    nkeep = max(1, kb - local_w)
    keep = imp.topk(min(nkeep, S), dim=-1).indices                        # [B,Hq,nkeep]
    out = torch.empty_like(q_r)
    for q0 in range(0, S, q_chunk):
        q1 = min(q0 + q_chunk, S)
        q_blk = q_r[:, :, q0:q1, :]
        q_pos = torch.arange(q0, q1, device=dev)
        Qc = q1 - q0
        idx = keep[:, :, None, :].expand(B, Hq, Qc, keep.shape[-1])
        if local_w > 0:
            ar = torch.arange(local_w, device=dev)
            rec = (q_pos[:, None] - ar[None, :]).clamp(min=0)             # [Qc,local_w]
            rec = rec[None, None].expand(B, Hq, Qc, local_w)
            idx = torch.cat([idx, rec], dim=-1)
        out[:, :, q0:q1, :] = attend_topk(q_blk, kfull, vfull, idx, q_pos, scaling)
    return out


def dense_attention(q_r, k_r, v, n_kv_groups, scaling):
    """Exact full causal attention via SDPA (flash/mem-efficient, O(S) memory).
    Used for the long-context dense reference; routing frac=1.0 through the top-k
    gather would materialize [Qc,S,D] and OOM."""
    kf = repeat_kv(k_r, n_kv_groups)
    vf = repeat_kv(v, n_kv_groups)
    return F.scaled_dot_product_attention(q_r, kf, vf, is_causal=True, scale=scaling)


def sparse_attention_chunked(q_r, k_r, v, sel_override, selector, frac, n_kv_groups,
                             scaling, page=16, q_chunk=256, k_chunk=2048,
                             generator=None, local_w=0):
    if selector == "dense":
        return dense_attention(q_r, k_r, v, n_kv_groups, scaling)
    """Full chunked sparse attention over a prefill window (all positions are queries).

    q_r:  [B,H_q,S,D] post-RoPE queries
    k_r:  [B,H_k,S,D] post-RoPE exact keys
    v:    [B,H_k,S,D] exact values
    kq_r: [B,H_k,S,D] post-RoPE approx keys (for 'approx'); may be None otherwise
    Returns out [B,H_q,S,D]. Peak extra memory O(q_chunk * (k_chunk + kb)).
    """
    B, Hq, S, D = q_r.shape
    kb = max(1, int(round(frac * S)))
    kfull = repeat_kv(k_r, n_kv_groups)
    vfull = repeat_kv(v, n_kv_groups)
    if selector in ("h2o", "snapkv"):
        # global heavy-hitter / observation-window keep-set + per-query recent window
        imp = _attention_importance(q_r, kfull, scaling, selector, q_chunk=q_chunk)
        return _global_keepset_attend(q_r, kfull, vfull, imp, kb, scaling, q_chunk, local_w)
    if selector in SOFT_SELECTORS and sel_override is not None:
        sel_keys = repeat_kv(sel_override, n_kv_groups)
    else:
        sel_keys = kfull  # exact / quest / recent / random use exact-key space (or ignore)
    out = torch.empty_like(q_r)
    for q0 in range(0, S, q_chunk):
        q1 = min(q0 + q_chunk, S)
        q_blk = q_r[:, :, q0:q1, :]
        q_pos = torch.arange(q0, q1, device=q_r.device)
        idx = select_topk(selector, q_blk, sel_keys, q_pos, S, kb, k_chunk,
                          scaling, page=page, generator=generator, local_w=local_w)
        out[:, :, q0:q1, :] = attend_topk(q_blk, kfull, vfull, idx, q_pos, scaling)
    return out
