import math
import torch
import torch.nn as nn
import torch.optim as optim
import triton
import triton.language as tl

@triton.jit
def _quantize_keys_hybrid_kernel(
    X_ptr, Centroids_ptr, Centroid_Norms_ptr, Indices_ptr,
    stride_x_n, stride_x_d,
    stride_c_m, stride_c_k, stride_c_d,
    stride_i_n, stride_i_m,
    total_tokens,
    M_k: tl.constexpr, K_k: tl.constexpr, d_sub: tl.constexpr,
    K_k_padded: tl.constexpr,
    BLOCK_S: tl.constexpr
):
    pid_s = tl.program_id(0) # Token block index
    
    start_s = pid_s * BLOCK_S
    offsets_s = start_s + tl.arange(0, BLOCK_S)
    mask_s = offsets_s < total_tokens
    
    # We pad the inner dimension to 16 for Tensor Cores
    d_sub_padded = 16
    
    for j in range(M_k):
        # 1. Load X and pad inner dimension to 16
        x_cols = tl.arange(0, 16)
        x_offsets = offsets_s[:, None] * stride_x_n + (j * d_sub + x_cols[None, :]) * stride_x_d
        x_mask = mask_s[:, None] & (x_cols[None, :] < d_sub)
        x = tl.load(X_ptr + x_offsets, mask=x_mask, other=0.0) # [BLOCK_S, 16]
        
        # 2. Load Centroids^T of shape [16, K_k_padded]
        c_cols = tl.arange(0, K_k_padded)
        c_rows = tl.arange(0, 16)
        c_offsets = (j * stride_c_m + c_cols[None, :] * stride_c_k + c_rows[:, None] * stride_c_d)
        c_mask = (c_rows[:, None] < d_sub) & (c_cols[None, :] < K_k)
        C_j_T = tl.load(Centroids_ptr + c_offsets, mask=c_mask, other=0.0) # [16, K_k_padded]
        
        # 3. Compute dot product: tl.dot(X, C_j_T) -> [BLOCK_S, K_k_padded] on Tensor Cores
        dot_prod = tl.dot(x.to(tl.float16), C_j_T.to(tl.float16))
        
        # 4. Load precomputed centroid norms: [K_k_padded]
        norms_offsets = j * K_k_padded + tl.arange(0, K_k_padded)
        norms_mask = tl.arange(0, K_k_padded) < K_k
        norms = tl.load(Centroid_Norms_ptr + norms_offsets, mask=norms_mask, other=0.0) # [K_k_padded]
        
        # 5. Compute S = dot_prod - norms
        scores = dot_prod - norms[None, :]
        if K_k_padded > K_k:
            # Mask out padded centroids with -inf to prevent them from being chosen
            pad_mask = tl.arange(0, K_k_padded) < K_k
            scores = tl.where(pad_mask[None, :], scores, -float('inf'))
            
        # 6. Find argmax using ALU:
        idx = tl.argmax(scores, axis=1) # [BLOCK_S]
        
        # 7. Store index to Indices_ptr: [N, M_k]
        i_offsets = offsets_s * stride_i_n + j * stride_i_m
        i_ptr = Indices_ptr + i_offsets
        tl.store(i_ptr, idx.to(tl.uint8), mask=mask_s)


def get_hadamard_matrix(n: int, device="cpu") -> torch.Tensor:
    """
    Recursively construct a normalized Walsh-Hadamard matrix of size n.
    n must be a power of 2.
    """
    assert (n & (n - 1)) == 0 and n > 0, "n must be a power of 2"
    if n == 1:
        return torch.ones((1, 1), device=device)
    h_half = get_hadamard_matrix(n // 2, device=device)
    h = torch.cat([
        torch.cat([h_half, h_half], dim=1),
        torch.cat([h_half, -h_half], dim=1)
    ], dim=0)
    return h / (2.0 ** 0.5)

def rotate_vectors(X: torch.Tensor, sign_pattern = None) -> torch.Tensor:
    """
    Applies Hadamard rotation to vectors X.
    X: [..., d]
    sign_pattern: [d] tensor containing values in {-1, 1}, or string "no_rot" to bypass.
    """
    if isinstance(sign_pattern, str) and sign_pattern == "no_rot":
        return X
    d = X.shape[-1]
    device = X.device
    dtype = X.dtype
    H = get_hadamard_matrix(d, device=device).to(dtype=dtype)
    if sign_pattern is not None:
        # R = H * diag(s)
        # X * R = X * H * diag(s) = (X * H) * s
        rotated = torch.matmul(X, H) * sign_pattern.to(dtype=dtype)
    else:
        rotated = torch.matmul(X, H)
    return rotated


def kmeans_pytorch(X: torch.Tensor, k: int, num_iters: int = 20) -> torch.Tensor:
    """
    Simple k-means implementation in PyTorch for GPU acceleration.
    X: [N, d_sub]
    k: number of centroids
    """
    N, d_sub = X.shape
    if N <= k:
        # If fewer points than clusters, pad with zeros or repeat
        padding = torch.zeros((k - N, d_sub), device=X.device, dtype=X.dtype)
        return torch.cat([X.clone(), padding], dim=0)
    
    # Initialize centroids randomly from points
    perm = torch.randperm(N, device=X.device)
    centroids = X[perm[:k]].clone()
    
    for _ in range(num_iters):
        # Compute pairwise distances using cdist: [N, k]
        dists = torch.cdist(X, centroids)
        labels = torch.argmin(dists, dim=1)
        
        # Update centroids
        for i in range(k):
            mask = (labels == i)
            if mask.sum() > 0:
                centroids[i] = X[mask].mean(dim=0)
            else:
                # Handle empty cluster: reinitialize to a random point
                centroids[i] = X[torch.randint(0, N, (1,), device=X.device)]
                
    return centroids

class ProductQuantizer:
    def __init__(self, d: int, d_sub: int, bits: int = 3, device="cuda"):
        self.d = d
        self.d_sub = d_sub
        self.bits = bits
        self.k = 2 ** bits
        self.M = d // d_sub
        assert d % d_sub == 0, f"d ({d}) must be divisible by d_sub ({d_sub})"
        self.device = device
        
        # Centroids: [M, k, d_sub]
        self.centroids = torch.zeros((self.M, self.k, self.d_sub), device=device)
        
    def fit(self, X: torch.Tensor, num_iters=20):
        """
        X: [N, d] - Calibration keys in rotated space
        """
        X = X.to(self.device)
        N = X.shape[0]
        # Split into M subvectors of shape [N, d_sub]
        X_split = X.view(N, self.M, self.d_sub)
        
        for j in range(self.M):
            X_j = X_split[:, j, :] # [N, d_sub]
            self.centroids[j] = kmeans_pytorch(X_j, self.k, num_iters)
            
    def quantize(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [..., d] - Vectors to quantize
        Returns indices of shape [..., M] as long/int tensor
        """
        orig_shape = list(X.shape[:-1])
        N = X.numel() // self.d
        
        if X.is_cuda:
            # Reshape centroids and X for Triton
            K_padded = max(16, self.k)
            centroid_norms = 0.5 * torch.sum(self.centroids.float() ** 2, dim=-1) # [M, K]
            if K_padded > self.k:
                # Pad norms
                pad_norms = torch.zeros(self.M, K_padded, dtype=centroid_norms.dtype, device=X.device)
                pad_norms[:, :self.k] = centroid_norms
                centroid_norms = pad_norms
                # Pad centroids
                centroids_padded = torch.zeros(self.M, K_padded, self.d_sub, dtype=self.centroids.dtype, device=X.device)
                centroids_padded[:, :self.k, :] = self.centroids
            else:
                centroids_padded = self.centroids
            
            centroids_padded_val = centroids_padded.to(X.dtype)
            centroid_norms_val = centroid_norms.to(X.dtype)
            X_contiguous = X.reshape(N, self.d).contiguous()
            indices = torch.empty((N, self.M), dtype=torch.uint8, device=X.device)
            BLOCK_S = 32
            grid = (triton.cdiv(N, BLOCK_S),)
            
            _quantize_keys_hybrid_kernel[grid](
                X_contiguous, centroids_padded_val, centroid_norms_val, indices,
                X_contiguous.stride(0), X_contiguous.stride(1),
                centroids_padded_val.stride(0), centroids_padded_val.stride(1), centroids_padded_val.stride(2),
                indices.stride(0), indices.stride(1),
                N,
                M_k=self.M, K_k=self.k, d_sub=self.d_sub,
                K_k_padded=K_padded,
                BLOCK_S=BLOCK_S,
                num_warps=8
            )
            return indices.view(*(orig_shape + [self.M])).long()
        
        # Fall back to PyTorch CPU code
        X_flat = X.reshape(N, self.M, self.d_sub)
        indices = torch.zeros((N, self.M), dtype=torch.long, device=X.device)
        for j in range(self.M):
            # Compute distance to each centroid: [N, k]
            dists = torch.cdist(X_flat[:, j, :], self.centroids[j])
            indices[:, j] = torch.argmin(dists, dim=1)
            
        return indices.view(*(orig_shape + [self.M]))
        
    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [..., M]
        Returns reconstructed vectors of shape [..., d]
        """
        orig_shape = list(indices.shape[:-1])
        N = indices.numel() // self.M
        ind_flat = indices.view(N, self.M)
        
        reconstructed = torch.zeros((N, self.M, self.d_sub), device=indices.device, dtype=self.centroids.dtype)
        for j in range(self.M):
            reconstructed[:, j, :] = self.centroids[j][ind_flat[:, j].long()]
            
        return reconstructed.view(*(orig_shape + [self.d]))

class CalibratedPQ(nn.Module):
    def __init__(self, pq: ProductQuantizer):
        super().__init__()
        self.d = pq.d
        self.d_sub = pq.d_sub
        self.M = pq.M
        self.k = pq.k
        self.device = pq.device
        
        # Define centroids as PyTorch parameter so we can optimize them
        self.centroids = nn.Parameter(pq.centroids.clone())
        
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [..., M]
        Returns reconstructed vectors: [..., d]
        """
        orig_shape = list(indices.shape[:-1])
        N = indices.numel() // self.M
        ind_flat = indices.view(N, self.M)
        
        reconstructed_list = []
        for j in range(self.M):
            reconstructed_list.append(self.centroids[j][ind_flat[:, j].long()])
            
        reconstructed = torch.stack(reconstructed_list, dim=1) # [N, M, d_sub]
        return reconstructed.view(*(orig_shape + [self.d]))

def detect_outlier_channels(X: torch.Tensor, num_outliers: int) -> torch.Tensor:
    """
    X: [H_k, N, d] or [..., d] - key activations across sequence/batch/heads
    Returns: indices of top num_outliers channels with the highest variance (per-head or global)
    """
    if num_outliers <= 0:
        return torch.empty(0, dtype=torch.long, device=X.device)
    if X.dim() == 3:
        # [H_k, N, d] - head-aware detection
        variances = torch.var(X, dim=1) # [H_k, d]
        outlier_indices = torch.zeros((X.shape[0], num_outliers), dtype=torch.long, device=X.device)
        for h in range(X.shape[0]):
            outlier_indices[h] = torch.topk(variances[h], k=num_outliers).indices
        return outlier_indices
    else:
        X_flat = X.reshape(-1, X.shape[-1])
        variances = torch.var(X_flat, dim=0)
        topk = torch.topk(variances, k=min(num_outliers, X.shape[-1]))
        return topk.indices

def calibrate_codebook(
    queries: torch.Tensor,
    keys: torch.Tensor,
    pq: ProductQuantizer,
    sign_pattern: torch.Tensor = None,
    outlier_indices: torch.Tensor = None,
    lr: float = 5e-3,
    steps: int = 500,
    batch_size: int = 512
) -> CalibratedPQ:
    """
    Optimizes centroids to preserve the inner product:
    L = E_{q, k} [ (q_dense^T * k_dense - q_rot^T * k_rot_hat)^2 ]
    
    queries: [N_cal, d] - in original basis
    keys: [N_cal, d] - in original basis
    """
    device = pq.device
    queries = queries.to(device)
    keys = keys.to(device)
    
    # 1. Zero out outliers for dense calibration if indices are provided
    q_dense = queries.clone()
    k_dense = keys.clone()
    if outlier_indices is not None and len(outlier_indices) > 0:
        q_dense[..., outlier_indices] = 0.0
        k_dense[..., outlier_indices] = 0.0
        
    # 2. Rotate queries and keys in dense space
    q_rot = rotate_vectors(q_dense, sign_pattern)
    k_rot = rotate_vectors(k_dense, sign_pattern)
    
    # 3. Quantize keys in rotated space to get fixed indices
    indices = pq.quantize(k_rot)
    
    # 4. Create the optimization module
    cal_pq = CalibratedPQ(pq).to(device)
    optimizer = optim.Adam(cal_pq.parameters(), lr=lr)
    
    N = queries.shape[0]
    
    for step in range(steps):
        # Sample mini-batches
        perm = torch.randperm(N, device=device)[:batch_size]
        q_batch = q_dense[perm]
        k_batch = k_dense[perm]
        q_rot_batch = q_rot[perm]
        indices_batch = indices[perm]
        
        optimizer.zero_grad()
        
        # Reconstruct keys using current centroids
        k_rot_hat = cal_pq(indices_batch) # [batch_size, d]
        
        # Target dense inner product: q_dense^T * k_dense
        target_ip = torch.sum(q_batch * k_batch, dim=-1) # [batch_size]
        
        # Reconstructed inner product in rotated space
        recon_ip = torch.sum(q_rot_batch * k_rot_hat, dim=-1) # [batch_size]
        
        # Loss: mean squared error of the inner products
        loss = torch.mean((target_ip - recon_ip) ** 2)
        
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 100 == 0:
            print(f"Calibration Step {step+1}/{steps} | Loss: {loss.item():.6f}")
            
    # Copy calibrated centroids back to PQ object
    pq.centroids.copy_(cal_pq.centroids.data)
    return cal_pq


def generate_random_orthogonal(d: int, device="cuda") -> torch.Tensor:
    """
    Generate a Haar-distributed random orthogonal matrix of size [d, d].
    Used by TurboQuant-style data-oblivious rotation (no calibration data required).
    The randomness is seeded by the current torch state; fix torch.manual_seed()
    before calling if reproducibility is needed.
    """
    A = torch.randn(d, d, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q


class PerChannelScalarQuantizer:
    """
    Per-channel uniform scalar quantizer (TurboQuant-style).

    For each of the D coordinates independently:
      - Calibrates a [min, max] range from data
      - Maps values to N-bit integers via uniform quantization
      - Dequantizes back via linear rescaling

    This is the decode-side representation used by TurboQuant:
    keys are stored as D integer indices per token, each quantized
    per-coordinate. At decode time the model must dequantize all D
    coordinates and perform the full Q·K matmul — unlike PQ-LUT which
    avoids this via precomputed LUT lookup.
    """
    def __init__(self, d: int, bits: int, device="cuda"):
        self.d = d
        self.bits = bits
        self.levels = 2 ** bits
        self.device = device
        self.q_min = torch.zeros(d, device=device, dtype=torch.float32)
        self.q_max = torch.zeros(d, device=device, dtype=torch.float32)
        self.scale = torch.ones(d, device=device, dtype=torch.float32)

    def fit(self, X: torch.Tensor):
        """
        Calibrate per-channel min/max from data.
        X: [N, d] — calibration key activations in rotated space.
        """
        X = X.float().to(self.device)
        self.q_min = X.min(dim=0).values
        self.q_max = X.max(dim=0).values
        self.scale = (self.q_max - self.q_min).clamp(min=1e-8) / (self.levels - 1)

    def quantize(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [..., d] — input vectors to quantize.
        Returns: long tensor of same shape with integer indices in [0, levels-1].
        """
        X = X.float().to(self.device)
        X_clipped = X.clamp(self.q_min, self.q_max)
        return torch.round((X_clipped - self.q_min) / self.scale).long()

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [..., d] — integer indices.
        Returns: float tensor of reconstructed values.
        """
        return indices.float().to(self.device) * self.scale + self.q_min


def generate_jl_matrix(m: int, d: int, device="cuda") -> torch.Tensor:
    """
    Random Johnson-Lindenstrauss projection matrix P ∈ ℝ^{m×d},
    entries drawn i.i.d. from N(0, 1/m).

    Used for TurboQuant's 1-bit QJL residual correction:
      b_j = sign(P_j · r)   for residual r = k_rot - k_hat_rot

    Unbiased inner-product estimator (derivation via bivariate-normal):
      E[(P_j · q) · sign(P_j · r)] = sqrt(2/π) · (q·r) / (sqrt(m) · ‖r‖)
    Summing over m projections and rescaling:
      q · r ≈ (P@q) · b · ‖r‖ · sqrt(π / (2m))

    Adding m bits per key increases bpw by m/D.
    At m=D this costs +1 bpw; at m=D//2 it costs +0.5 bpw.
    """
    return torch.randn(m, d, device=device) * (1.0 / math.sqrt(m))


def _asym_uniform_fakequant(x: torch.Tensor, levels: int, dim: int) -> torch.Tensor:
    """
    Asymmetric uniform fake-quantization: quantize then dequantize in one step.
    The [min, max] range is computed over `dim` (so the grouping is along that
    axis); every other axis gets its own independent scale/zero-point.
    """
    xmin = x.amin(dim=dim, keepdim=True)
    xmax = x.amax(dim=dim, keepdim=True)
    scale = (xmax - xmin).clamp(min=1e-8) / (levels - 1)
    q = torch.clamp(torch.round((x - xmin) / scale), 0, levels - 1)
    return q * scale + xmin


class KIVIQuantizer:
    """
    Faithful KIVI key/value cache quantization (Liu et al., ICML 2024).

    - Keys: per-channel asymmetric uniform quant, grouped along the token axis in
      groups of `group_size`. KIVI observes that key outliers are concentrated in
      a few fixed channels, so quantizing per channel (scale computed over the
      tokens within a group) preserves those channels' dynamic range.
    - Values: per-token asymmetric uniform quant (scale per token, taken over the
      channel axis), matching KIVI's per-token value treatment.
    - Residual window: the most recent `residual_length` tokens are kept in full
      precision (FP16), as in KIVI's streaming formulation.

    Tuning-free: all scales are computed at runtime from the activations; no
    calibration pass is required. KIVI operates on the cache *as stored*
    (post-RoPE for keys), so the PPL hook applies `quantize_key` after RoPE.
    """
    def __init__(self, bits: int = 4, group_size: int = 32, residual_length: int = 32):
        self.bits = bits
        self.levels = 2 ** bits
        self.group_size = group_size
        self.residual_length = residual_length

    def quantize_key(self, k: torch.Tensor) -> torch.Tensor:
        # k: [B, H, S, D]. Per-channel scales, grouped along tokens.
        B, H, S, D = k.shape
        R = min(self.residual_length, S)
        S_q = S - R
        if S_q <= 0:
            return k
        k_q = k[:, :, :S_q, :].float()
        k_res = k[:, :, S_q:, :]
        G = self.group_size
        n_full = S_q // G
        parts = []
        if n_full > 0:
            main = k_q[:, :, :n_full * G, :].reshape(B, H, n_full, G, D)
            # per-channel within each token-group: range taken over the G tokens
            main_dq = _asym_uniform_fakequant(main, self.levels, dim=3)
            parts.append(main_dq.reshape(B, H, n_full * G, D))
        if n_full * G < S_q:
            tail = k_q[:, :, n_full * G:, :]               # [B,H,r,D]
            parts.append(_asym_uniform_fakequant(tail, self.levels, dim=2))
        k_dq = torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]
        return torch.cat([k_dq.to(k.dtype), k_res], dim=2)

    def quantize_value(self, v: torch.Tensor) -> torch.Tensor:
        # v: [B, H, S, D]. Per-token (range over channels), residual kept FP16.
        B, H, S, D = v.shape
        R = min(self.residual_length, S)
        S_q = S - R
        if S_q <= 0:
            return v
        v_q = v[:, :, :S_q, :].float()
        v_res = v[:, :, S_q:, :]
        v_dq = _asym_uniform_fakequant(v_q, self.levels, dim=3)   # per token over D
        return torch.cat([v_dq.to(v.dtype), v_res], dim=2)

    def bpw(self, seq_len: int) -> float:
        """Effective key bpw at a given context length (residual tokens cost 16b)."""
        R = min(self.residual_length, seq_len)
        frac_q = (seq_len - R) / seq_len
        return self.bits * frac_q + 16.0 * (R / seq_len)


class KVQuantNUQ:
    """
    Faithful KVQuant key quantization (Hooper et al., NeurIPS 2024).

    - Pre-RoPE per-channel quantization: keys are quantized *before* RoPE, where
      the per-channel outlier structure is stable. The PPL hook applies
      `quantize_key` in the pre-RoPE block.
    - Non-uniform (nuq) signposts: 2^bits levels *per channel*, fit by k-means on
      calibration activations. Full KVQuant weights the k-means objective by
      Fisher information (per-element sensitivity); this harness has no gradient
      signal at calibration time, so we use unweighted k-means — disclosed as a
      faithful-but-simplified element in the paper.
    - Dense-and-sparse: the top `sparse_frac` fraction of key entries by magnitude
      are stored in FP16 (sparse) and excluded from the dense grid; the dense
      remainder maps to its nearest per-channel signpost.
    - Values: per-token asymmetric uniform quant (KVQuant's contribution is on
      keys; values use the standard per-token treatment).

    Requires a calibration pass (`fit`) to derive the per-channel signposts.
    """
    def __init__(self, bits: int = 4, sparse_frac: float = 0.01, device="cuda"):
        self.bits = bits
        self.levels = 2 ** bits
        self.sparse_frac = sparse_frac
        self.device = device
        self.signposts = None    # [H, D, levels]
        self.threshold = None    # global |value| threshold for the sparse split

    def fit(self, k_cal: torch.Tensor):
        # k_cal: [H, N, D] per-head per-channel calibration keys (pre-RoPE).
        # KVQuant fits the non-uniform signposts on the DENSE part only: the top
        # `sparse_frac` |values| (which dominate attention logits) are split off
        # and stored exact, so the limited 2^bits levels concentrate resolution on
        # the bulk instead of being pulled toward the tails by the k-means MSE.
        k_cal = k_cal.to(self.device)
        H, N, D = k_cal.shape
        flat = k_cal.abs().reshape(-1)
        n_sparse = max(1, int(self.sparse_frac * flat.numel()))
        self.threshold = torch.topk(flat, n_sparse, sorted=False).values.min()
        signposts = torch.zeros(H, D, self.levels, device=self.device)
        for h in range(H):
            for d in range(D):
                vals = k_cal[h, :, d]
                dense = vals[vals.abs() < self.threshold]
                if dense.numel() < self.levels:
                    dense = vals
                c = kmeans_pytorch(dense.reshape(-1, 1), self.levels, num_iters=10)
                c = c.flatten().sort().values
                # Anchor the outer signposts to the channel's full range. k-means
                # is MSE-optimal, so it crowds centroids near the mode and under-
                # covers the tails; eval keys past the outermost centroid then
                # clamp, producing large per-channel errors on exactly the high-
                # magnitude channels attention is most sensitive to. Anchoring the
                # endpoints removes that clamping (a few catastrophic layers ->
                # near-lossless) while keeping the interior levels non-uniform.
                c[0] = torch.minimum(c[0], vals.min())
                c[-1] = torch.maximum(c[-1], vals.max())
                signposts[h, d] = c
        self.signposts = signposts

    def quantize_key(self, k: torch.Tensor) -> torch.Tensor:
        # k: [B, H, S, D] pre-RoPE. Returns fake-quantized keys.
        assert self.signposts is not None, "KVQuantNUQ.fit() must run before use"
        B, H, S, D = k.shape
        kf = k.float()
        # dense-and-sparse: |value| >= calibrated threshold is kept exact (FP16);
        # the dense remainder maps to its nearest per-channel signpost.
        sparse_mask = kf.abs() >= self.threshold
        sp = self.signposts.view(1, H, 1, D, self.levels)            # broadcast
        idx = (kf.unsqueeze(-1) - sp).abs().argmin(dim=-1)           # [B,H,S,D]
        dq = sp.expand(B, H, S, D, self.levels).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        dq = torch.where(sparse_mask, kf, dq)                        # reinject outliers
        return dq.to(k.dtype)

    def quantize_value(self, v: torch.Tensor) -> torch.Tensor:
        return _asym_uniform_fakequant(v.float(), self.levels, dim=3).to(v.dtype)

    def bpw(self, d_head: int) -> float:
        """Effective key bpw: dense bits + sparse overhead (FP16 value + index)."""
        return self.bits + self.sparse_frac * 32.0
