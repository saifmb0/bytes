import torch
from src.quantization import rotate_vectors, ProductQuantizer

def generate_activations_with_outliers(
    num_samples: int,
    d: int,
    outlier_channels: list = [12, 45],
    outlier_scale: float = 80.0,
    normal_std: float = 1.0,
    device="cuda",
    dtype=torch.float32
) -> torch.Tensor:
    """
    Generates synthetic activations with extreme channel-wise outliers,
    resembling the distributions seen in Qwen/SwiGLU models.
    """
    # 1. Base normal distribution
    X = torch.randn(num_samples, d, device=device, dtype=dtype) * normal_std
    
    # 2. Inject massive outliers into specific channels
    for ch in outlier_channels:
        if ch < d:
            # Outliers can have specific directional signs (e.g. mostly positive)
            # which creates strong anisotropy and sign-pattern issues.
            outliers = (torch.randn(num_samples, device=device, dtype=dtype) + 2.0) * outlier_scale
            X[:, ch] = outliers
            
    return X

def compute_sign_error(
    queries: torch.Tensor,
    keys: torch.Tensor,
    pq: ProductQuantizer,
    sign_pattern: torch.Tensor
) -> float:
    """
    Computes the Quantization Error Covariance Metric (sign-selection error):
    sign-selection error = | E[ q_rot^T * (k_rot - k_rot_hat) ] | / E[ |q^T * k| ]
    """
    # Rotate queries and keys with the given sign pattern
    q_rot = rotate_vectors(queries, sign_pattern)
    k_rot = rotate_vectors(keys, sign_pattern)
    
    # Quantize and dequantize keys
    indices = pq.quantize(k_rot)
    k_rot_hat = pq.dequantize(indices)
    
    # Compute true inner product in original basis
    true_ip = torch.sum(queries * keys, dim=-1) # [N]
    
    # Compute reconstructed inner product
    recon_ip = torch.sum(q_rot * k_rot_hat, dim=-1) # [N]
    
    # Quantization error inner product: q_rot^T * (k_rot - k_rot_hat)
    # which is exactly: true_ip - recon_ip
    error_ip = true_ip - recon_ip
    
    sign_error = torch.abs(torch.mean(error_ip)) / torch.mean(torch.abs(true_ip))
    return sign_error.item()

def search_best_sign_pattern(
    queries: torch.Tensor,
    keys: torch.Tensor,
    pq: ProductQuantizer,
    num_candidates: int = 128
) -> torch.Tensor:
    """
    Performs a gradient-free search over random sign patterns to find the one
    that minimizes sign-selection error.
    """
    d = queries.shape[-1]
    device = queries.device
    
    best_sign_error = float('inf')
    best_pattern = None
    
    print(f"Searching for stable sign pattern (evaluating {num_candidates} candidates)...")
    
    # Always include a default all-ones sign pattern as candidate 0
    default_pattern = torch.ones(d, device=device)
    default_sign_error = compute_sign_error(queries, keys, pq, default_pattern)
    best_sign_error = default_sign_error
    best_pattern = default_pattern
    
    for i in range(num_candidates - 1):
        # Generate a random sign pattern in {-1, 1}
        pattern = torch.randint(0, 2, (d,), device=device).float() * 2.0 - 1.0
        sign_error = compute_sign_error(queries, keys, pq, pattern)
        
        if sign_error < best_sign_error:
            best_sign_error = sign_error
            best_pattern = pattern
            
    print(f"Default (all-ones) sign pattern sign-selection error: {default_sign_error:.6f}")
    print(f"Best found sign pattern sign-selection error:          {best_sign_error:.6f} (reduction: {default_sign_error / (best_sign_error + 1e-9):.2f}x)")
    return best_pattern
