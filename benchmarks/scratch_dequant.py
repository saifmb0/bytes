import torch
from src.quantization import ProductQuantizer
from src.attention_kernels import decode_attn_dequant

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
k_hat = pq.dequantize(k_idx) # Keep in float32

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

print("Max diff:", (ref_out - triton_out).abs().max().item())
print("Mean diff:", (ref_out - triton_out).abs().mean().item())
print("Ref sample (first 5):", ref_out[0, 0, :5].tolist())
print("Triton sample (first 5):", triton_out[0, 0, :5].tolist())
