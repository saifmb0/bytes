import torch
from src.attention_kernels import decode_attn_bf16

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

print("Reference shape:", ref_out.shape)
print("Triton shape:", triton_out.shape)
print("Max diff:", (ref_out - triton_out).abs().max().item())
print("Mean diff:", (ref_out - triton_out).abs().mean().item())
print("Reference sample (first 5 elements of first head):", ref_out[0, 0, :5].tolist())
print("Triton sample (first 5 elements of first head):", triton_out[0, 0, :5].tolist())
