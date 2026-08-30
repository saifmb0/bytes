import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("=== TESTING SDPA ===")
model_sdpa = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
layer_sdpa = model_sdpa.model.layers[0].self_attn
orig_forward_sdpa = layer_sdpa.forward

def debug_forward_sdpa(hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
    print("SDPA attention_mask type:", type(attention_mask))
    if attention_mask is not None:
        print("SDPA attention_mask shape:", attention_mask.shape)
    return orig_forward_sdpa(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)

layer_sdpa.forward = debug_forward_sdpa
inputs = tokenizer("Hello, how are you today?", return_tensors="pt").to(model_sdpa.device)
with torch.no_grad():
    _ = model_sdpa(**inputs)

print("=== TESTING EAGER ===")
model_eager = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
)
layer_eager = model_eager.model.layers[0].self_attn
orig_forward_eager = layer_eager.forward

def debug_forward_eager(hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
    print("Eager attention_mask type:", type(attention_mask))
    if attention_mask is not None:
        print("Eager attention_mask shape:", attention_mask.shape)
        print("Eager attention_mask first row of last 2 dims:")
        print(attention_mask[0, 0, :, :])
    return orig_forward_eager(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)

layer_eager.forward = debug_forward_eager
inputs = tokenizer("Hello, how are you today?", return_tensors="pt").to(model_eager.device)
with torch.no_grad():
    _ = model_eager(**inputs)
