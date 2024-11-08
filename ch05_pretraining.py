# %% 
import torch
from ch04_impl_gpt import GPTModel, generate_text_simple
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,    #1 short context 1024->512
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12, 
    "drop_rate": 0.1,       #2 set dropdown 
    "qkv_bias": False
}

# %%
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()
# %%
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(model=model, idx=text_to_token_ids(start_context, tokenizer),
                                 max_new_tokens=10, context_size=GPT_CONFIG_124M['context_length'])
print(token_ids_to_text(token_ids, tokenizer))
# %%
inputs = torch.tensor([
    [16833, 3626, 6100], 
    [40, 1107, 588]
])

targets = torch.tensor([
    [3626, 6100, 345],
    [1107, 588, 11311]
])

with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1)
print(probas.shape)
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print(token_ids)
# %%
print(f"Tragets batch 1 : {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1 :{token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
# %%
# the model training aims to increase the softmax probability in the index positions
# corresponding to the correct target token IDs.

# the softmax probability is also used in the evaluation metric 


text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print(f"Text 1: {target_probas_1}")
# %%
text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2],targets[text_idx]]
print(f"Text 2 : {target_probas_2}")
# %%
# Calculate the loss involves several steps;

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
print(log_probas.shape)
# %%
avg_log_probas = torch.mean(log_probas)
avg_log_probas
# %%
neg_avg_log_probas = avg_log_probas * -1
neg_avg_log_probas
# %%

# %% cross entrypy loss
print(f"Logits shape: {logits.shape}")
print(f"Traget shape: {targets.shape}")

# %%
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()

print(f"Flattened logits: {logits_flat.shape}")
print(f"Flattened target: {targets_flat.shape}")
# %%
# torch cross_entropy take care of softmax select probability scores to the target ids

loss = torch.nn.functional.cross_entropy(
    logits_flat,
    targets_flat
)
print(loss)
# %%