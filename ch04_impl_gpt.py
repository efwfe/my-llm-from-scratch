# %% 
import torch



# %%
GPT_CONFIG_124M = {
    "vocab_size": 50257, # vocabulary size
    "context_length": 1024, # context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}
# %%
import torch.nn as nn
# %%
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(*[
          DummyTransformerBlock(cfg) 
          for _ in range(cfg['n_layers'])  
        ])
        
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg['emb_dim'],
            cfg['vocab_size'],
            bias = False
        )
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    
    def forward(self, x):
        return x
    
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    
    
    def forward(self, x):
        return x
# %%
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch = torch.stack([
        torch.tensor(tokenizer.encode(txt1)),
        torch.tensor(tokenizer.encode(txt2))
    ], dim=0)
batch
# %%
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
logits
# %%
logits.shape
# %%
torch.manual_seed(123)

batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)

out
# %%
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
mean, var
# %%

# a layer normalization 


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# %%
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print(f"Mean: {mean}")
print(f"Variance: {var}")
# %%
class GELU(nn.Module):

    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 *torch.pow(x, 3))
        ))
# %%
import matplotlib.pyplot as plt
gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))

for i, (y, lable) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{lable} activate function")
    plt.xlabel("x")
    plt.ylabel(f"{lable}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()
    
# %%
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
            
        )
    def forward(self, x):
        return self.layers(x)
# %%
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
out.shape
# %%
# shortcut connections involve adding the inputs to its output,
# They play a crucial role in preserving the flow of gradients during the backward pass in training.

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])
        
    def forward(self, x):
        for layer in self.layers:
            layer_out = layer(x)
            if self.use_shortcut and x.shape == layer_out.shape:
                x = layer_out + x
            else:
                x = layer_out
        return x
# %%
layers_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layers_sizes, use_shortcut=False)


# %%
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    
    loss = nn.MSELoss()
    loss = loss(output, target)
    
    loss.backward()
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
            
# %%
print_gradients(model_with_shortcut, sample_input)
# %%
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layers_sizes,
    use_shortcut=True
)

print_gradients(model_with_shortcut, sample_input)
# %%
# shutcut connections are important for overcommint the limitations posed by the 
# vanishing gradient problem in deep neural networks.

# %%
from ch02_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in  = cfg['emb_dim'],
            d_out= cfg['emb_dim'],
            context_length = cfg['context_length'],
            num_heads = cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])
        
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x) # Pre-LayerNorm
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
# %%
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print(f"Input Shape: {x.shape}")
print(f"Output Shape: {output.shape}")
# %%

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        
        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(cfg) for _ in range(cfg['n_layers'])
            ]
        )
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias=False
        )
        
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
# %%
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print(f"Input batch: {batch.shape}")
print(f"Output shape: {out.shape}")
print(out)
# %%
total_params = sum(p.numel() for p in model.parameters())
print(total_params)
# %%
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print(f"Output layer shape: {model.out_head.weight.shape}")
# %%
total_params - sum(p.numel() for p in model.out_head.parameters())
# %%
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)

print(f"Total size of the model is :{total_size_mb} MB")
# %%
# %% 1.Extracts the last vector, which corresponds to the next token

# %% 2. Converts logits into probility distribution using softmax

# %% 3. Indentifies the index position of the largest value, 

# %% 4. Appends token to the previous inputs for the next round

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
# %%
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print(f"Encoded : {encoded}")
encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
print(f"Encoded_tensor.shape, {encoded_tensor.shape}")

# %%
model.eval() # disables random components like dropout

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M['context_length']
)

print(out)
print(len(out[0]))
# %%
decoded_text =tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
# %%
