# %%
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# %%
# Step 1. compute the unnormalized attention scores 
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)
# %%
# understand dot product
res = 0.
for idx, element in enumerate(inputs[0]):
    res += element * query[idx]
print(res, torch.dot(inputs[0], query))
# %%
atten_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print(f"Attention weights: {atten_weights_2_tmp}")
print(f"Sum: {atten_weights_2_tmp.sum()}")
# %%

# sfortmax function also meets the objective and normalizes the attention weights such that they sum to 1:

def softmax_native(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2 = softmax_native(attn_scores_2)
print(f"Attention weights: {attn_weights_2}")
print(f"Sum: {attn_weights_2.sum()}")

# %%
# Step2. normalize the unormalized attention scores

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print(attn_weights_2)
print(attn_weights_2.sum())
# %%
# Step3. compute the context vector by multiplying the embedded input tokens

query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)

# %%
# Step 4. Computing attention weights for all input tokens

# Attention weights; tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]) 
# Context weights: tensor([0.4419, 0.6515, 0.5683])

attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)
# %%
attn_scores = inputs @ inputs.T
print(attn_scores)
# %%
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
# the values in each row sum up to 1
# %%
print(attn_weights.sum(dim=-1))

# %%
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
# %%
"""
设计 Q, K, V 矩阵的目的是为了让模型具备学习能力，能够灵活调整输入表示，使其在各种任务和场景下都能表现良好。
直接点积虽然可以计算出某些简单的注意力分数，但没有学习和优化的能力，这限制了其表达能力和泛化性能。
"""
x_2 = inputs[1]
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2
print(f"Inputs shape: {inputs.shape}")
# %%
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad=False)

# %%
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2)
# %%
keys = inputs@ W_key
values = inputs @ W_value
print(f"Keys.shape: {keys.shape}")
print(f"Values.shape: {values.shape}")
# %%
keys_2 = keys[1]
atten_score_22 = query_2.dot(key_2)
print(atten_score_22)
# %%
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)
# %%
d_k = keys.shape[-1]
# 防止点积过大，导致注意力分布极端，稳定
# called scaled-dot product attention
attn_weights_2 = torch.softmax(attn_scores_2/d_k ** 0.5, dim=-1)
print(attn_weights_2)
# %%
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
# %%
from torch import nn

class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        
    
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_score = queries @ keys.T
        attn_score = torch.softmax(attn_score / keys.shape[-1]**0.5, dim=-1)        
        context_vec = attn_score @ values
        return context_vec
        
# %%
torch.manual_seed(123)
sa_v1 = SelfAttentionV1(d_in, d_out)
print(sa_v1(inputs))
# %%

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        attn_score = queries @ keys.T
        attn_score = torch.softmax(attn_score / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_score @ values
        return context_vec        
# %%
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
sa_v2.W_key.weight = torch.nn.Parameter(sa_v1.W_key.T)
sa_v2.W_query.weight = torch.nn.Parameter(sa_v1.W_query.T)
sa_v2.W_value.weight = torch.nn.Parameter(sa_v1.W_value.T)
print(sa_v2(inputs))
# %%
# causl attention mask
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_scores = torch.softmax(attn_scores/keys.shape[1] ** 0.5, dim=-1)
print(attn_scores)
# %%
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
# %%
masked_simple = attn_scores * mask_simple
print(masked_simple)
# %%
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
# %%
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
# %%
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print(attn_weights)
# %%
# Dropout
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))
# %%
torch.manual_seed(123)
print(dropout(attn_weights))
# %%

# 
batch = torch.stack((inputs, inputs,), dim=0)
print(batch.shape)
# %%
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out  # 输出维度
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 查询线性层
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # 键线性层
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 值线性层
        # self.context_length = context_length  # 上下文长度
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)  # 上三角掩码
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 获取输入的形状
        queries = self.W_query(x)  # 计算查询向量
        keys = self.W_key(x)  # 计算键向量
        values = self.W_value(x)  # 计算值向量
        
        attn_scores = queries @ keys.transpose(1, 2)  # 计算注意力分数
        # 将批次维度保持在第一位
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf  # 使用掩码填充注意力分数
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1  # 计算注意力权重
        )
        
        attn_weights = self.dropout(attn_weights)  # 应用Dropout
        context_vec = attn_weights @ values  # 计算上下文向量
        return context_vec  # 返回上下文向量
# %%
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.)
context_vecxs = ca(batch)
print(context_vecxs.shape)
# %%

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
        
    def forward(self, x):
        return torch.cat([
            head(x) for head in self.heads
        ], dim=-1)
# %%
torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 1

mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0., num_heads=2)

context_vec = mha(batch)
print(context_vec.shape)
# %%

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divible by num heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
# %%
torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out =2 

mha = MultiHeadAttention(d_in, d_out, context_length, 0., num_heads=2)
context_vec = mha(batch)

print(context_vec, context_vec.shape)
# %%
gpt_mha = MultiHeadAttention(d_in=768, d_out=768, context_length=1024, dropout=0., num_heads=12)