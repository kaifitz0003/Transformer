"""
The code below shows a manual Transformer Encoder and Decoder built from scratch.

The architecture was copied from the Attention Is All You Need paper. The code makes an encoder and decoder 
layer by using nn.MultiheadAttention, nn.LayerNorm, and linear layers.
It copies the weights from Pytorches built in transformer encoder and decoder layers. The model also has self attention, cross attention, feed foward layer, and normalization layers. 
At the end, I compared the manual results to PyTorches refernence decoder to ensure they were the same.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)
d_in=4 # Input feature groups (temperature, humidity, windspeed, and rain percentage)
L = 5 # Readings in a week
n_heads=3 # Number of attention heads
d_head=2 # Features per head
d_model= n_heads*d_head # 6
d_feedforward = 8

X = torch.rand(L,d_in)
X_proj = X@torch.rand(d_in,d_model) # (L, d_model)

EncoderLayer = nn.TransformerEncoderLayer(d_model=d_model, nhead = n_heads, dim_feedforward=d_feedforward, dropout=0, batch_first=True, bias=True)
out_reference_encoder = EncoderLayer(X_proj)
### Multihead Attention
my_self_attn = nn.MultiheadAttention(d_model,n_heads, batch_first=True)

my_self_attn.in_proj_weight.data = EncoderLayer.self_attn.in_proj_weight.data
my_self_attn.out_proj.weight.data = EncoderLayer.self_attn.out_proj.weight.data

my_self_attn.in_proj_bias.data = EncoderLayer.self_attn.in_proj_bias.data
my_self_attn.out_proj.bias.data = EncoderLayer.self_attn.out_proj.bias.data

out_attention,_ = my_self_attn(X_proj, X_proj, X_proj) # Since Q,K,V are the same as X_proj, we are preforming self-attention.

### Add & Norm
my_norm1 = nn.LayerNorm(d_model)
out_norm1 = my_norm1(X_proj+out_attention)

### Feed Forward Neural Network
my_linear1 = nn.Linear(in_features=d_model,out_features=d_feedforward, bias=True)
my_linear2 = nn.Linear(in_features=d_feedforward, out_features=d_model, bias=True)

my_linear1.weight.data = EncoderLayer.linear1.weight.data
my_linear2.weight.data = EncoderLayer.linear2.weight.data
my_linear1.bias.data = EncoderLayer.linear1.bias.data
my_linear2.bias.data = EncoderLayer.linear2.bias.data

out_linear1 = my_linear1(out_norm1)
out_relu = F.relu(out_linear1)
out_linear2 = my_linear2(out_relu)

### Add & Norm
my_norm2 = nn.LayerNorm(d_model)
out_final = my_norm2(out_norm1+out_linear2) # (L, d_model)

torch.allclose(out_reference_encoder, out_final)

import torch
import torch.nn as nn
DecoderLayer = nn.TransformerDecoderLayer(d_model=d_model,nhead=n_heads, dim_feedforward=d_feedforward, dropout=0, batch_first=True, bias=True)
out_reference_decoder = DecoderLayer(X_proj, X_proj)

### Self Attention
my_self_attn = nn.MultiheadAttention(d_model,n_heads, batch_first=True)

my_self_attn.in_proj_weight.data = DecoderLayer.self_attn.in_proj_weight.data
my_self_attn.out_proj.weight.data = DecoderLayer.self_attn.out_proj.weight.data

my_self_attn.in_proj_bias.data = DecoderLayer.self_attn.in_proj_bias.data
my_self_attn.out_proj.bias.data = DecoderLayer.self_attn.out_proj.bias.data

out_self_attention,_ = my_self_attn(X_proj, X_proj, X_proj) # Since Q,K,V are the same as X_proj, we are preforming self-attention.

### Add & Norm 1
my_norm1 = nn.LayerNorm(d_model)
out_norm1 = my_norm1(X_proj+out_self_attention)

### Cross Attention
my_multihead_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

my_multihead_attn.in_proj_weight.data = DecoderLayer.multihead_attn.in_proj_weight.data
my_multihead_attn.out_proj.weight.data = DecoderLayer.multihead_attn.out_proj.weight.data

my_multihead_attn.in_proj_bias.data = DecoderLayer.multihead_attn.in_proj_bias.data
my_multihead_attn.out_proj.bias.data = DecoderLayer.multihead_attn.out_proj.bias.data

out_cross_attn,_ = my_multihead_attn(out_norm1, X_proj,X_proj)

### Add & Norm 2
my_norm2 = nn.LayerNorm(d_model)
out_norm2 = my_norm2(out_norm1+out_cross_attn)

### Feed Forward Neural Network
my_linear1 = nn.Linear(in_features=d_model,out_features=d_feedforward, bias=True)
my_linear2 = nn.Linear(in_features=d_feedforward, out_features=d_model, bias=True)

my_linear1.weight.data = DecoderLayer.linear1.weight.data
my_linear2.weight.data = DecoderLayer.linear2.weight.data
my_linear1.bias.data = DecoderLayer.linear1.bias.data
my_linear2.bias.data = DecoderLayer.linear2.bias.data

out_linear1 = my_linear1(out_norm2)
out_relu = F.relu(out_linear1)
out_linear2 = my_linear2(out_relu)

### Add & Norm 3
my_norm3 = nn.LayerNorm(d_model)
out_final = my_norm3(out_norm2+out_linear2)

torch.allclose(out_final, out_reference_decoder)

