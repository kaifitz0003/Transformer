"""
The following code demonstrates multi-head attention by comparing manual computation with PyTorch’s built in functions.

This code generates random input data and computes query (Q), key (K), and value (V) matrices using linear projections.
It first applies scaled dot-product attention manually and makes sure that it matches PyTorch’s built-in function.
The internal projection weights are extracted and split into separate heads.
Attention is computed independently for each head using scaled dot-product attention.
The outputs of all heads are concatenated and passed through the final projection layer.
Finally, the result is compared to PyTorch’s result.
"""

import torch
import torch.nn
import torch.nn.functional as F
L=5 # Time (9am, 12pm, etc)
d_in=4 # These are the variables/features that we are measuring (Humidity, Temperature, etc)
d_head=2
n_heads=3
d_model=d_head*n_heads # 6
X = torch.rand(L,d_in)
WQ = torch.rand(d_model,d_in)
Q = X@WQ.T # (L,d_model)
WK = torch.rand(d_model,d_in)
K = X@WK.T # (L,d_model)
WV = torch.rand(d_model,d_in)
V = X@WV.T # (L,d_model)

sdpa_reference=F.scaled_dot_product_attention(Q,K,V)
scores = (Q@K.T)/(d_model**0.5)
weights = scores.softmax(axis=1)
sdpa_manual = weights@V
torch.allclose(sdpa_reference,sdpa_manual)
mha_model=torch.nn.MultiheadAttention(d_model,num_heads=n_heads,bias=False,batch_first=True)
attention_reference,_=mha_model(Q,K,V)
mha_model.in_proj_weight.shape

#print(model.in_proj_weight.shape) # (3*d_model,d_model)
WQ_mha=mha_model.in_proj_weight[:d_model,:]
WQ_H1=WQ_mha[:d_head,:]
WQ_H2=WQ_mha[d_head:2*d_head,:]
WQ_H3=WQ_mha[2*d_head:,:]
WK_mha=mha_model.in_proj_weight[d_model:d_model*2,:]
WK_H1=WK_mha[:d_head,:]
WK_H2=WK_mha[d_head:2*d_head,:]
WK_H3=WK_mha[2*d_head:,:]
WV_mha=mha_model.in_proj_weight[d_model*2:,:]
WV_H1=WV_mha[:d_head,:]
WV_H2=WV_mha[d_head:2*d_head,:]
WV_H3=WV_mha[2*d_head:,:]

Q1=Q@WQ_H1.T
Q2=Q@WQ_H2.T
Q3=Q@WQ_H3.T

K1 = K @ WK_H1.T
K2 = K @ WK_H2.T
K3 = K @ WK_H3.T

V1 = V@WV_H1.T
V2 = V@WV_H2.T
V3 = V@WV_H3.T

#H1:
#head1 = (Q1@K1.T)/d_head
#weights1=head1.softmax(dim=1)
#output1=weights1@V1
head1=F.scaled_dot_product_attention(Q1,K1,V1)
head2=F.scaled_dot_product_attention(Q2,K2,V2)
head3=F.scaled_dot_product_attention(Q3,K3,V3)

h=torch.cat([head1,head2,head3],dim=1) # (L,d_model)

WO = mha_model.out_proj.weight.data
out_final = h@WO.T
torch.allclose(out_final,attention_reference)
