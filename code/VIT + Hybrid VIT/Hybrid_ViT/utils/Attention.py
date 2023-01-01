from torch import nn
from torch import Tensor
from einops import rearrange
import torch
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0.1, seq_len: int = 197):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        # 得到q、k、v，这里也可以通过一个全连接层得到：self.qkv = nn.Linear(dim, dim * 3)

        # 然后做一个dropout
        self.att_drop = nn.Dropout(dropout)

        # 针对每一个header的dim：是对qkv的进一步分解
        self.gate1 = nn.Linear(emb_size//num_heads, emb_size//num_heads)
        self.gate2 = nn.Linear(emb_size//num_heads, emb_size//num_heads)
        self.gate3 = nn.Linear(emb_size//num_heads, 1)
        
        self.scaling = (self.emb_size // num_heads) ** -0.5  # scaling 就是根号下的 1 / Q^k


        self.W = nn.Parameter(torch.randn(seq_len,seq_len))
        self.Norm = nn.LayerNorm([seq_len,seq_len])

        # 最终的线性层
        self.projection = nn.Linear(emb_size, emb_size)






    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # 分解q、k、v
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)


        # energy = QK
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        att = F.softmax(energy * self.scaling, dim=-1)
        # QK / d ** -0.5

        att = self.Norm(self.W * att)
        att = self.att_drop(att)
        # attention(Q,K,V) = softmax(QK / dk^1/2)

        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        # attention(Q,K,V) = softmax(QK / dk^1/2) * V

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
