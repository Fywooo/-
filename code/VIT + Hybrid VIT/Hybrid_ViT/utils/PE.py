from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat
from torch import Tensor
import torch


# [batch,channels,h,w] -> [batch,num+1,emb_size] num:拆分平铺后数量  +1:加一个CLS
# 图像分块、Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 512, patch_size: int = 16, emb_size: int = 768, seq_len=None):

        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # kernel_size=块大小，即每个块输出一个值，类似每个块展平后使用相同的全连接层进行处理
            # 输入维度为3，输出维度为块向量长度
            # 与原文中：分块、展平、全连接降维保持一致
            # 输出为[B, C, H, W]
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),

            # (batch_size, channels, high, weight) -> (batch_size, high * weight, channels)
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # img size是长和宽相等的，所以img_size//patch_size就是长和宽有多少个patch + 1(位置0）
        # self.positions = nn.Parameter(torch.randn(14 ** 2 + 1, emb_size))
        self.positions = nn.Parameter(torch.randn(seq_len, emb_size))



    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        # b, _ = x.shape
        # print("---------------------------------------")
        # print(x.shape)

        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
