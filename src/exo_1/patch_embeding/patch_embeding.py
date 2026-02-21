import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, emb_size=768):
        self.img_size = img_size
        self.patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        super().__init__()

        self.linear_proj = nn.Linear(self.patch_dim, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_size))

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Image size must match"

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)

        patches = patches.contiuous().view(B, C, -1, self.patch_size, self.patch_size)

        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.flatten(2)

        x = self.linear_proj(patches)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        
        return x