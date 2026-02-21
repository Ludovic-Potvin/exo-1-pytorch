from torch import nn

from patch_embeding.patch_embeding import PatchEmbedding
from patch_embeding.transformer_encoder import TransformerEncoderBlock


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        emb_size=768,
        depth=12,
        heads=12,
        mlp_ratio=4,
        num_classes=1000,
    ):
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)

        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(emb_size, heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)

        x = self.norm(x)

        return self.head(x[:, 0])


