from torch import nn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=768, heads=12, dropout=0.1, mlp_ratio=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(
            emb_size, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(emb_size)

        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(emb_size * mlp_ratio, emb_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x
