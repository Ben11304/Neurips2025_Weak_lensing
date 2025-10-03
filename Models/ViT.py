import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from collections import defaultdict
import random
from torch.utils.data import BatchSampler

# Helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Classes
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            #multilayer attention
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # residual addition
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class CustomViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head=64, dropout=0., emb_dropout=0., spec_dim=10, output_dim=2):
        super().__init__()
        self.name = "CustomViT"
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert num_patches >= 16, 'Number of patches must be >= 16'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1 if pool == 'cls' else num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) if pool == 'cls' else None
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.spec_mlp = nn.Sequential(
            nn.Linear(spec_dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )


        self.regression_head = nn.Linear(dim + dim, output_dim)

    def forward(self, input):
        img,specs=input
        # img: [batch_size, channels, image_height, image_width]
        # specs: [batch_size, spec_dim]
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
        else:
            x += self.pos_embedding

        x = self.dropout(x)
        x = self.transformer(x)

        # Pooling
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        features = self.to_latent(x)  # [batch_size, dim]

        spec_features = self.spec_mlp(specs)  # [batch_size, dim]

        combined_features = torch.cat((features, spec_features), dim=1)  # [batch_size, dim + dim]

        output = self.regression_head(combined_features)  # [batch_size, output_dim]

        # return output, features
        return output





# from types import SimpleNamespace

# # Example config
# config = SimpleNamespace(
#     EPOCHS=50,
#     BATCH_SIZE=8,  # Adjust based on GPU memory
#     LEARNING_RATE=1e-3,
#     DEVICE='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
#     MODEL_SAVE_PATH=None
# )

# # Instantiate CustomViT
# model = CustomViT(
#     image_size=(256, 512),  # Example: adjust to your map size (height, width)
#     patch_size=16,          # patch_size=(16, 16)
#     dim=768,                # Feature dimension
#     depth=12,               # Number of Transformer blocks
#     heads=12,               # Number of attention heads
#     mlp_dim=3072,           # MLP dimension (4 * dim)
#     channels=1,             # Weak lensing maps are 1-channel
#     dim_head=64,
#     dropout=0.1,
#     emb_dropout=0.1,
#     pool='cls',             # or 'mean'
#     spec_dim=10,            # Power spectrum dimension
#     output_dim=2            # For Ω_m, S_8
# )

# optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
# criterion = nn.MSELoss()