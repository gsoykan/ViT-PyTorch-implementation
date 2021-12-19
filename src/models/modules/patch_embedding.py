import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce
from einops import repeat
from torch import Tensor
import enum


class PatchProjectionMode(enum.Enum):
    Linear = 1
    Conv = 2


class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 512,
                 img_size: int = 224,
                 projection_mode: PatchProjectionMode = PatchProjectionMode.Linear):
        self.patch_size = patch_size
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # in ViT positional embeddings are parameterized for model learn.
        # tensor of shape N_PATCHES + 1 (token), EMBED_SIZE,
        #  each patch would have different position embedding added
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))
        if projection_mode == PatchProjectionMode.Linear:
            self.projection = nn.Sequential(
                # break-down the image in s1 x s2 patches and flat then
                Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',
                          s1=patch_size,
                          s2=patch_size),
                nn.Linear(patch_size * patch_size * in_channels, emb_size)
            )
        elif projection_mode == PatchProjectionMode.Conv:
            self.projection = nn.Sequential(
                # using a conv layer instead of a linear one -> performance gains
                nn.Conv2d(in_channels,
                          emb_size,
                          kernel_size=patch_size,
                          stride=patch_size),
                Rearrange('b e (h) (w) -> b (h w) e'),
            )

    def forward(self, x: Tensor):
        b, _, _, _ = x.shape
        x = self.projection(x)  # [1, 196, 512]
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        # TODO: @gsoykan check how it is done in the original paper
        x = torch.cat([cls_tokens, x], dim=1)  # [1, 197, 512]
        # add position embeddings
        x += self.positions
        return x
