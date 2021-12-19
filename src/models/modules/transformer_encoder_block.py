from torch import nn

from src.models.modules.feed_forward_block import FeedForwardBlock
from src.models.modules.multi_head_attention import MultiHeadAttention
from src.models.modules.residual_add import ResidualAdd


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=512,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p)
                )
            ),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size,
                                 expansion=forward_expansion,
                                 drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )
