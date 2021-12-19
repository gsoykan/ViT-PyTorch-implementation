from torch import nn

# TODO: @gsoykan interesting usage
class FeedForwardBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 512,
                 expansion: int = 4,
                 drop_p: float = 0.
                 ):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )
