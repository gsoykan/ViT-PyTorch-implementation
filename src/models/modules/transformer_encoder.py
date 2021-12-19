from torch import nn

from src.models.modules.transformer_encoder_block import TransformerEncoderBlock


class TransformerEncoder(nn.Sequential):
    def __init__(self,
                 depth: int = 12,
                 **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
