from torch import nn

from src.models.modules.classification_head import ClassificationHead
from src.models.modules.patch_embedding import PatchEmbedding
from src.models.modules.transformer_encoder import TransformerEncoder


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 224,
                 depth: int = 12,
                 n_classes: int = 100,
                 use_mean_of_outputs_instead_of_cls: bool = False,
                 **kwargs
                 ):
        super().__init__(
            PatchEmbedding(in_channels,
                           patch_size,
                           emb_size,
                           img_size),
            TransformerEncoder(depth,
                               emb_size=emb_size,
                               **kwargs),
            ClassificationHead(emb_size,
                               n_classes,
                               use_mean_of_outputs_instead_of_cls)
        )
