import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce
from einops import repeat, rearrange
from torch import Tensor
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 emb_size: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0,
                 use_single_matrix_for_qkv: bool = True):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.use_single_matrix_for_qkv = use_single_matrix_for_qkv
        # creating k, q, v of all heads at the same time
        # TODO: @gsoykan should we have bias or not?
        if use_single_matrix_for_qkv:
            self.qkv = nn.Linear(emb_size, 3 * emb_size)
        else:
            self.keys = nn.Linear(emb_size, emb_size)
            self.queries = nn.Linear(emb_size, emb_size)
            self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self,
                x: Tensor,
                mask: Tensor = None) -> Tensor:
        if self.use_single_matrix_for_qkv:
            qkv = rearrange(self.qkv(x),
                            "b n (h d qkv) -> (qkv) b h n d",
                            h=self.num_heads,
                            qkv=3)
            queries, keys, values = qkv[0], qkv[1], qkv[2]
        else:
            # split keys, queries, values in num_heads
            queries = rearrange(self.queries(x), "b n (h d) -> b h n d",
                                h=self.num_heads)  # BATCH, HEADS, SEQUENCE_LEN, EMBEDDING_SIZE
            keys = rearrange(self.keys(x), "b n (h d) -> b h n d",
                             h=self.num_heads)  # BATCH, HEADS, SEQUENCE_LEN, EMBEDDING_SIZE
            values = rearrange(self.values(x), "b n (h d) -> b h n d",
                               h=self.num_heads)  # BATCH, HEADS, SEQUENCE_LEN, EMBEDDING_SIZE
        # TODO: @gsoykan understand einsum issue better and explain it in the class.
        # sum up over the last axis
        # torch.einsum source: https://pytorch.org/docs/stable/generated/torch.einsum.html
        #   einsum: Einstein summation convention
        # For example, matrix multiplication can be computed using einsum as torch.einsum(“ij,jk->ik”, A, B).
        #   Here, j is the summation subscript and i and k the output subscripts
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # BATCH, HEADS, QUERY_LEN, KEY_LEN
        if mask is not None:
            # A torch.finfo is an object that represents the numerical properties of a floating point torch.dtype,
            # (i.e. torch.float32, torch.float64, and torch.float16). This is similar to numpy.finfo.
            fill_value = torch.finfo(torch.float32).min  # represents -inf
            # (~): the bitwise negation operator
            energy.mask_fill = (~mask, fill_value)
        # TODO: @gsoykan this should have been sqrt(dk), dk = self.emb_size / num_heads ?
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav', att, values)  # BATCH, HEADS, VALUES_LEN, SINGLE_EMBEDDING_SIZE
        out = rearrange(out, "b h n d -> b n (h d)")  # Can be considered as concatting
        out = self.projection(out)  # BATCH, SEQUENCE_LEN, EMBEDDING_SIZE
        return out
