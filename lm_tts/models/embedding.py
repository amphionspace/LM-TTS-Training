"""An embedding that trains only the rows appended to a pretrained vocabulary."""

import torch
from torch import nn
from torch.nn import functional as F


class AddedRowsEmbedding(nn.Module):
    """Frozen pretrained rows followed by trainable added rows (e.g. new tag tokens).

    The two lookups avoid concatenating the full table on every forward pass."""

    def __init__(self, embedding, pretrained_rows):
        super().__init__()
        if type(pretrained_rows) is not int or not 0 < pretrained_rows < embedding.num_embeddings:
            raise ValueError("Added rows need a pretrained row count below the vocabulary size")
        weight = embedding.weight.detach()
        self.pretrained = nn.Parameter(weight[:pretrained_rows].clone(), requires_grad=False)
        self.added = nn.Parameter(weight[pretrained_rows:].clone())
        self.num_embeddings, self.padding_idx = embedding.num_embeddings, embedding.padding_idx

    def forward(self, ids):
        rows = len(self.pretrained)
        added = ids >= rows
        return torch.where(
            added[..., None],
            F.embedding((ids - rows).clamp(min=0), self.added),
            F.embedding(ids.clamp(max=rows - 1), self.pretrained),
        )

    @torch.no_grad()
    def merged(self):
        weight = torch.cat([self.pretrained, self.added])
        return nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=self.padding_idx)
