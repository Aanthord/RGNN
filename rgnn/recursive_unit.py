# rgnn/recursive_unit.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursiveUnit(nn.Module):
    """
    Basic Recursive Unit with gating and symbolic bridge potential
    """

    def __init__(self, config):
        super(RecursiveUnit, self).__init__()
        self.config = config
        self.input_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.hidden_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.gate = nn.Sigmoid()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, h_prev):
        """
        x: (batch_size, seq_len, embed_dim)
        h_prev: (batch_size, embed_dim)
        """
        x_pooled = x.mean(dim=1)  # Mean pooling across sequence
        x_proj = self.input_proj(x_pooled)
        h_proj = self.hidden_proj(h_prev)

        gated = self.gate(x_proj + h_proj)
        h_new = gated * torch.tanh(x_proj + h_proj)
        return self.dropout(h_new)
