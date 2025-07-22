import torch
import torch.nn as nn
import torch.nn.functional as F
from rgnn.recursive_unit import RecursiveUnit
from rgnn.config import RGNNConfig

class RGNN(nn.Module):
    """
    Recursive Graded Neural Network Core Module
    Supports recursive attention and entropy-modulated state folding.
    """

    def __init__(self, config: RGNNConfig):
        super(RGNN, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Stack of recursive units
        self.recursive_layers = nn.ModuleList([
            RecursiveUnit(config) for _ in range(config.depth)
        ])

        # Final linear projection
        self.output_layer = nn.Linear(config.embed_dim, config.output_dim)

    def forward(self, input_ids):
        """
        Forward pass through recursive layers.
        Each layer updates the hidden state recursively.
        """
        # Embedding input tokens
        x = self.embedding(input_ids)

        # Initial hidden state (batch_size, embed_dim)
        h = torch.zeros(x.size(0), self.config.embed_dim).to(x.device)

        # Process through recursive layers
        for layer in self.recursive_layers:
            h = layer(x, h)

        # Final output projection
        logits = self.output_layer(h)
        return logits
