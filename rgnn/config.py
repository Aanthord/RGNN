# rgnn/config.py

class RGNNConfig:
    def __init__(
        self,
        vocab_size=2048,
        embed_dim=256,
        output_dim=2,
        depth=4,
        dropout=0.1
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.depth = depth
        self.dropout = dropout
