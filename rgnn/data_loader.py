import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class RecursiveDummyDataset(Dataset):
    """
    Synthetic recursive-structured dataset for RGNN testing.
    Generates pseudo-symbolic input sequences with nested patterns.
    """

    def __init__(self, vocab_size=512, seq_len=16, num_samples=1000, num_classes=2, seed=42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.samples = self._generate()

    def _generate(self):
        samples = []
        for _ in range(self.num_samples):
            # Recursive token patterns: even-indexed = stable, odd-indexed = variant
            base = np.random.randint(0, self.vocab_size // 2, self.seq_len // 2)
            variants = np.random.randint(self.vocab_size // 2, self.vocab_size, self.seq_len // 2)
            tokens = np.empty(self.seq_len, dtype=int)
            tokens[::2] = base
            tokens[1::2] = variants

            label = np.sum(base) % self.num_classes  # pseudo-structured label
            samples.append((tokens, label))
        return samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens, label = self.samples[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def get_dataloader(batch_size=32, **kwargs):
    dataset = RecursiveDummyDataset(**kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
