import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursiveEntropyLoss(nn.Module):
    """
    Custom loss function for RGNN that includes:
    - CrossEntropy for supervised labels
    - Entropy penalty to reward structural certainty
    """

    def __init__(self, entropy_weight=0.05):
        super(RecursiveEntropyLoss, self).__init__()
        self.entropy_weight = entropy_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size,)
        """
        ce = self.ce_loss(logits, targets)

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)

        # Shannon entropy for each sample
        entropy = -torch.sum(probs * log_probs, dim=-1)
        avg_entropy = torch.mean(entropy)

        # Total loss = CE + entropy penalty (we want low uncertainty)
        loss = ce + self.entropy_weight * avg_entropy
        return loss, {'cross_entropy': ce.item(), 'entropy': avg_entropy.item()}
