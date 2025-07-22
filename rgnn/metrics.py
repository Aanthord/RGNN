import torch

class MetricsTracker:
    """
    Utility to track and update core metrics:
    - Accuracy
    - CrossEntropy
    - Entropy
    - Total Loss
    """

    def __init__(self):
        self.reset()

    def update(self, preds, targets, loss_dict):
        self.total += 1
        self.correct += (preds.argmax(dim=-1) == targets).sum().item()
        self.total_loss += loss_dict['cross_entropy']
        self.total_entropy += loss_dict['entropy']

    def compute(self):
        accuracy = self.correct / (self.total * 1.0)
        avg_loss = self.total_loss / self.total
        avg_entropy = self.total_entropy / self.total
        return {
            "accuracy": accuracy,
            "avg_cross_entropy": avg_loss,
            "avg_entropy": avg_entropy
        }

    def reset(self):
        self.total = 0
        self.correct = 0
        self.total_loss = 0.0
        self.total_entropy = 0.0

    def report(self, step=None):
        metrics = self.compute()
        if step is not None:
            print(f"[Step {step}] ", end="")
        print(
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Loss: {metrics['avg_cross_entropy']:.4f} | "
            f"Entropy: {metrics['avg_entropy']:.4f}"
        )
