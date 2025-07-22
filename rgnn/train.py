import torch
from torch import optim
from torch.nn import functional as F

from core import RGNN
from loss import RGNNLoss
from data_loader import get_dataloader
from metrics import MetricsTracker

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    metrics = MetricsTracker()

    for step, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss, loss_components = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        metrics.update(outputs, targets, loss_components)

        if step % 10 == 0:
            metrics.report(step)

    return metrics.compute()

def validate(model, dataloader, criterion, device):
    model.eval()
    metrics = MetricsTracker()

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss, loss_components = criterion(outputs, targets)
            metrics.update(outputs, targets, loss_components)

    return metrics.compute()

def run_training(
    vocab_size=512,
    seq_len=16,
    num_classes=2,
    embed_dim=64,
    hidden_dim=128,
    depth=2,
    epochs=5,
    batch_size=32,
    learning_rate=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Initialize components
    model = RGNN(vocab_size, embed_dim, hidden_dim, num_classes, depth).to(device)
    criterion = RGNNLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = get_dataloader(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_classes=num_classes,
        batch_size=batch_size,
        num_samples=512
    )

    val_loader = get_dataloader(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_classes=num_classes,
        batch_size=batch_size,
        num_samples=128
    )

    print("🚀 Starting RGNN training...")
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1} ===")
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train - Acc: {train_metrics['accuracy']:.4f}")

        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Val   - Acc: {val_metrics['accuracy']:.4f}")

    print("\n✅ Training complete.")
    return model

if __name__ == "__main__":
    run_training()
