import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DentalMultiLabelDataset
from src.models.model import MultiLabelLoss, build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)
        total_loss += loss.item()
        all_probs.append(probs.cpu())
        all_targets.append(y.cpu())
    return (
        total_loss / max(len(loader), 1),
        torch.cat(all_probs, dim=0),
        torch.cat(all_targets, dim=0),
    )


def run(args):
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    train_ds = DentalMultiLabelDataset(args.train_csv, train=True, img_size=args.img_size)
    val_ds = DentalMultiLabelDataset(args.val_csv, train=False, img_size=args.img_size)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=4).to(device)
    criterion = MultiLabelLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = eval_one_epoch(model, val_loader, criterion, device)
        print(f"epoch={epoch} train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(ckpt, output_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, output_dir / "best.pt")

    print(f"Training finished. best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/processed/train.csv")
    parser.add_argument("--val-csv", default="data/processed/val.csv")
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    run(parser.parse_args())
