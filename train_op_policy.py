"""Train a lightweight OP-SieveKV segment retention policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from op_policy_model import FEATURE_NAMES, SegmentRetentionMLP, save_policy_checkpoint


def print_dataset_summary(labels: torch.Tensor, weights: torch.Tensor, metadata: list[dict] | None) -> None:
    """Print compact label diagnostics before training."""
    hard_total = int((labels >= 0.5).sum().item())
    print(
        f"Dataset: rows={labels.numel()} hard={hard_total} "
        f"label_mass={float(labels.sum().item()):.1f} mean_weight={float(weights.mean().item()):.3f}"
    )
    if not metadata:
        return

    budget_to_indices: dict[float, list[int]] = {}
    for idx, item in enumerate(metadata):
        if "budget" not in item:
            continue
        budget = round(float(item["budget"]), 4)
        budget_to_indices.setdefault(budget, []).append(idx)
    if not budget_to_indices:
        return

    print("Dataset by budget:")
    for budget in sorted(budget_to_indices):
        idx = torch.tensor(budget_to_indices[budget], dtype=torch.long)
        budget_labels = labels[idx]
        budget_weights = weights[idx]
        print(
            f"  b={budget:g}: rows={idx.numel()} hard={int((budget_labels >= 0.5).sum().item())} "
            f"label_mass={float(budget_labels.sum().item()):.1f} "
            f"mean_weight={float(budget_weights.mean().item()):.3f}"
        )


def weighted_bce_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp(min=1e-6)


@torch.no_grad()
def evaluate(model, loader, feature_mean, feature_std, device) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    positives = 0
    predicted_positive = 0
    prob_sum = 0.0
    label_sum = 0.0
    abs_error_sum = 0.0
    for features, labels, weights in loader:
        features = ((features.to(device) - feature_mean) / feature_std).float()
        labels = labels.to(device).float()
        weights = weights.to(device).float()
        logits = model(features)
        loss = weighted_bce_loss(logits, labels, weights)
        probs = torch.sigmoid(logits)
        pred = probs >= 0.5
        total_loss += float(loss.item()) * labels.numel()
        total += labels.numel()
        correct += int((pred == (labels >= 0.5)).sum().item())
        positives += int((labels >= 0.5).sum().item())
        predicted_positive += int(pred.sum().item())
        prob_sum += float(probs.sum().item())
        label_sum += float(labels.sum().item())
        abs_error_sum += float(torch.abs(probs - labels).sum().item())

    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "positives": positives,
        "predicted_positive": predicted_positive,
        "mean_prob": prob_sum / max(1, total),
        "mean_label": label_sum / max(1, total),
        "mae": abs_error_sum / max(1, total),
        "total": total,
    }


@torch.no_grad()
def evaluate_by_budget(
    model,
    features: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    metadata: list[dict] | None,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    device,
) -> dict[float, dict]:
    """Evaluate validation calibration separately for each budget."""
    if not metadata:
        return {}

    budget_to_indices: dict[float, list[int]] = {}
    for idx in indices.tolist():
        item = metadata[idx] if idx < len(metadata) else {}
        if "budget" not in item:
            continue
        budget = round(float(item["budget"]), 4)
        budget_to_indices.setdefault(budget, []).append(idx)
    if not budget_to_indices:
        return {}

    model.eval()
    results: dict[float, dict] = {}
    for budget, row_indices in sorted(budget_to_indices.items()):
        idx = torch.tensor(row_indices, dtype=torch.long)
        batch_features = ((features[idx].to(device) - feature_mean) / feature_std).float()
        batch_labels = labels[idx].to(device).float()
        batch_weights = weights[idx].to(device).float()
        logits = model(batch_features)
        probs = torch.sigmoid(logits)
        pred = probs >= 0.5
        loss = weighted_bce_loss(logits, batch_labels, batch_weights)
        results[budget] = {
            "loss": float(loss.item()),
            "accuracy": float((pred == (batch_labels >= 0.5)).float().mean().item()),
            "mae": float(torch.abs(probs - batch_labels).mean().item()),
            "mean_prob": float(probs.mean().item()),
            "mean_label": float(batch_labels.mean().item()),
            "predicted_positive": int(pred.sum().item()),
            "hard_positive": int((batch_labels >= 0.5).sum().item()),
            "total": int(batch_labels.numel()),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OP-SieveKV learned segment policy")
    parser.add_argument("--dataset", default="results/op_distill_dataset.pt")
    parser.add_argument("--output", default="results/op_policy.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--print-budget-metrics-every",
        type=int,
        default=25,
        help="Print validation metrics split by budget every N epochs; use 0 to disable.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after N epochs without validation-loss improvement; 0 disables early stopping.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = torch.load(Path(args.dataset), map_location="cpu")
    features = dataset["features"].float()
    labels = dataset["labels"].float()
    weights = dataset["weights"].float()
    if dataset.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Dataset feature schema does not match current OP policy feature schema.")
    if features.numel() == 0:
        raise ValueError("Empty distillation dataset.")
    print_dataset_summary(labels, weights, dataset.get("metadata"))

    perm = torch.randperm(features.shape[0])
    val_count = max(1, int(features.shape[0] * args.val_ratio))
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    if train_idx.numel() == 0:
        raise ValueError("Not enough rows for a train/validation split.")

    train_features = features[train_idx]
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0, unbiased=False).clamp(min=1e-6)

    train_ds = TensorDataset(features[train_idx], labels[train_idx], weights[train_idx])
    val_ds = TensorDataset(features[val_idx], labels[val_idx], weights[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegmentRetentionMLP(input_dim=features.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    feature_mean = feature_mean.to(device)
    feature_std = feature_std.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0
        for batch_features, batch_labels, batch_weights in train_loader:
            batch_features = ((batch_features.to(device) - feature_mean) / feature_std).float()
            batch_labels = batch_labels.to(device).float()
            batch_weights = batch_weights.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = weighted_bce_loss(logits, batch_labels, batch_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * batch_labels.numel()
            total += batch_labels.numel()

        val_metrics = evaluate(model, val_loader, feature_mean, feature_std, device)
        train_loss = epoch_loss / max(1, total)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_mean_prob={val_metrics['mean_prob']:.3f}/{val_metrics['mean_label']:.3f} "
            f"val_pred_pos={val_metrics['predicted_positive']}/{val_metrics['total']}"
        )
        if args.print_budget_metrics_every > 0 and (
            epoch == 1 or epoch % args.print_budget_metrics_every == 0 or epoch == args.epochs
        ):
            budget_metrics = evaluate_by_budget(
                model,
                features,
                labels,
                weights,
                val_idx,
                dataset.get("metadata"),
                feature_mean,
                feature_std,
                device,
            )
            if budget_metrics:
                print("  val_by_budget:")
                for budget, metrics in budget_metrics.items():
                    print(
                        f"    b={budget:g}: loss={metrics['loss']:.4f} "
                        f"mae={metrics['mae']:.4f} "
                        f"mean_prob={metrics['mean_prob']:.3f}/{metrics['mean_label']:.3f} "
                        f"pred_pos={metrics['predicted_positive']}/{metrics['total']} "
                        f"hard={metrics['hard_positive']}"
                    )

        if val_metrics["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best_val_loss={best_val_loss:.4f}"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metadata = {
        "dataset": args.dataset,
        "feature_names": FEATURE_NAMES,
        "train_rows": int(train_idx.numel()),
        "val_rows": int(val_idx.numel()),
        "hard_positive_rows": int((labels >= 0.5).sum().item()),
        "label_mass": float(labels.sum().item()),
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    save_policy_checkpoint(
        args.output,
        model.cpu(),
        feature_mean=feature_mean.cpu(),
        feature_std=feature_std.cpu(),
        metadata=metadata,
    )
    print(f"Saved learned OP policy to {args.output}")


if __name__ == "__main__":
    main()
