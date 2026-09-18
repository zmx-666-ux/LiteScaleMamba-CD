import argparse
import copy
import math
import random
from pathlib import Path

import numpy as np

import yaml


class BinaryMetrics:
    def __init__(self):
        self.confusion = np.zeros((2, 2), dtype=np.int64)

    def reset(self):
        self.confusion.fill(0)

    def update(self, labels, predictions):
        labels = np.asarray(labels)
        predictions = np.asarray(predictions)
        if labels.shape != predictions.shape:
            raise ValueError("Labels and predictions must have the same shape")
        valid = (labels == 0) | (labels == 1)
        selected = predictions[valid]
        if np.any((selected != 0) & (selected != 1)):
            raise ValueError("Predictions must be binary")
        indices = labels[valid].astype(np.int64) * 2 + selected.astype(np.int64)
        self.confusion += np.bincount(indices, minlength=4).reshape(2, 2)

    def compute(self):
        tn, fp, fn, tp = self.confusion.ravel().astype(np.float64)
        total = tn + fp + fn + tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        oa = (tn + tp) / total if total else 0.0
        expected = ((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp)) / total**2 if total else 0.0
        kappa = (oa - expected) / (1 - expected) if total and expected < 1 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "oa": oa,
            "kappa": kappa,
        }


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "lsmamba_cd.yaml"


def load_model_kwargs(config_path=DEFAULT_CONFIG):
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("model"), dict):
        raise ValueError("The config must contain a model mapping")
    return data["model"]


def build_model(config_path=DEFAULT_CONFIG, mobilenet_pretrained=None, vssm_pretrained=None):
    from models.LSMamba_CD import LSMambaCD

    kwargs = load_model_kwargs(config_path)
    return LSMambaCD(
        mobilenet_pretrained=mobilenet_pretrained,
        vssm_pretrained=vssm_pretrained,
        **kwargs,
    )


def lovasz_gradient(sorted_labels):
    count = sorted_labels.numel()
    positives = sorted_labels.sum()
    intersection = positives - sorted_labels.cumsum(0)
    union = positives + (1 - sorted_labels).cumsum(0)
    jaccard = 1.0 - intersection / union
    if count > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1].clone()
    return jaccard


def lovasz_softmax(probabilities, labels, ignore=255):
    import torch

    classes = probabilities.shape[1]
    flattened = probabilities.permute(0, 2, 3, 1).reshape(-1, classes)
    labels = labels.reshape(-1)
    valid = labels != ignore
    flattened = flattened[valid]
    labels = labels[valid]
    if labels.numel() == 0:
        return probabilities.sum() * 0.0
    losses = []
    for class_index in range(classes):
        foreground = (labels == class_index).float()
        if foreground.sum() == 0:
            continue
        errors = (foreground - flattened[:, class_index]).abs()
        errors_sorted, order = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_gradient(foreground[order])))
    return torch.stack(losses).mean()


def model_state(checkpoint):
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a model state dict")
    return {
        name: tensor for name, tensor in state.items()
        if name.rsplit(".", 1)[-1] not in ("total_ops", "total_params")
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Train LSMamba-CD")
    parser.add_argument("--dataset", choices=("LEVIR-CD", "SYSU-CD", "WHU-CD"), default="LEVIR-CD")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mobilenet-pretrained")
    parser.add_argument("--vssm-pretrained")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-iters", type=int, default=8000)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--pos-weight-seg", type=float, default=3.0)
    parser.add_argument("--lovasz-weight", type=float, default=0.75)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--disable-ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def compute_loss(outputs, labels, pos_weight_seg, lovasz_weight):
    import torch
    import torch.nn.functional as F

    class_weights = labels.new_tensor([1.0, pos_weight_seg], dtype=torch.float32)

    def cross_entropy(logits):
        return F.cross_entropy(logits, labels, weight=class_weights, ignore_index=255)

    def segmentation_loss(logits):
        return cross_entropy(logits) + lovasz_weight * lovasz_softmax(
            F.softmax(logits, dim=1), labels, ignore=255
        )

    loss = (segmentation_loss(outputs["main"])
            + 0.3 * segmentation_loss(outputs["aux2"])
            + 0.2 * segmentation_loss(outputs["aux3"])
            + 0.1 * cross_entropy(outputs["aux4"]))
    if "delta_prior" in outputs:
        prior = outputs["delta_prior"].clamp(1e-6, 1.0 - 1e-6)
        target = F.adaptive_max_pool2d((labels == 1).float().unsqueeze(1), prior.shape[-2:])
        intersection = (prior * target).sum(dim=(1, 2, 3))
        dice = 1.0 - (2 * intersection + 1.0) / (
            prior.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
        )
        loss = loss + 0.1 * (F.binary_cross_entropy(prior, target) + dice.mean())
    return loss


class ModelEMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.decay = decay

    def update(self, model):
        import torch

        with torch.no_grad():
            source = model.state_dict()
            for name, averaged in self.model.state_dict().items():
                current = source[name].detach()
                if averaged.dtype.is_floating_point:
                    averaged.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
                else:
                    averaged.copy_(current)


def make_optimizer(model, learning_rate, weight_decay):
    import torch

    modules = (
        (model.encoder.cnn, 0.1),
        (model.encoder.vss, 0.5),
        (model.encoder.msdf, 1.0),
        (model.encoder.dam, 1.0),
        (model.encoder.lfmwb, 1.0),
        (model.encoder.stage3_adapter, 1.0),
        (model.encoder.cnn_residual_proj, 1.0),
        (model.decoder, 1.0),
        (model.main_clf, 1.0),
        (model.freq_refine, 1.0),
        (model.upsample_refine, 1.0),
        (model.refine_head, 1.0),
    )
    groups = []
    grouped_ids = set()
    for module, multiplier in modules:
        parameters = list(module.parameters())
        groups.append({"params": parameters, "lr": learning_rate * multiplier})
        for parameter in parameters:
            if id(parameter) in grouped_ids:
                raise ValueError("A model parameter appears in multiple optimizer groups")
            grouped_ids.add(id(parameter))
    required_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if grouped_ids != required_ids:
        raise ValueError("Optimizer groups do not cover all trainable model parameters")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def learning_rate_factor(step, total_steps, warmup_iters, base_lr, min_lr):
    if step < warmup_iters:
        return (step + 1) / max(warmup_iters, 1)
    minimum = min_lr / base_lr
    progress = min(max((step - warmup_iters) / max(total_steps - warmup_iters, 1), 0.0), 1.0)
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(model, loader, device):
    import torch

    model.eval()
    meter = BinaryMetrics()
    with torch.no_grad():
        for before, after, labels, _ in loader:
            outputs = model(before.to(device), after.to(device))
            predictions = outputs["main"].argmax(dim=1).cpu().numpy()
            meter.update(labels.numpy(), predictions)
    return meter.compute()


def main():
    args = build_parser().parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    import torch
    from torch.utils.data import DataLoader

    from datasets.change_detection import ChangeDetectionDataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the selective scan kernel")
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_data = ChangeDetectionDataset(args.train_dir, args.train_list, train=True,
                                        crop_size=args.crop_size)
    val_data = ChangeDetectionDataset(args.val_dir, args.val_list)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    model = build_model(args.config, args.mobilenet_pretrained, args.vssm_pretrained).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        state = model_state(checkpoint)
        model.load_state_dict(state, strict=True)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    initial_lrs = [group["lr"] for group in optimizer.param_groups]
    ema = None if args.disable_ema else ModelEMA(model, args.ema_decay)
    total_steps = len(train_loader) * args.epochs
    step = 0
    best_f1 = -1.0
    save_path = Path(args.output_dir) / args.dataset / f"seed_{args.seed}" / "best.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for before, after, labels, _ in train_loader:
            factor = learning_rate_factor(step, total_steps, args.warmup_iters,
                                          args.learning_rate, args.min_lr)
            for group, initial in zip(optimizer.param_groups, initial_lrs):
                group["lr"] = initial * factor
            before = before.to(device, non_blocking=True)
            after = after.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(before, after)
            loss = compute_loss(outputs, labels, args.pos_weight_seg, args.lovasz_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            epoch_loss += loss.item()
            step += 1
        evaluation_model = ema.model if ema is not None else model
        scores = evaluate(evaluation_model, val_loader, device)
        if scores["f1"] > best_f1:
            best_f1 = scores["f1"]
            torch.save({"model": evaluation_model.state_dict(),
                        "epoch": epoch, "val_metrics": scores,
                        "model_config": load_model_kwargs(args.config)}, save_path)
        print(f"Epoch {epoch:03d}/{args.epochs} | Loss {epoch_loss / len(train_loader):.4f} | "
              f"Val F1 {scores['f1'] * 100:.2f}% | IoU {scores['iou'] * 100:.2f}%")
    print(f"Best validation F1: {best_f1 * 100:.2f}%")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
