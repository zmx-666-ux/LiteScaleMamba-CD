import argparse
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


def model_state(checkpoint):
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a model state dict")
    return {
        name: tensor for name, tensor in state.items()
        if name.rsplit(".", 1)[-1] not in ("total_ops", "total_params")
    }


BEST_WEIGHTS = {
    "LEVIR-CD": "LEVIR-CD92.11F1.pth",
    "SYSU-CD": "SYSU-CD84.07F1.pth",
    "WHU-CD": "WHU-CD94.36F1.pth",
}


def default_checkpoint(dataset):
    return Path(__file__).resolve().parent / "weights" / BEST_WEIGHTS[dataset]


def build_parser():
    parser = argparse.ArgumentParser(description="Test LSMamba-CD")
    parser.add_argument("--dataset", choices=("LEVIR-CD", "SYSU-CD", "WHU-CD"), default="LEVIR-CD")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--checkpoint", help="Override the supplied best dataset weight")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main():
    args = build_parser().parse_args()
    import torch
    from torch.utils.data import DataLoader

    from datasets.change_detection import ChangeDetectionDataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the selective scan kernel")
    data = ChangeDetectionDataset(args.test_dir, args.test_list)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=args.workers)
    model = build_model(args.config).cuda()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint(args.dataset)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = model_state(checkpoint)
    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        if checkpoint["model_config"] != load_model_kwargs(args.config):
            raise ValueError("Checkpoint and model config do not match")
    model.load_state_dict(state, strict=True)
    model.eval()
    meter = BinaryMetrics()
    with torch.no_grad():
        for before, after, labels, _ in loader:
            logits = model(before.cuda(non_blocking=True), after.cuda(non_blocking=True))["main"]
            predictions = logits.argmax(dim=1).cpu().numpy()
            meter.update(labels.numpy(), predictions)
    scores = meter.compute()
    for label, key in (("F1", "f1"), ("IoU", "iou"), ("Precision", "precision"),
                       ("Recall", "recall"), ("OA", "oa"), ("Kappa", "kappa")):
        print(f"{label}: {scores[key] * 100:.2f}%")


if __name__ == "__main__":
    main()
