import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from utils.metrics import BinaryMetrics
from utils.model import DEFAULT_CONFIG


def build_parser():
    parser = argparse.ArgumentParser(description="Test HybridBCD")
    parser.add_argument("--dataset", choices=("LEVIR-CD", "SYSU-CD", "WHU-CD"), default="LEVIR-CD")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-dir")
    return parser


def main():
    args = build_parser().parse_args()
    import torch
    from torch.utils.data import DataLoader

    from datasets.change_detection import ChangeDetectionDataset
    from utils.model import build_model, load_model_kwargs

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the selective scan kernel")
    data = ChangeDetectionDataset(args.test_dir, args.test_list)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=args.workers)
    model = build_model(args.config).cuda()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        if checkpoint["model_config"] != load_model_kwargs(args.config):
            raise ValueError("Checkpoint and model config do not match")
    model.load_state_dict(state, strict=True)
    model.eval()
    meter = BinaryMetrics()
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for before, after, labels, names in loader:
            logits = model(before.cuda(non_blocking=True), after.cuda(non_blocking=True))["main"]
            predictions = logits.argmax(dim=1).cpu().numpy()
            meter.update(labels.numpy(), predictions)
            if save_dir is not None:
                Image.fromarray((predictions[0] * 255).astype(np.uint8)).save(
                    save_dir / f"{Path(names[0]).stem}.png"
                )
    scores = meter.compute()
    for label, key in (("F1", "f1"), ("IoU", "iou"), ("Precision", "precision"),
                       ("Recall", "recall"), ("OA", "oa"), ("Kappa", "kappa")):
        print(f"{label}: {scores[key] * 100:.2f}%")


if __name__ == "__main__":
    main()
