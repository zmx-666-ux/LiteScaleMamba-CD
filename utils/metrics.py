import numpy as np


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
