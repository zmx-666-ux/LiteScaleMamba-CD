import torch


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
