import os

import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision import transforms

from cifar_cnn import ResNet
from dataset import CIFAR10TrainDataset


cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

class_to_idx = {cls_name: idx for idx, cls_name in enumerate(cifar10_classes)}
idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])


def eval_metrics(model, loader, num_classes, device):
    model.eval()
    correct = 0
    total = 0
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)

            correct += (pred == y).sum().item()
            total += y.numel()

            for t, p in zip(y.view(-1), pred.view(-1)):
                conf[t.long(), p.long()] += 1

    acc = correct / max(total, 1)
    precision = []
    recall = []
    for c in range(num_classes):
        tp = conf[c, c].item()
        fp = conf[:, c].sum().item() - tp
        fn = conf[c, :].sum().item() - tp
        precision.append(tp / max(tp + fp, 1))
        recall.append(tp / max(tp + fn, 1))
    return acc, precision, recall


def load_state_dict(model, model_path, device):
    state = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)

def plot_precision_recall(precision, recall, class_names, output_path):
    x = list(range(len(class_names)))
    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - width / 2 for i in x], precision, width=width, label="precision")
    ax.bar([i + width / 2 for i in x], recall, width=width, label="recall")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0.9, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Per-class Precision/Recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)

def main():
    model_path = "./runs/baseline/best_model_baseline_acc_0.9359.pt"
    dataset_dir = "./dataset"
    batch_size = 256
    output_path = "./runs/baseline/precision_recall.png"

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    labels_csv = os.path.join(dataset_dir, "trainLabels.csv")
    images_dir = os.path.join(dataset_dir, "train")

    dataset = CIFAR10TrainDataset(
        images_dir=images_dir,
        labels_csv=labels_csv,
        transform=eval_transform,
        class_to_idx=class_to_idx,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    model = ResNet(3, 10).to(device)
    load_state_dict(model, model_path, device)

    acc, precision, recall = eval_metrics(model, loader, num_classes=10, device=device)
    print(f"accuracy: {acc:.4f}")
    for i, (p, r) in enumerate(zip(precision, recall)):
        name = idx_to_class.get(i, str(i))
        print(f"{name}: precision={p:.4f}, recall={r:.4f}")
    print(f"macro precision: {sum(precision) / len(precision):.4f}")
    print(f"macro recall: {sum(recall) / len(recall):.4f}")
    plot_precision_recall(precision, recall, cifar10_classes, output_path)
    print(f"saved chart: {output_path}")


if __name__ == "__main__":
    main()
