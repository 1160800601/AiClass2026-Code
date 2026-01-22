import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset, random_split

from cifar_cnn import ResNet
from dataset import CIFAR10TestDataset, CIFAR10TrainDataset
import utils

run_name = "00"

g_eval_batch_size = 10000
g_num_epochs = 20
g_lr = 0.01
g_batch_size = 512
g_weight_decay = 1e-4

# Dataset classes
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

# Class-to-index mapping
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(cifar10_classes)}
# Index-to-class mapping
idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}


class CachedDataset(Dataset):
    def __init__(self, dataset):
        self.samples = [dataset[i] for i in range(len(dataset))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=g_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=g_eval_batch_size)
    parser.add_argument("--epochs", type=int, default=g_num_epochs)
    parser.add_argument("--lr", type=float, default=g_lr)
    parser.add_argument("--weight-decay", type=float, default=g_weight_decay)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument("--no-persistent-workers", action="store_false", dest="persistent_workers")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--drop-last", action="store_true", default=False)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--cudnn-benchmark", action="store_true", default=True)
    parser.add_argument("--no-cudnn-benchmark", action="store_false", dest="cudnn_benchmark")
    parser.add_argument("--use-amp", action="store_true", default=False)
    parser.add_argument("--use-compile", action="store_true", default=False)
    parser.add_argument("--cache-data", action="store_true", default=True)
    parser.add_argument("--no-cache-data", action="store_false", dest="cache_data")
    parser.add_argument("--synthetic-bench", action="store_true", default=True)
    parser.add_argument("--no-synthetic-bench", action="store_false", dest="synthetic_bench")
    parser.add_argument("--synthetic-iters", type=int, default=200)
    parser.add_argument("--batch-sweep", action="store_true", default=False)
    parser.add_argument("--batch-sweep-sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--batch-sweep-iters", type=int, default=50)
    parser.add_argument("--preview-batch", action="store_true", default=False)
    return parser.parse_args()


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_cuda_memory_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    alloc = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    max_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return alloc, reserved, max_alloc


def synthetic_benchmark(model_factory, device, batch_size, iters, use_amp):
    model = model_factory().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    model.train()
    X = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        sync_if_cuda(device)

    start = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    images_sec = (batch_size * iters) / max(elapsed, 1e-9)
    alloc, reserved, max_alloc = get_cuda_memory_mb()
    return images_sec, alloc, reserved, max_alloc


def batch_size_sweep(model_factory, device, sizes, iters, use_amp):
    print("Batch size sweep:")
    for bs in sizes:
        try:
            ips, alloc, reserved, max_alloc = synthetic_benchmark(
                model_factory, device, bs, iters, use_amp
            )
            print(
                f"  bs={bs} | throughput={ips:.1f} img/s | "
                f"mem_alloc={alloc:.0f}MB | mem_reserved={reserved:.0f}MB | max_alloc={max_alloc:.0f}MB"
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"  bs={bs} | OOM")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            raise


def train_one_epoch(
    model,
    optimizer,
    loss_fn,
    scaler,
    train_loader,
    device,
    n_train,
    epoch,
    log_interval,
    use_amp,
):
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    totals = {"data": 0.0, "h2d": 0.0, "fwd_bwd": 0.0, "opt": 0.0, "step": 0.0}
    interval = {"data": 0.0, "h2d": 0.0, "fwd_bwd": 0.0, "opt": 0.0, "step": 0.0}
    interval_images = 0
    epoch_images = 0

    epoch_start = time.perf_counter()
    prev_end = epoch_start
    correct_num = 0
    epoch_loss = 0.0
    step = 0

    for X_batch, y_batch in train_loader:
        iter_start = time.perf_counter()
        data_time = iter_start - prev_end

        sync_if_cuda(device)
        t0 = time.perf_counter()
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        sync_if_cuda(device)
        h2d_time = time.perf_counter() - t0

        sync_if_cuda(device)
        t1 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        sync_if_cuda(device)
        fwd_bwd_time = time.perf_counter() - t1

        sync_if_cuda(device)
        t2 = time.perf_counter()
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        sync_if_cuda(device)
        opt_time = time.perf_counter() - t2

        iter_end = time.perf_counter()
        step_time = iter_end - iter_start
        prev_end = iter_end

        correct_num += (torch.argmax(y_pred, dim=1) == y_batch).sum().item()
        epoch_loss += loss.item()

        totals["data"] += data_time
        totals["h2d"] += h2d_time
        totals["fwd_bwd"] += fwd_bwd_time
        totals["opt"] += opt_time
        totals["step"] += step_time

        interval["data"] += data_time
        interval["h2d"] += h2d_time
        interval["fwd_bwd"] += fwd_bwd_time
        interval["opt"] += opt_time
        interval["step"] += step_time

        step += 1
        batch_size = X_batch.size(0)
        epoch_images += batch_size
        interval_images += batch_size

        if step % log_interval == 0:
            avg_data = interval["data"] / log_interval
            avg_h2d = interval["h2d"] / log_interval
            avg_fwd_bwd = interval["fwd_bwd"] / log_interval
            avg_opt = interval["opt"] / log_interval
            avg_step = interval["step"] / log_interval
            ips = interval_images / max(interval["step"], 1e-9)
            alloc, reserved, max_alloc = get_cuda_memory_mb()
            print(
                f"[epoch {epoch} step {step}] data={avg_data*1000:.2f}ms "
                f"h2d={avg_h2d*1000:.2f}ms fwd+bwd={avg_fwd_bwd*1000:.2f}ms "
                f"opt={avg_opt*1000:.2f}ms step={avg_step*1000:.2f}ms "
                f"ips={ips:.1f} mem={alloc:.0f}MB max={max_alloc:.0f}MB"
            )
            interval = {k: 0.0 for k in interval}
            interval_images = 0

    epoch_time = time.perf_counter() - epoch_start
    avg_loss = epoch_loss / max(step, 1)
    train_accuracy = correct_num / max(n_train, 1)
    epoch_ips = epoch_images / max(epoch_time, 1e-9)
    alloc, reserved, max_alloc = get_cuda_memory_mb()
    print(
        f"[epoch {epoch}] loss={avg_loss:.4f} acc={train_accuracy:.4f} "
        f"time={epoch_time:.2f}s ips={epoch_ips:.1f} "
        f"mem={alloc:.0f}MB max={max_alloc:.0f}MB"
    )
    return avg_loss, train_accuracy, epoch_ips


def main():
    args = parse_args()

    if os.name == "nt":
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Task Manager hint: switch GPU graph to CUDA/Compute_0 for DL usage.")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    torch.backends.cudnn.benchmark = args.cudnn_benchmark and device.type == "cuda"

    train_dataset = CIFAR10TrainDataset(
        images_dir="./dataset/train",
        labels_csv="./dataset/trainLabels.csv",
        class_to_idx=class_to_idx,
    )
    test_dataset = CIFAR10TestDataset(
        images_dir="./dataset/test",
    )

    if args.cache_data:
        train_dataset = CachedDataset(train_dataset)
        test_dataset = CachedDataset(test_dataset)

    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    num_workers = max(args.num_workers, 0)
    pin_memory = args.pin_memory and device.type == "cuda"
    persistent_workers = args.persistent_workers and num_workers > 0
    prefetch_factor = args.prefetch_factor if num_workers > 0 else None

    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.drop_last,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **dl_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **dl_kwargs,
    )

    if args.drop_last:
        n_train = len(train_loader) * args.batch_size
    else:
        n_train = len(train_loader.dataset)

    print(f"train subset size: {len(train_subset)}")
    print(f"val subset size: {len(val_subset)}")

    if args.preview_batch:
        for data, target in train_loader:
            data_cpu = data.cpu()
            target_cpu = target.cpu()
            utils.draw_imgs(data_cpu, [idx_to_class[t.item()] for t in target_cpu])
            break

    def model_factory():
        return ResNet(3, 10)

    if args.batch_sweep and device.type == "cuda":
        batch_size_sweep(model_factory, device, args.batch_sweep_sizes, args.batch_sweep_iters, args.use_amp)
        return

    model = model_factory().to(device)
    if args.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")

    writer = SummaryWriter(f"runs/{run_name}")

    synthetic_ips = None
    if args.synthetic_bench and device.type == "cuda":
        synthetic_ips, alloc, reserved, max_alloc = synthetic_benchmark(
            model_factory, device, args.batch_size, args.synthetic_iters, args.use_amp
        )
        print(
            f"Synthetic benchmark: {synthetic_ips:.1f} img/s | "
            f"mem_alloc={alloc:.0f}MB | mem_reserved={reserved:.0f}MB | max_alloc={max_alloc:.0f}MB"
        )

    for epoch in range(args.epochs):
        avg_loss, train_accuracy, epoch_ips = train_one_epoch(
            model,
            optimizer,
            loss_fn,
            scaler,
            train_loader,
            device,
            n_train,
            epoch,
            args.log_interval,
            args.use_amp,
        )

        model.eval()
        with torch.no_grad():
            X_val, y_val = next(iter(val_loader))
            X_val = X_val.to(device, non_blocking=True)
            y_val = y_val.to(device, non_blocking=True)
            y_val_pred = model(X_val)
            val_correct_num = (torch.argmax(y_val_pred, dim=1) == y_val).sum().item()
            val_accuracy = val_correct_num / max(len(y_val), 1)

        print(
            f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}, "
            f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}"
        )
        writer.add_scalar("train/accuracy", train_accuracy, epoch)
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("val/accuracy", val_accuracy, epoch)

        if synthetic_ips is not None and epoch == 0:
            ratio = epoch_ips / max(synthetic_ips, 1e-9)
            if ratio < 0.7:
                conclusion = "DataLoader bottleneck likely"
            else:
                conclusion = "Compute-bound or balanced"
            print(f"Throughput vs synthetic: {epoch_ips:.1f}/{synthetic_ips:.1f} img/s -> {conclusion}")

    # Test set is large; accumulate results in res_frame
    res_frame = pd.DataFrame(columns=["id", "label"])

    print(f"test dataset size: {len(test_dataset)}")
    model.eval()
    with torch.no_grad():
        for data, img_idx in test_loader:
            data = data.to(device, non_blocking=True)
            y_pred = model(data)
            y = torch.argmax(y_pred, dim=1).cpu()
            y = [idx_to_class[t.item()] for t in y]
            res_frame = pd.concat(
                [res_frame, pd.DataFrame({"id": img_idx, "label": y})],
                ignore_index=True,
            )
            print(f"{res_frame.shape[0]} / {len(test_dataset)}")

    res_frame = res_frame.sort_values("id")
    print(res_frame.shape)
    res_frame.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
