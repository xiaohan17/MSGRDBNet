import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from msgrdbnet.data import PairedSRDataset, SyntheticHRDataset
from msgrdbnet.degradation import SRMDDegradation
from msgrdbnet.metrics import psnr, ssim
from msgrdbnet.utils import (
    build_model,
    denormalize,
    extract_model_state,
    init_weights,
    load_yaml,
    normalize,
    save_checkpoint,
    seed_everything,
    unwrap_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MSGRDBNet")
    parser.add_argument("--config", default="configs/synthetic_x4.yaml")
    parser.add_argument("--output", default="outputs/train")
    parser.add_argument("--devices", default="0", help="GPU ids, e.g. 0 or 0,1; use cpu for CPU")
    parser.add_argument("--resume", default=None, help="Resume a checkpoint created by this project")
    parser.add_argument("--pretrained", default=None, help="Load model weights only; supports legacy Lightning checkpoints")
    return parser.parse_args()


def get_device(devices):
    if devices.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu"), []
    ids = [int(x) for x in devices.split(",") if x.strip()]
    if not ids:
        ids = [0]
    torch.cuda.set_device(ids[0])
    return torch.device(f"cuda:{ids[0]}"), ids


def make_dataset(split_cfg, mode, scale, training):
    if mode == "paired":
        return PairedSRDataset(
            lr_dir=split_cfg["lr_dir"],
            hr_dir=split_cfg["hr_dir"],
            scale=scale,
            lr_patch_size=split_cfg.get("lr_patch_size"),
            training=training,
        )
    if mode == "synthetic":
        return SyntheticHRDataset(
            hr_dir=split_cfg["hr_dir"],
            scale=scale,
            hr_patch_size=split_cfg.get("hr_patch_size"),
            training=training,
        )
    raise ValueError("data.mode must be paired or synthetic")


def make_degrader(cfg, scale, key, device):
    if cfg["data"]["mode"] != "synthetic":
        return None
    return SRMDDegradation(scale=scale, **cfg["degradation"][key]).to(device)


def prepare_batch(batch, mode, device, degrader=None, randomize=True):
    hr = batch["hr"].to(device, non_blocking=True)
    if mode == "paired":
        lr = batch["lr"].to(device, non_blocking=True)
    else:
        lr = degrader(hr, randomize=randomize)
    return lr, hr


@torch.no_grad()
def validate(model, loader, mode, device, degrader, mean, std, scale, amp):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    for batch in tqdm(loader, desc="Validation", leave=False):
        lr, hr = prepare_batch(batch, mode, device, degrader, randomize=False)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            sr = model(normalize(lr, mean, std))
        sr = denormalize(sr, mean, std).clamp(0, 1)
        for i in range(sr.shape[0]):
            total_psnr += psnr(sr[i : i + 1], hr[i : i + 1], crop_border=scale)
            total_ssim += ssim(sr[i : i + 1], hr[i : i + 1], crop_border=scale)
            count += 1
    return total_psnr / count, total_ssim / count


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    output_dir = Path(args.output)
    weight_dir = output_dir / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)

    device, gpu_ids = get_device(args.devices)
    model_cfg = cfg["model"]
    scale = int(model_cfg["scale"])
    mode = cfg["data"]["mode"]
    mean = cfg["normalization"]["mean"]
    std = cfg["normalization"]["std"]

    train_ds = make_dataset(cfg["data"]["train"], mode, scale, training=True)
    val_ds = make_dataset(cfg["data"]["val"], mode, scale, training=False)
    workers = int(cfg["data"].get("num_workers", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["data"]["train"].get("batch_size", 4)),
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["data"]["val"].get("batch_size", 1)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(model_cfg)
    init_weights(model)
    model.to(device)

    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(extract_model_state(ckpt), strict=False)
        print(f"Loaded pretrained weights: missing={len(missing)}, unexpected={len(unexpected)}")

    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    optim_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(optim_cfg.get("milestones", [200, 400, 600, 800])),
        gamma=float(optim_cfg.get("gamma", 0.5)),
    )

    start_epoch = 1
    best_psnr = float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        unwrap_model(model).load_state_dict(extract_model_state(checkpoint), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint.get("best_psnr", best_psnr))
        print(f"Resumed from epoch {start_epoch - 1}")

    train_degrader = make_degrader(cfg, scale, "train", device)
    val_degrader = make_degrader(cfg, scale, "val", device)

    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    epochs = int(cfg["training"]["epochs"])
    save_every = int(cfg["training"].get("save_every", 100))
    patience = int(cfg["training"].get("early_stop_patience", 0))
    stale_epochs = 0

    log_path = output_dir / "train_log.csv"
    if not log_path.exists() or start_epoch == 1:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_l1", "val_psnr", "val_ssim", "lr"])

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress:
            lr, hr = prepare_batch(batch, mode, device, train_degrader, randomize=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                sr = model(normalize(lr, mean, std))
                target = normalize(hr, mean, std)
                loss = F.l1_loss(sr, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.5f}")

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)
        val_psnr, val_ssim = validate(
            model, val_loader, mode, device, val_degrader, mean, std, scale, amp
        )
        lr_value = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}: train_l1={train_loss:.6f}, "
            f"val_psnr={val_psnr:.3f}, val_ssim={val_ssim:.5f}, lr={lr_value:.2e}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_psnr, val_ssim, lr_value])

        save_checkpoint(
            weight_dir / "last.pth",
            model,
            model_cfg,
            cfg,
            epoch,
            max(best_psnr, val_psnr),
            optimizer,
            scheduler,
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            stale_epochs = 0
            save_checkpoint(
                weight_dir / "best.pth",
                model,
                model_cfg,
                cfg,
                epoch,
                best_psnr,
                optimizer,
                scheduler,
            )
        else:
            stale_epochs += 1

        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                weight_dir / f"epoch_{epoch:04d}.pth",
                model,
                model_cfg,
                cfg,
                epoch,
                best_psnr,
                optimizer,
                scheduler,
            )

        if patience > 0 and stale_epochs >= patience:
            print(f"Early stopping after {stale_epochs} epochs without PSNR improvement")
            break


if __name__ == "__main__":
    main()
