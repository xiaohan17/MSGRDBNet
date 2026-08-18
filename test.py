import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from msgrdbnet.data import PairedSRDataset, SyntheticHRDataset, tensor_to_image, write_image
from msgrdbnet.degradation import SRMDDegradation
from msgrdbnet.metrics import psnr, ssim
from msgrdbnet.utils import denormalize, load_model_from_checkpoint, load_yaml, normalize


def parse_args():
    parser = argparse.ArgumentParser(description="Test MSGRDBNet")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", default=None, help="Required only for legacy checkpoints")
    parser.add_argument("--output", default="outputs/test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    config = load_yaml(args.config) if args.config else None
    model, checkpoint, _ = load_model_from_checkpoint(args.weights, config=config, device="cpu", strict=True)
    cfg = config or checkpoint.get("train_config")
    if cfg is None:
        raise ValueError("Checkpoint has no embedded configuration; pass --config")

    model = model.to(device).eval()
    scale = int(cfg["model"]["scale"])
    mean = cfg["normalization"]["mean"]
    std = cfg["normalization"]["std"]
    mode = cfg["data"]["mode"]
    test_cfg = cfg["data"]["test"]

    if mode == "paired":
        dataset = PairedSRDataset(test_cfg["lr_dir"], test_cfg["hr_dir"], scale=scale, training=False)
        degrader = None
    elif mode == "synthetic":
        dataset = SyntheticHRDataset(test_cfg["hr_dir"], scale=scale, training=False)
        degrader = SRMDDegradation(scale=scale, **cfg["degradation"]["test"]).to(device)
    else:
        raise ValueError("data.mode must be paired or synthetic")

    loader = DataLoader(
        dataset,
        batch_size=int(test_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    output = Path(args.output)
    sr_dir = output / "sr"
    output.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        sr_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            hr = batch["hr"].to(device, non_blocking=True)
            lr = batch["lr"].to(device, non_blocking=True) if mode == "paired" else degrader(hr, randomize=False)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            sr = model(normalize(lr, mean, std))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            sr = denormalize(sr, mean, std).clamp(0, 1)

            names = batch["name"]
            for i in range(sr.shape[0]):
                row = {
                    "name": names[i],
                    "psnr": psnr(sr[i : i + 1], hr[i : i + 1], crop_border=scale),
                    "ssim": ssim(sr[i : i + 1], hr[i : i + 1], crop_border=scale),
                    "runtime_s": elapsed / sr.shape[0],
                }
                rows.append(row)
                if args.save_images:
                    write_image(sr_dir / f"{names[i]}_SR.png", tensor_to_image(sr[i]))

    mean_row = {
        "name": "mean",
        "psnr": sum(r["psnr"] for r in rows) / len(rows),
        "ssim": sum(r["ssim"] for r in rows) / len(rows),
        "runtime_s": sum(r["runtime_s"] for r in rows) / len(rows),
    }
    rows.append(mean_row)

    csv_path = output / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "psnr", "ssim", "runtime_s"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"PSNR={mean_row['psnr']:.3f}, SSIM={mean_row['ssim']:.5f}, "
        f"runtime={mean_row['runtime_s']:.4f}s/image"
    )
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
