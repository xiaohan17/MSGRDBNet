import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .model import MSGRDBNet


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(x, mean, std):
    mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = torch.as_tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean) / std


def denormalize(x, mean, std):
    mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = torch.as_tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return x * std + mean


def init_weights(model):
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _strip_prefix(state_dict, prefix):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    state = _strip_prefix(state, "module.")
    state = _strip_prefix(state, "model.")
    return state


def build_model(model_cfg):
    return MSGRDBNet(**model_cfg)


def load_model_from_checkpoint(weight_path, config=None, device="cpu", strict=True):
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    if model_cfg is None:
        if config is None:
            raise ValueError("Legacy checkpoint requires --config to construct MSGRDBNet")
        model_cfg = config["model"]
    model = build_model(model_cfg)
    state = extract_model_state(checkpoint)
    incompatible = model.load_state_dict(state, strict=strict)
    return model, checkpoint, incompatible


def save_checkpoint(path, model, model_cfg, train_cfg, epoch, best_psnr, optimizer, scheduler):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "project": "MSGRDBNet",
            "epoch": epoch,
            "best_psnr": best_psnr,
            "model_config": model_cfg,
            "train_config": train_cfg,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )
