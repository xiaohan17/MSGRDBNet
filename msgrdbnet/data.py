from pathlib import Path
import random

import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No images found in: {folder}")
    return files


def _to_hwc3(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3:
        if arr.shape[0] <= 16 and arr.shape[-1] > 16:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[-1] >= 3:
            arr = arr[..., :3]
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return arr


def image_to_float(arr):
    arr = _to_hwc3(arr)
    if np.issubdtype(arr.dtype, np.integer):
        if arr.dtype == np.uint8:
            denom = 255.0
        else:
            vmax = float(np.nanmax(arr))
            denom = 10000.0 if vmax <= 10000.0 else float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max(denom, 1.0)
    else:
        arr = arr.astype(np.float32)
        vmax = float(np.nanmax(arr)) if arr.size else 1.0
        if vmax > 1.0:
            arr = arr / (10000.0 if vmax <= 10000.0 else vmax)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def read_image(path):
    return image_to_float(iio.imread(path))


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image, 0.0, 1.0)
    iio.imwrite(path, np.round(arr * 255.0).astype(np.uint8))


def to_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def tensor_to_image(tensor):
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    return tensor.permute(1, 2, 0).numpy()


def mod_crop(image, scale):
    h, w = image.shape[:2]
    h -= h % scale
    w -= w % scale
    return image[:h, :w]


def augment_pair(lr, hr):
    if random.random() < 0.5:
        lr, hr = np.fliplr(lr), np.fliplr(hr)
    if random.random() < 0.5:
        lr, hr = np.flipud(lr), np.flipud(hr)
    k = random.randint(0, 3)
    if k:
        lr, hr = np.rot90(lr, k), np.rot90(hr, k)
    return np.ascontiguousarray(lr), np.ascontiguousarray(hr)


def augment_single(hr):
    if random.random() < 0.5:
        hr = np.fliplr(hr)
    if random.random() < 0.5:
        hr = np.flipud(hr)
    k = random.randint(0, 3)
    if k:
        hr = np.rot90(hr, k)
    return np.ascontiguousarray(hr)


class PairedSRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, scale=4, lr_patch_size=None, training=False):
        self.scale = int(scale)
        self.lr_patch_size = lr_patch_size
        self.training = training

        lr_map = {p.stem: p for p in list_images(lr_dir)}
        hr_map = {p.stem: p for p in list_images(hr_dir)}
        names = sorted(set(lr_map) & set(hr_map))
        if not names:
            raise RuntimeError("No LR/HR pairs with matching file stems were found")
        self.items = [(lr_map[n], hr_map[n], n) for n in names]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        lr_path, hr_path, name = self.items[index]
        lr = read_image(lr_path)
        hr = read_image(hr_path)

        target_h = min(hr.shape[0], lr.shape[0] * self.scale)
        target_w = min(hr.shape[1], lr.shape[1] * self.scale)
        target_h -= target_h % self.scale
        target_w -= target_w % self.scale
        hr = hr[:target_h, :target_w]
        lr = lr[: target_h // self.scale, : target_w // self.scale]

        if self.training:
            p = int(self.lr_patch_size)
            if lr.shape[0] < p or lr.shape[1] < p:
                raise ValueError(f"{lr_path.name} is smaller than lr_patch_size={p}")
            top = random.randint(0, lr.shape[0] - p)
            left = random.randint(0, lr.shape[1] - p)
            lr = lr[top : top + p, left : left + p]
            s = self.scale
            hr = hr[top * s : (top + p) * s, left * s : (left + p) * s]
            lr, hr = augment_pair(lr, hr)

        return {"lr": to_tensor(lr), "hr": to_tensor(hr), "name": name}


class SyntheticHRDataset(Dataset):
    def __init__(self, hr_dir, scale=4, hr_patch_size=None, training=False):
        self.files = list_images(hr_dir)
        self.scale = int(scale)
        self.hr_patch_size = hr_patch_size
        self.training = training

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        hr = mod_crop(read_image(path), self.scale)

        if self.training:
            p = int(self.hr_patch_size)
            if p % self.scale:
                raise ValueError("hr_patch_size must be divisible by scale")
            if hr.shape[0] < p or hr.shape[1] < p:
                raise ValueError(f"{path.name} is smaller than hr_patch_size={p}")
            top = random.randint(0, hr.shape[0] - p)
            left = random.randint(0, hr.shape[1] - p)
            hr = hr[top : top + p, left : left + p]
            hr = augment_single(hr)

        return {"hr": to_tensor(hr), "name": path.stem}
