import argparse
from pathlib import Path

import torch

from msgrdbnet.data import read_image, tensor_to_image, to_tensor, write_image
from msgrdbnet.utils import denormalize, load_model_from_checkpoint, load_yaml, normalize


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image inference with MSGRDBNet")
    parser.add_argument("--input", required=True, help="LR image")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default=None, help="Output SR image path")
    parser.add_argument("--config", default=None, help="Required only for legacy checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=0, help="LR tile size; 0 disables tiled inference")
    parser.add_argument("--overlap", type=int, default=16, help="LR overlap for tiled inference")
    return parser.parse_args()


def forward(model, lr, mean, std):
    return denormalize(model(normalize(lr, mean, std)), mean, std).clamp(0, 1)


@torch.no_grad()
def tiled_forward(model, lr, mean, std, scale, tile_size, overlap):
    if tile_size <= 0:
        return forward(model, lr, mean, std)
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")

    _, c, h, w = lr.shape
    step = tile_size - overlap
    out = torch.zeros((1, c, h * scale, w * scale), device=lr.device, dtype=lr.dtype)
    weight = torch.zeros_like(out)

    ys = list(range(0, max(h - tile_size, 0) + 1, step))
    xs = list(range(0, max(w - tile_size, 0) + 1, step))
    if not ys or ys[-1] != max(h - tile_size, 0):
        ys.append(max(h - tile_size, 0))
    if not xs or xs[-1] != max(w - tile_size, 0):
        xs.append(max(w - tile_size, 0))

    for y in ys:
        for x in xs:
            tile = lr[:, :, y : min(y + tile_size, h), x : min(x + tile_size, w)]
            sr_tile = forward(model, tile, mean, std)
            sy, sx = y * scale, x * scale
            sh, sw = sr_tile.shape[-2:]
            out[:, :, sy : sy + sh, sx : sx + sw] += sr_tile
            weight[:, :, sy : sy + sh, sx : sx + sw] += 1
    return out / weight.clamp_min(1)


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    config = load_yaml(args.config) if args.config else None
    model, checkpoint, _ = load_model_from_checkpoint(args.weights, config=config, device="cpu", strict=True)
    cfg = config or checkpoint.get("train_config")
    if cfg is None:
        raise ValueError("Checkpoint has no embedded configuration; pass --config")

    model = model.to(device).eval()
    mean = cfg["normalization"]["mean"]
    std = cfg["normalization"]["std"]
    scale = int(cfg["model"]["scale"])

    image = read_image(args.input)
    lr = to_tensor(image).unsqueeze(0).to(device)
    sr = tiled_forward(model, lr, mean, std, scale, args.tile_size, args.overlap)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path("outputs/infer") / f"{input_path.stem}_SR.png"
    write_image(output_path, tensor_to_image(sr[0]))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
