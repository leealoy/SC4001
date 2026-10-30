"""Extract frozen ImageNet ResNet embeddings for controlled card-image views.

The script reads data/image_views/manifest.csv rather than scanning folders, so
all conditions contain the same card IDs in a deterministic order. Full-card
and layout-only images are padded to a square before resizing; this preserves
the entire card instead of removing its top and bottom with a centre crop.

Example:
    python pokeprice/extract_image_embeddings.py --encoder resnet50
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CONDITIONS = ("full_card", "artwork_only", "layout_only")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


class SquarePad:
    """Pad a PIL image to a square without discarding card content."""

    def __init__(self, fill=(127, 127, 127)):
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        return ImageOps.expand(
            image,
            border=(left, top, side - width - left, side - height - top),
            fill=self.fill,
        )


class ManifestImages(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        path = Path(self.frame.at[index, "image_path"])
        if not path.exists():
            raise FileNotFoundError(f"manifest image does not exist: {path}")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        return self.transform(image), index


def build_encoder(name, weights_name):
    if weights_name != "DEFAULT":
        raise ValueError("only pretrained ImageNet DEFAULT weights are supported")
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(name)
    dimension = model.fc.in_features
    model.fc = nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, dimension, weights


def image_transform(size):
    return transforms.Compose([
        SquarePad(),
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def validate_manifest(frame, conditions):
    required = {"card_id", "set_id", "image_condition", "image_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    unknown = set(conditions) - set(frame.image_condition.unique())
    if unknown:
        raise ValueError(f"manifest lacks image conditions: {sorted(unknown)}")
    for condition in conditions:
        subset = frame[frame.image_condition == condition]
        if subset.card_id.duplicated().any():
            raise ValueError(f"duplicate card IDs in {condition}")
    expected = set(frame.loc[
        frame.image_condition == conditions[0], "card_id"])
    for condition in conditions[1:]:
        observed = set(frame.loc[frame.image_condition == condition, "card_id"])
        if observed != expected:
            raise ValueError(
                f"{condition} does not contain the same cards as {conditions[0]}")


def extract_condition(model, dimension, frame, transform, args, device):
    dataset = ManifestImages(frame, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    output_dtype = np.float16 if args.dtype == "float16" else np.float32
    embeddings = np.empty((len(dataset), dimension), dtype=output_dtype)
    complete = 0
    with torch.inference_mode():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            values = model(images).detach().cpu().numpy().astype(
                output_dtype, copy=False)
            embeddings[indices.numpy()] = values
            complete += len(indices)
            if complete % (args.batch_size * 20) < args.batch_size:
                print(f"  embedded {complete}/{len(dataset)}")
    if not np.isfinite(embeddings).all():
        raise ValueError("encoder produced non-finite embeddings")
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/image_views/manifest.csv")
    parser.add_argument("--out", default="data/features")
    parser.add_argument(
        "--encoder", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument("--weights", choices=["DEFAULT"], default="DEFAULT")
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.image_size <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("image size and batch size must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")

    manifest = pd.read_csv(args.manifest, dtype={"card_id": "string"})
    validate_manifest(manifest, args.conditions)
    model, dimension, weights = build_encoder(args.encoder, args.weights)
    model.to(device)
    transform = image_transform(args.image_size)

    output_dir = Path(args.out) / args.encoder
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "encoder": args.encoder,
        "weights": weights.__class__.__name__ + ".DEFAULT",
        "embedding_dimension": dimension,
        "image_size": args.image_size,
        "preprocessing": "square-pad, resize, ImageNet normalisation",
        "manifest": str(args.manifest),
        "dtype": args.dtype,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    for condition in args.conditions:
        output = output_dir / f"{condition}.npz"
        if output.exists() and not args.overwrite:
            print(f"skipping existing {output}; use --overwrite to replace it")
            continue
        subset = (manifest[manifest.image_condition == condition]
                  .sort_values("card_id").reset_index(drop=True))
        print(f"{condition}: {len(subset)} images")
        embeddings = extract_condition(
            model, dimension, subset, transform, args, device)
        np.savez(
            output,
            card_id=subset.card_id.astype(str).to_numpy(dtype=str),
            embedding=embeddings,
        )
        print(f"wrote {output} {embeddings.shape} {embeddings.dtype}")


if __name__ == "__main__":
    main()
