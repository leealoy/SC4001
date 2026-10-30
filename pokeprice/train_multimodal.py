"""Train cohort-matched metadata, image and fusion price regressors.

All models use the standard-layout cards in the controlled image manifest. The
metadata and image preprocessors are fitted on training rows only. Validation is
always predicted; the untouched test split is emitted only with --include-test.

The residual-gated model is trained jointly: its metadata prediction is an
internal branch, not an in-sample prediction from a separately pretrained
model. This avoids residual-target leakage without requiring out-of-fold base
predictions.
"""
import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .evaluation import metrics
except ImportError:
    from evaluation import metrics


MODEL_NAMES = ("metadata", "image", "concat", "gated", "gated_dropout")
CATEGORY_COLUMNS = [
    "pokemon_identity", "rarity", "supertype", "subtypes", "types", "artist",
    "set_id", "series", "regulation_mark",
]
NUMERIC_COLUMNS = [
    "hp", "number_rel", "printed_total", "n_attacks", "n_abilities",
    "retreat_cost", "age_years",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def categorical_values(frame, column):
    return frame[column].astype("string").fillna("NA").astype(str)


def numeric_frame(frame, reference_date):
    values = pd.DataFrame(index=frame.index)
    values["hp"] = pd.to_numeric(frame.hp, errors="coerce")
    number = pd.to_numeric(
        frame.number.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    printed = pd.to_numeric(frame.printed_total, errors="coerce")
    values["number_rel"] = number / printed.replace(0, np.nan)
    values["printed_total"] = printed
    for column in ["n_attacks", "n_abilities", "retreat_cost"]:
        values[column] = pd.to_numeric(frame[column], errors="coerce")
    values["age_years"] = (
        reference_date - pd.to_datetime(frame.release_date)
    ).dt.days / 365.25
    return values[NUMERIC_COLUMNS]


class MetadataEncoder:
    """Training-fitted categorical indices and standardised numeric features."""

    def fit(self, frame, reference_date):
        self.reference_date = pd.Timestamp(reference_date)
        self.category_maps = {}
        for column in CATEGORY_COLUMNS:
            categories = sorted(categorical_values(frame, column).unique())
            self.category_maps[column] = {
                value: index + 1 for index, value in enumerate(categories)}
        numeric = numeric_frame(frame, self.reference_date)
        self.medians = numeric.median().fillna(0.0)
        filled = numeric.fillna(self.medians)
        self.means = filled.mean()
        self.stds = filled.std(ddof=0).replace(0, 1.0).fillna(1.0)
        return self

    @property
    def cardinalities(self):
        # Index zero is reserved for validation/test categories unseen in train.
        return [len(self.category_maps[column]) + 1 for column in CATEGORY_COLUMNS]

    def transform(self, frame):
        categorical = np.column_stack([
            categorical_values(frame, column)
            .map(self.category_maps[column]).fillna(0).to_numpy(dtype=np.int64)
            for column in CATEGORY_COLUMNS
        ])
        numeric = numeric_frame(frame, self.reference_date)
        numeric = ((numeric.fillna(self.medians) - self.means) / self.stds)
        return categorical, numeric.to_numpy(dtype=np.float32)

    def state(self):
        return {
            "category_columns": CATEGORY_COLUMNS,
            "numeric_columns": NUMERIC_COLUMNS,
            "category_maps": self.category_maps,
            "medians": self.medians.to_dict(),
            "means": self.means.to_dict(),
            "stds": self.stds.to_dict(),
            "reference_date": str(self.reference_date.date()),
        }


def load_visual_cohort(args):
    cards = pd.read_csv(
        Path(args.data) / "cards.csv",
        parse_dates=["release_date"],
        dtype={"card_id": "string", "pokemon_identity": "string"},
    )
    manifest = pd.read_csv(
        args.manifest, dtype={"card_id": "string"})
    required = {
        "card_id", "set_id", "layout_family", "image_condition",
        args.split,
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    view = manifest[manifest.image_condition == args.condition].copy()
    if view.empty:
        raise ValueError(f"manifest has no {args.condition} rows")
    if view.card_id.duplicated().any():
        raise ValueError(f"duplicate card IDs for {args.condition}")

    seen_column = (
        "identity_seen_in_temporal_train" if args.split == "split_temporal"
        else "identity_seen_in_random_train")
    splits = pd.read_csv(
        Path(args.data) / "splits.csv",
        usecols=["card_id", args.split, seen_column],
        dtype={"card_id": "string"},
    )
    splits = splits.rename(columns={
        args.split: "recorded_split",
        seen_column: "identity_seen_in_train",
    })
    frame = (view[[
        "card_id", "set_id", "layout_family", args.split,
    ]].merge(cards, on=["card_id", "set_id"], validate="one_to_one")
      .merge(splits, on="card_id", validate="one_to_one"))
    if not frame[args.split].equals(frame.recorded_split):
        mismatch = frame.loc[
            frame[args.split] != frame.recorded_split,
            ["card_id", args.split, "recorded_split"]].head()
        raise ValueError(f"manifest and split table disagree:\n{mismatch}")
    frame = frame.rename(columns={args.split: "split"}).drop(
        columns="recorded_split")
    if frame.price_market.isna().any() or (frame.price_market <= 0).any():
        raise ValueError("visual cohort contains missing/non-positive targets")
    if frame.pokemon_identity.isna().any():
        raise ValueError("visual cohort contains missing Pokémon identities")
    frame["y_true_log"] = np.log1p(frame.price_market.astype(float))
    frame = frame.sort_values("card_id").reset_index(drop=True)
    sizes = frame.split.value_counts().to_dict()
    if any(sizes.get(part, 0) == 0 for part in ["train", "val", "test"]):
        raise ValueError(f"empty partition in visual cohort: {sizes}")
    return frame


def load_embeddings(path, card_ids):
    if not path.exists():
        raise FileNotFoundError(
            f"missing image embeddings: {path}; run extract_image_embeddings.py")
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["card_id"].astype(str)
        embeddings = archive["embedding"]
    if embeddings.ndim != 2 or len(ids) != len(embeddings):
        raise ValueError(f"invalid embedding arrays in {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate card IDs in {path}")
    locations = {card_id: index for index, card_id in enumerate(ids)}
    missing = [card_id for card_id in card_ids if card_id not in locations]
    if missing:
        raise ValueError(f"embeddings lack {len(missing)} cards, e.g. {missing[:5]}")
    ordered = embeddings[[locations[card_id] for card_id in card_ids]].astype(
        np.float32)
    if not np.isfinite(ordered).all():
        raise ValueError("image embeddings contain non-finite values")
    return ordered


class MetadataBackbone(nn.Module):
    def __init__(self, cardinalities, numeric_dimension, hidden, dropout):
        super().__init__()
        dimensions = [
            min(32, max(4, int(round(4 * cardinality ** 0.25))))
            for cardinality in cardinalities
        ]
        self.embeddings = nn.ModuleList([
            # Index zero represents a category unseen during training. Keeping
            # it as a fixed zero vector avoids an untrained random embedding
            # whose value would otherwise change across seeds. This is crucial
            # for temporal evaluation because every validation/test set_id is
            # necessarily unseen.
            nn.Embedding(cardinality, dimension, padding_idx=0)
            for cardinality, dimension in zip(cardinalities, dimensions)
        ])
        input_dimension = sum(dimensions) + numeric_dimension
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, categorical, numeric):
        embedded = [
            layer(categorical[:, index])
            for index, layer in enumerate(self.embeddings)
        ]
        latent = self.network(torch.cat([*embedded, numeric], dim=1))
        return latent, self.regressor(latent).squeeze(1)


class ImageBackbone(nn.Module):
    def __init__(self, input_dimension, hidden, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, image):
        latent = self.network(image)
        return latent, self.regressor(latent).squeeze(1)


class MetadataModel(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata

    def forward(self, categorical, numeric, image):
        return self.metadata(categorical, numeric)[1]


class ImageModel(nn.Module):
    def __init__(self, image):
        super().__init__()
        self.image = image

    def forward(self, categorical, numeric, image):
        return self.image(image)[1]


class ConcatenationModel(nn.Module):
    def __init__(self, metadata, image, hidden, dropout):
        super().__init__()
        self.metadata = metadata
        self.image = image
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, categorical, numeric, image):
        metadata_latent, _ = self.metadata(categorical, numeric)
        image_latent, _ = self.image(image)
        return self.head(torch.cat([metadata_latent, image_latent], dim=1)).squeeze(1)


class ResidualGatedModel(nn.Module):
    def __init__(self, metadata, image, hidden, dropout, modality_dropout):
        super().__init__()
        self.metadata = metadata
        self.image = image
        self.modality_dropout = modality_dropout
        fusion_dimension = hidden * 2 + 1
        self.gate = nn.Sequential(
            nn.Linear(fusion_dimension, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1), nn.Sigmoid(),
        )
        self.residual = nn.Sequential(
            nn.Linear(fusion_dimension, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, categorical, numeric, image):
        metadata_latent, metadata_prediction = self.metadata(categorical, numeric)
        availability = torch.ones(
            (len(image), 1), dtype=image.dtype, device=image.device)
        if self.training and self.modality_dropout > 0:
            availability = (
                torch.rand_like(availability) >= self.modality_dropout).to(image.dtype)
        image_latent, _ = self.image(image * availability)
        features = torch.cat(
            [metadata_latent, image_latent, availability], dim=1)
        correction = self.gate(features) * self.residual(features)
        # A dropped image must reduce exactly to the metadata branch.
        return metadata_prediction + (availability * correction).squeeze(1)


def build_model(name, cardinalities, numeric_dimension, image_dimension, args):
    metadata = MetadataBackbone(
        cardinalities, numeric_dimension, args.hidden_dim, args.dropout)
    image = ImageBackbone(
        image_dimension, args.hidden_dim, args.dropout)
    if name == "metadata":
        return MetadataModel(metadata)
    if name == "image":
        return ImageModel(image)
    if name == "concat":
        return ConcatenationModel(
            metadata, image, args.hidden_dim, args.dropout)
    if name in {"gated", "gated_dropout"}:
        probability = args.modality_dropout if name == "gated_dropout" else 0.0
        return ResidualGatedModel(
            metadata, image, args.hidden_dim, args.dropout, probability)
    raise ValueError(name)


def make_loader(arrays, indices, batch_size, shuffle, seed, device):
    categorical, numeric, image, target = arrays
    dataset = TensorDataset(
        torch.from_numpy(categorical[indices]),
        torch.from_numpy(numeric[indices]),
        torch.from_numpy(image[indices]),
        torch.from_numpy(target[indices]),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=generator if shuffle else None,
        pin_memory=device.type == "cuda")


def predict(model, loader, device):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for categorical, numeric, image, _ in loader:
            values = model(
                categorical.to(device, non_blocking=True),
                numeric.to(device, non_blocking=True),
                image.to(device, non_blocking=True),
            )
            predictions.append(values.cpu().numpy())
    return np.concatenate(predictions)


def train_one(model, arrays, train_indices, val_indices, args, device,
              seed, checkpoint, checkpoint_metadata):
    train_loader = make_loader(
        arrays, train_indices, args.batch_size, True, seed, device)
    val_loader = make_loader(
        arrays, val_indices, args.batch_size, False, seed, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.loss == "huber":
        criterion = nn.HuberLoss(delta=args.huber_delta)
    else:
        criterion = nn.L1Loss()

    model.to(device)
    best_state = None
    best_mae = math.inf
    best_epoch = 0
    stale = 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    y_val = arrays[3][val_indices]

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for categorical, numeric, image, target in train_loader:
            categorical = categorical.to(device, non_blocking=True)
            numeric = numeric.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(categorical, numeric, image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(target)
            count += len(target)
        val_prediction = predict(model, val_loader, device)
        val_mae = float(np.mean(np.abs(y_val - val_prediction)))
        if val_mae < best_mae - args.min_delta:
            best_mae = val_mae
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "state_dict": best_state,
                "best_epoch": best_epoch,
                "best_val_mae": best_mae,
                **checkpoint_metadata,
            }, checkpoint)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale == 0:
            print(
                f"  epoch {epoch:03d} train_loss={loss_sum / count:.4f} "
                f"val_mae={val_mae:.4f} best={best_mae:.4f}")
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    print(f"  selected epoch {best_epoch}, validation MAE {best_mae:.4f}")
    return model


def model_id(name, encoder, condition):
    if name == "metadata":
        return "torch_metadata_identity_metadata"
    labels = {
        "image": "image",
        "concat": "concat",
        "gated": "residual_gated",
        "gated_dropout": "residual_gated_modality_dropout",
    }
    return f"{encoder}_{labels[name]}_{condition}"


def prediction_rows(model, model_name, arrays, frame, indices, evaluation,
                    args, device, seed):
    loader = make_loader(
        arrays, indices, args.batch_size, False, seed, device)
    values = predict(model, loader, device)
    subset = frame.iloc[indices]
    feature_set = "image" if model_name == "image" else "identity_metadata"
    condition = "none" if model_name == "metadata" else args.condition
    return pd.DataFrame({
        "card_id": subset.card_id.to_numpy(),
        "set_id": subset.set_id.to_numpy(),
        "model": model_id(model_name, args.encoder, args.condition),
        "seed": seed,
        "eval": evaluation,
        "y_true_log": subset.y_true_log.to_numpy(),
        "y_pred_log": values,
        "pokemon_identity": subset.pokemon_identity.to_numpy(),
        "identity_seen_in_train": subset.identity_seen_in_train.to_numpy(),
        "feature_set": feature_set,
        "image_condition": condition,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--manifest", default="data/image_views/manifest.csv")
    parser.add_argument("--features", default="data/features")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument(
        "--split", choices=["split_temporal", "split_random"],
        default="split_temporal")
    parser.add_argument(
        "--condition", choices=["full_card", "artwork_only", "layout_only"],
        default="artwork_only")
    parser.add_argument("--encoder", default="resnet50")
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument(
        "--load-checkpoints", action="store_true",
        help="skip training and predict with existing validation-selected checkpoints")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--modality-dropout", type=float, default=0.25)
    parser.add_argument("--loss", choices=["huber", "mae"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.5)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("batch size, epochs and patience must be positive")
    if not 0 <= args.dropout < 1 or not 0 <= args.modality_dropout < 1:
        raise ValueError("dropout probabilities must lie in [0, 1)")

    split_name = args.split.removeprefix("split_")
    output = Path(args.out) if args.out else Path(args.results) / (
        f"multimodal_{split_name}_{args.condition}.csv")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"device: {device}")
    frame = load_visual_cohort(args)
    indices = {
        part: np.flatnonzero(frame.split.to_numpy() == part)
        for part in ["train", "val", "test"]
    }
    print({part: len(value) for part, value in indices.items()})

    reference_date = (Path(args.data) / "download_date.txt").read_text(
        encoding="utf-8").strip()
    metadata_encoder = MetadataEncoder().fit(
        frame.iloc[indices["train"]], reference_date)
    categorical, numeric = metadata_encoder.transform(frame)

    needs_images = any(name != "metadata" for name in args.models)
    if needs_images:
        embedding_path = Path(args.features) / args.encoder / f"{args.condition}.npz"
        image = load_embeddings(embedding_path, frame.card_id.astype(str).tolist())
        image_mean = image[indices["train"]].mean(axis=0, dtype=np.float64).astype(
            np.float32)
        image_std = image[indices["train"]].std(axis=0).astype(np.float32)
        image_std[image_std < 1e-6] = 1.0
        image = ((image - image_mean) / image_std).astype(np.float32)
    else:
        image = np.zeros((len(frame), 1), dtype=np.float32)
        image_mean = np.zeros(1, dtype=np.float32)
        image_std = np.ones(1, dtype=np.float32)
    target = frame.y_true_log.to_numpy(dtype=np.float32)
    arrays = (categorical, numeric, image, target)

    predictions = []
    for name in args.models:
        for seed in args.seeds:
            print(f"{name}, seed {seed}")
            seed_everything(seed)
            model = build_model(
                name, metadata_encoder.cardinalities, numeric.shape[1],
                image.shape[1], args)
            identifier = model_id(name, args.encoder, args.condition)
            checkpoint = (Path(args.checkpoint_dir) / split_name /
                          args.condition / f"{identifier}_seed{seed}.pt")
            checkpoint_metadata = {
                "model": identifier,
                "seed": seed,
                "arguments": vars(args),
                "metadata_encoder": metadata_encoder.state(),
                "image_mean": image_mean,
                "image_std": image_std,
            }
            if args.load_checkpoints:
                if not checkpoint.exists():
                    raise FileNotFoundError(
                        f"missing validation-selected checkpoint: {checkpoint}")
                saved = torch.load(checkpoint, map_location="cpu")
                if saved.get("model") != identifier or saved.get("seed") != seed:
                    raise ValueError(f"checkpoint identity mismatch: {checkpoint}")
                saved_mean = np.asarray(saved["image_mean"], dtype=np.float32)
                saved_std = np.asarray(saved["image_std"], dtype=np.float32)
                if name != "metadata" and not (
                        np.allclose(saved_mean, image_mean) and
                        np.allclose(saved_std, image_std)):
                    raise ValueError(
                        f"training-fitted image scaling differs from {checkpoint}")
                model.load_state_dict(saved["state_dict"])
                model.to(device)
                print(
                    f"  loaded validation checkpoint from epoch "
                    f"{saved['best_epoch']} (MAE {saved['best_val_mae']:.4f})")
            else:
                model = train_one(
                    model, arrays, indices["train"], indices["val"], args,
                    device, seed, checkpoint, checkpoint_metadata)
            evaluations = ["val", "test"] if args.include_test else ["val"]
            for evaluation in evaluations:
                rows = prediction_rows(
                    model, name, arrays, frame, indices[evaluation], evaluation,
                    args, device, seed)
                predictions.append(rows)
                score = metrics(rows.y_true_log, rows.y_pred_log)
                print(f"  {evaluation}: {json.dumps(score, sort_keys=True)}")

    result = pd.concat(predictions, ignore_index=True)
    key = ["card_id", "model", "seed", "eval"]
    if result.duplicated(key).any():
        raise ValueError("duplicate prediction keys were generated")
    result.to_csv(output, index=False)
    print(f"wrote {len(result)} predictions to {output}")


if __name__ == "__main__":
    main()
