"""Build matched image views from reviewed, reusable layout templates.

Run make_artwork_layouts.py first. Each standard framed card is assigned to a
generation/layout family. One reviewed normalised artwork box is then reused for
that family. A separate override CSV can optionally supply audited per-card
exceptions; no override file is required by default.

Views:
    full_card      complete official scan
    artwork_only   template/override artwork box resized to a common square
    layout_only    complete card with that artwork box replaced by gray
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

try:  # support direct script execution and package imports
    from .path_utils import card_image_filename
except ImportError:
    from path_utils import card_image_filename

TEMPLATE_COLUMNS = {
    "layout_family", "x0", "y0", "x1", "y1", "box_source", "reviewed",
    "example_card_id",
}
ASSIGNMENT_COLUMNS = {"card_id", "set_id", "layout_family", "image_file"}
OVERRIDE_COLUMNS = {
    "card_id", "x0", "y0", "x1", "y1", "box_source", "reviewed",
}


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate_coordinates(values, label):
    try:
        values = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"missing/non-numeric artwork box for {label}") from error
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"invalid normalised box for {label}: {values}")
    return values


def pixel_box(normalised, width, height):
    x0, y0, x1, y1 = normalised
    return (
        round(x0 * width), round(y0 * height),
        round(x1 * width), round(y1 * height),
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(image, size=8):
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = []
    for row in range(size):
        offset = row * (size + 1)
        bits.extend(
            pixels[offset + col] > pixels[offset + col + 1]
            for col in range(size))
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:0{size * size // 4}x}"


def describe_view(image, path):
    return dict(
        image_path=path.as_posix(), width=image.width, height=image.height,
        sha256=file_sha256(path), difference_hash=difference_hash(image),
    )


def save_view(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", optimize=True)
    return describe_view(image, path)


def load_csv(path, required, label):
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--images", default="data/images")
    parser.add_argument(
        "--templates", default="data/artwork_layout_templates.csv")
    parser.add_argument(
        "--assignments", default="data/artwork_layout_assignments.csv")
    parser.add_argument(
        "--overrides", default=None,
        help="optional CSV of audited per-card corrections to family templates")
    parser.add_argument("--out", default="data/image_views")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument(
        "--families", nargs="+", default=None,
        help="only process these layout families")
    parser.add_argument(
        "--card-list", default=None,
        help="optional CSV with card_id values, e.g. artwork_layout_audit.csv")
    parser.add_argument(
        "--limit-per-family", type=int, default=None,
        help="pilot only: process the first N assigned cards per family")
    parser.add_argument(
        "--require-all-templates", action="store_true",
        help="fail if any assigned family lacks a reviewed template")
    args = parser.parse_args()

    image_dir, out_dir = Path(args.images), Path(args.out)
    templates = load_csv(
        Path(args.templates), TEMPLATE_COLUMNS, "layout template file")
    assignments = load_csv(
        Path(args.assignments), ASSIGNMENT_COLUMNS, "layout assignment file")
    if templates.layout_family.duplicated().any():
        raise ValueError("layout template file contains duplicate families")
    if assignments.card_id.duplicated().any():
        raise ValueError("layout assignment file contains duplicate card IDs")
    if args.families:
        unknown = set(args.families) - set(templates.layout_family)
        if unknown:
            raise ValueError(f"unknown requested layout families: {sorted(unknown)}")
        assignments = assignments[
            assignments.layout_family.isin(args.families)].copy()
    if args.card_list:
        card_list = load_csv(Path(args.card_list), {"card_id"}, "card list")
        unknown = set(card_list.card_id) - set(assignments.card_id)
        if unknown:
            raise ValueError(
                f"card list contains ineligible/unassigned IDs: {sorted(unknown)[:10]}")
        assignments = assignments[
            assignments.card_id.isin(card_list.card_id)].copy()

    reviewed = templates.reviewed.map(truthy)
    unreviewed = sorted(
        set(assignments.layout_family) - set(templates.loc[reviewed, "layout_family"]))
    if unreviewed and args.require_all_templates:
        raise ValueError(f"assigned families lack reviewed templates: {unreviewed}")
    if unreviewed:
        print("skipping unreviewed layout families: " + ", ".join(unreviewed))
    templates = templates[reviewed].copy()
    frame = assignments.merge(
        templates, on="layout_family", how="inner", validate="many_to_one")
    if frame.empty:
        raise ValueError("no assigned cards have a reviewed layout template")

    if args.limit_per_family is not None:
        if args.limit_per_family <= 0:
            raise ValueError("--limit-per-family must be positive")
        frame = (frame.sort_values("card_id").groupby(
            "layout_family", group_keys=False).head(args.limit_per_family))

    if args.overrides:
        override_path = Path(args.overrides)
        overrides = load_csv(override_path, OVERRIDE_COLUMNS, "override file")
        if overrides.card_id.duplicated().any():
            raise ValueError("override file contains duplicate card IDs")
        unknown = set(overrides.card_id) - set(assignments.card_id)
        if unknown:
            raise ValueError(
                f"overrides contain cards outside the eligible assignments: "
                f"{sorted(unknown)[:10]}")
        overrides = overrides.rename(columns={
            name: f"override_{name}" for name in
            ["x0", "y0", "x1", "y1", "box_source", "reviewed"]})
        frame = frame.merge(
            overrides, on="card_id", how="left", validate="one_to_one")

    records = []
    for row in frame.itertuples(index=False):
        has_override = (
            hasattr(row, "override_x0") and pd.notna(row.override_x0))
        if has_override:
            if not truthy(row.override_reviewed):
                raise ValueError(f"override for {row.card_id} is not reviewed")
            normalised = validate_coordinates(
                [row.override_x0, row.override_y0, row.override_x1, row.override_y1],
                f"override {row.card_id}")
            box_source = f"card_override:{row.override_box_source}"
            box_level = "card_override"
        else:
            normalised = validate_coordinates(
                [row.x0, row.y0, row.x1, row.y1], row.layout_family)
            box_source = f"layout_template:{row.box_source}"
            box_level = "layout_template"

        filename = (
            str(row.image_file) if pd.notna(row.image_file)
            else card_image_filename(row.card_id))
        source = image_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"missing official image: {source}")
        with Image.open(source) as opened:
            full = opened.convert("RGB")
        box = pixel_box(normalised, *full.size)
        output_name = card_image_filename(row.card_id)

        artwork = ImageOps.fit(
            full.crop(box), (args.size, args.size),
            method=Image.Resampling.LANCZOS)
        layout = full.copy()
        ImageDraw.Draw(layout).rectangle(box, fill=(127, 127, 127))
        views = [
            # The full-card condition references the existing official image;
            # copying every high-resolution source would waste substantial disk.
            ("full_card", source, full, False),
            ("artwork_only", out_dir / "artwork_only" / output_name, artwork, True),
            ("layout_only", out_dir / "layout_only" / output_name, layout, True),
        ]

        for condition, path, image, materialise in views:
            record = dict(
                card_id=row.card_id, set_id=row.set_id,
                layout_family=row.layout_family, image_condition=condition,
                source_path=source.as_posix(), box_source=box_source,
                box_level=box_level, template_example_card_id=row.example_card_id,
                box_reviewed=True, x0=normalised[0], y0=normalised[1],
                x1=normalised[2], y1=normalised[3],
            )
            for split_col in ["split_temporal", "split_random"]:
                if hasattr(row, split_col):
                    record[split_col] = getattr(row, split_col)
            record.update(
                save_view(image, path) if materialise
                else describe_view(image, path))
            records.append(record)

    manifest = pd.DataFrame(records)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(
        f"wrote {len(manifest)} matched views for "
        f"{manifest.card_id.nunique()} cards across "
        f"{manifest.layout_family.nunique()} reviewed families")


if __name__ == "__main__":
    main()
