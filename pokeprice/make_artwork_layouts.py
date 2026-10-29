"""Assign standard framed Pokemon cards to reusable artwork-layout families.

The shortcut experiment deliberately excludes irregular/full-art mechanics. One
reviewed normalised artwork box is maintained per layout family. Audited
exceptions can optionally be supplied to the view builder as per-card overrides.

Outputs:
    data/artwork_layout_assignments.csv  eligible cards and assigned family
    data/artwork_layout_exclusions.csv   excluded cards with an explicit reason
    data/artwork_layout_audit.csv        seeded examples to inspect per family
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:  # support direct script execution and package imports
    from .path_utils import card_image_filename
except ImportError:
    from path_utils import card_image_filename


SERIES_TO_LAYOUT = {
    "Base": "wotc_base_gym_standard",
    "Gym": "wotc_base_gym_standard",
    "Neo": "wotc_neo_standard",
    "E-Card": "ecard_standard",
    "EX": "ex_era_standard",
    "Diamond & Pearl": "dp_platinum_standard",
    "Platinum": "dp_platinum_standard",
    "HeartGold & SoulSilver": "hgss_standard",
    "Black & White": "bw_standard",
    "XY": "xy_standard",
    "Sun & Moon": "sm_standard",
    "Sword & Shield": "swsh_standard",
    "Scarlet & Violet": "sv_standard",
    "Mega Evolution": "mega_standard",
}

STANDARD_RARITIES = {"Common", "Uncommon", "Rare", "Rare Holo"}
IRREGULAR_SUBTYPES = {
    "ex", "EX", "GX", "V", "VMAX", "VSTAR", "BREAK", "LEGEND",
    "Level-Up", "MEGA", "Prime", "Radiant", "V-UNION", "Eternamax", "Star",
}


def is_pokemon(value):
    return str(value).lower().startswith("pok")


def exclusion_reason(row, image_dir):
    if pd.isna(row.price_market) or float(row.price_market) <= 0:
        return "no_usable_price"
    if not is_pokemon(row.supertype):
        return "not_pokemon"
    if pd.isna(row.pokemon_identity) or not str(row.pokemon_identity).strip():
        return "missing_pokemon_identity"
    if row.series not in SERIES_TO_LAYOUT:
        return "unsupported_series_or_promo_layout"
    if row.rarity not in STANDARD_RARITIES:
        return "nonstandard_or_special_rarity"
    subtypes = set(str(row.subtypes).split("|")) if pd.notna(row.subtypes) else set()
    if subtypes & IRREGULAR_SUBTYPES:
        return "irregular_mechanic_or_full_art_layout"
    filename = (
        row.image_file if pd.notna(row.image_file) and str(row.image_file).strip()
        else card_image_filename(row.card_id))
    if not (image_dir / str(filename)).exists():
        return "missing_local_image"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--audit-per-family", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir, image_dir = Path(args.data), Path(args.images)
    cards = pd.read_csv(data_dir / "cards.csv")
    splits = pd.read_csv(data_dir / "splits.csv")
    frame = cards.merge(
        splits[[
            "card_id", "split_temporal", "split_random",
            "identity_seen_in_temporal_train", "identity_seen_in_random_train",
        ]],
        on="card_id", how="left", validate="one_to_one")
    frame["exclusion_reason"] = [
        exclusion_reason(row, image_dir) for row in frame.itertuples(index=False)]
    frame["layout_family"] = frame.series.map(SERIES_TO_LAYOUT)
    frame["assignment_rule"] = np.where(
        frame.exclusion_reason.eq(""),
        "series+standard_rarity+non_irregular_subtype", "")

    columns = [
        "card_id", "name", "set_id", "set_name", "series", "release_date",
        "rarity", "subtypes", "pokemon_identity", "image_file",
        "split_temporal", "split_random", "identity_seen_in_temporal_train",
        "identity_seen_in_random_train", "layout_family", "assignment_rule",
    ]
    assignments = frame[frame.exclusion_reason.eq("")][columns].copy()
    exclusions = frame[~frame.exclusion_reason.eq("")][
        ["card_id", "name", "set_id", "series", "rarity", "subtypes",
         "exclusion_reason"]].copy()

    rng = np.random.default_rng(args.seed)
    audit_parts = []
    for family, group in assignments.groupby("layout_family", sort=True):
        n = min(args.audit_per_family, len(group))
        chosen = rng.choice(group.index.to_numpy(), size=n, replace=False)
        sample = group.loc[chosen].copy()
        sample["audit_status"] = "pending"
        sample["audit_notes"] = ""
        audit_parts.append(sample)
    audit = pd.concat(audit_parts, ignore_index=True)

    assignments.to_csv(data_dir / "artwork_layout_assignments.csv", index=False)
    exclusions.to_csv(data_dir / "artwork_layout_exclusions.csv", index=False)
    audit.to_csv(data_dir / "artwork_layout_audit.csv", index=False)

    print(f"assigned {len(assignments)} standard-layout cards")
    print(assignments.layout_family.value_counts().sort_index().to_string())
    print(f"excluded {len(exclusions)} cards; reasons:")
    print(exclusions.exclusion_reason.value_counts().to_string())
    print(f"audit sample: {len(audit)} cards ({args.audit_per_family}/family maximum)")


if __name__ == "__main__":
    main()
