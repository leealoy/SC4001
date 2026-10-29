"""Create leakage-safe train/val/test splits and write data/splits.csv.

Two splits are produced for every priced card:

split_temporal  Whole SETS are assigned by release date, so the model is tested on
                sets it never saw and that were released AFTER everything in
                training.  test = sets released in the last --test-months,
                val = the --val-months before that, train = everything earlier.
                This is the MAIN experiment.
split_random    Cards are shuffled at random (set-agnostic) with the same
                train/val/test sizes as the temporal split.  Used only to show how
                much a random split overstates performance.

Cards with no usable TCGplayer price get the label "none".  Live eBay asking
prices are a separate exploratory snapshot and are never used as labels or to
construct these splits.

Usage: python pokeprice/make_splits.py --test-months 12 --val-months 12
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--test-months", type=int, default=12)
    ap.add_argument("--val-months", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = Path(args.data)

    cards = pd.read_csv(d / "cards.csv", parse_dates=["release_date"])
    required = {"card_id", "set_id", "release_date", "price_market", "supertype",
                "pokemon_identity"}
    missing = required - set(cards.columns)
    if missing:
        raise ValueError(
            f"cards.csv is missing {sorted(missing)}; re-run fetch_cards.py so the "
            "identity-aware schema is available")
    cards = cards[list(required)].copy()

    priced = cards.price_market.notna() & (cards.price_market > 0)

    # ---- temporal split: whole sets by release date -----------------------
    last = cards.loc[priced, "release_date"].max()
    test_start = last - pd.DateOffset(months=args.test_months)
    val_start = test_start - pd.DateOffset(months=args.val_months)
    temporal = np.where(cards.release_date > test_start, "test",
                np.where(cards.release_date > val_start, "val", "train"))
    cards["split_temporal"] = temporal

    # ---- random split with the same sizes ---------------------------------
    rng = np.random.default_rng(args.seed)
    pool = cards.index[priced]
    n_test = ((cards.split_temporal == "test") & priced).sum()
    n_val = ((cards.split_temporal == "val") & priced).sum()
    perm = rng.permutation(pool)
    cards["split_random"] = "train"
    cards.loc[perm[:n_test], "split_random"] = "test"
    cards.loc[perm[n_test:n_test + n_val], "split_random"] = "val"

    # ---- unusable rows override both splits -------------------------------
    for col in ["split_temporal", "split_random"]:
        cards.loc[~priced, col] = "none"

    # Whether the Pokemon identity was available to the model during training.
    # This supports a predeclared seen-vs-unseen identity analysis.  Missing
    # identity is left NA rather than silently being treated as unseen.
    for split_col in ["split_temporal", "split_random"]:
        train_identities = set(cards.loc[
            priced & (cards[split_col] == "train") & cards.pokemon_identity.notna(),
            "pokemon_identity"])
        seen_col = f"identity_seen_in_{split_col.removeprefix('split_')}_train"
        cards[seen_col] = cards.pokemon_identity.isin(train_identities).astype("boolean")
        cards.loc[cards.pokemon_identity.isna(), seen_col] = pd.NA

    out_cols = ["card_id", "set_id", "release_date", "supertype", "pokemon_identity",
                "split_temporal", "split_random", "identity_seen_in_temporal_train",
                "identity_seen_in_random_train"]
    cards[out_cols].to_csv(d / "splits.csv", index=False)

    # ---- report ------------------------------------------------------------
    n = priced.sum()
    print(f"priced cards: {n}")
    print(f"latest priced release: {last.date()}  val_start>{val_start.date()}  test_start>{test_start.date()}")
    for col in ["split_temporal", "split_random"]:
        vc = cards[col].value_counts()
        print(f"\n{col}")
        print((vc.to_frame("n").assign(frac=lambda x: (x.n / n).round(3))).to_string())

    # ---- leakage checks (fail loudly) --------------------------------------
    tt = cards[cards.split_temporal.isin(["train", "val", "test"]) & priced]
    assert (tt.groupby("set_id").split_temporal.nunique() == 1).all(), "a set spans two temporal splits"
    mx = lambda s: tt.loc[tt.split_temporal == s, "release_date"]
    assert mx("train").max() <= mx("val").min() <= mx("val").max() <= mx("test").min(), "temporal order violated"
    print("\nleakage checks passed")


if __name__ == "__main__":
    main()
