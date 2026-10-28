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

eBay cards (data/ebay_cards.csv, if it exists) get the label "ebay" in BOTH splits
and are never used for training, early stopping or tuning.  Cards with no usable
price get the label "none".

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
    cards = cards[["card_id", "set_id", "release_date", "price_market"]].copy()
    ebay = set()
    if (d / "ebay_cards.csv").exists():
        ebay = set(pd.read_csv(d / "ebay_cards.csv").card_id)
    else:
        print("WARNING: data/ebay_cards.csv not found; no eBay hold-out will be created")

    priced = cards.price_market.notna() & (cards.price_market > 0)
    is_ebay = cards.card_id.isin(ebay)

    # ---- temporal split: whole sets by release date -----------------------
    last = cards.loc[priced, "release_date"].max()
    test_start = last - pd.DateOffset(months=args.test_months)
    val_start = test_start - pd.DateOffset(months=args.val_months)
    temporal = np.where(cards.release_date > test_start, "test",
                np.where(cards.release_date > val_start, "val", "train"))
    cards["split_temporal"] = temporal

    # ---- random split with the same sizes ---------------------------------
    rng = np.random.default_rng(args.seed)
    pool = cards.index[priced & ~is_ebay]
    n_test = ((cards.split_temporal == "test") & priced & ~is_ebay).sum()
    n_val = ((cards.split_temporal == "val") & priced & ~is_ebay).sum()
    perm = rng.permutation(pool)
    cards["split_random"] = "train"
    cards.loc[perm[:n_test], "split_random"] = "test"
    cards.loc[perm[n_test:n_test + n_val], "split_random"] = "val"

    # ---- hold-outs and unusable rows override both splits -----------------
    for col in ["split_temporal", "split_random"]:
        cards.loc[is_ebay & priced, col] = "ebay"
        cards.loc[~priced, col] = "none"

    cards[["card_id", "set_id", "release_date", "split_temporal", "split_random"]].to_csv(
        d / "splits.csv", index=False)

    # ---- report ------------------------------------------------------------
    n = priced.sum()
    print(f"priced cards: {n}   eBay hold-out cards: {(is_ebay & priced).sum()}")
    print(f"latest priced release: {last.date()}  val_start>{val_start.date()}  test_start>{test_start.date()}")
    for col in ["split_temporal", "split_random"]:
        vc = cards[col].value_counts()
        print(f"\n{col}")
        print((vc.to_frame("n").assign(frac=lambda x: (x.n / n).round(3))).to_string())

    # ---- leakage checks (fail loudly) --------------------------------------
    tt =cards[cards.split_temporal.isin(["train", "val", "test"]) & priced & ~is_ebay]
    assert (tt.groupby("set_id").split_temporal.nunique() == 1).all(), "a set spans two temporal splits"
    mx = lambda s: tt.loc[tt.split_temporal == s, "release_date"]
    assert mx("train").max() <= mx("val").min() <= mx("val").max() <= mx("test").min(), "temporal order violated"
    for col in ["split_temporal", "split_random"]:
        assert not (cards[col].isin(["train", "val"]) & is_ebay).any(), "eBay card leaked into training"
    print("\nleakage checks passed")


if __name__ == "__main__":
    main()
