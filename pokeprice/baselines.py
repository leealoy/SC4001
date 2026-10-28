"""Metadata-only baselines for Pokemon card price prediction.

Target : y = log1p(TCGplayer market price in USD), standard printing only.
Models : rarity_median  (median log-price of training cards with the same rarity)
         hgb            (HistGradientBoostingRegressor)
         mlp            (sklearn MLPRegressor on standardised / one-hot features)
Features (metadata only, no images): rarity, supertype, subtypes, energy type, HP,
         artist, set id, series, card-number position in the set, printed set
         size, attacks/abilities counts, retreat cost, set age.

Everything (encoders, scalers, medians) is FITTED ON TRAIN ROWS ONLY.  Unseen
categories (e.g. every set in the temporal test split) map to an "unknown" code,
which is exactly what a future set looks like to the model.
Val is for model selection; the test split and eBay hold-out are only scored,
never used to choose settings.  Each model is run with 3 seeds.

Usage:
    python pokeprice/baselines.py --split split_temporal
    python pokeprice/baselines.py --split split_random
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEEDS = [0, 1, 2]
# Date the prices were downloaded; read from data/download_date.txt (written by
# fetch_cards.py).  Override with --ref-date if needed.
REF_DATE = None

CAT_COLS = ["rarity", "supertype", "subtypes", "types", "artist", "set_id", "series",
            "regulation_mark"]
NUM_COLS = ["hp", "number_rel", "printed_total", "n_attacks", "n_abilities",
            "retreat_cost", "age_years"]


def make_features(df, ref_date):
    """Turn raw cards.csv rows into model features (no fitting happens here)."""
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = df[c].fillna("NA").astype(str)
    X["hp"] = pd.to_numeric(df["hp"], errors="coerce")
    # card number relative to the set size; secret rares have number > printed_total
    num = pd.to_numeric(df["number"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    X["number_rel"] = num / df["printed_total"].replace(0, np.nan)
    for c in ["printed_total", "n_attacks", "n_abilities", "retreat_cost"]:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X["age_years"] = (ref_date - pd.to_datetime(df["release_date"])).dt.days / 365.25
    return X


def metrics(y_true_log, y_pred_log):
    """MAE on log price, Spearman, median abs % error (USD), share within 2x."""
    t, p = np.expm1(y_true_log), np.expm1(np.clip(y_pred_log, 0, None))
    return dict(
        mae_log=float(np.mean(np.abs(y_true_log - y_pred_log))),
        spearman=float(spearmanr(y_true_log, y_pred_log).correlation),
        medape=float(np.median(np.abs(p - t) / t)),
        within_2x=float(np.mean((p / t >= 0.5) & (p / t <= 2.0))),
    )


class RarityMedian:
    def fit(self, X, y):
        self.med = pd.Series(y, index=X.index).groupby(X["rarity"].values).median()
        self.glob = float(np.median(y))
        return self

    def predict(self, X):
        return X["rarity"].map(self.med).fillna(self.glob).to_numpy()


def build_model(name, seed):
    if name == "hgb":
        pre = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   min_frequency=5, encoded_missing_value=-1), CAT_COLS),
            ("num", "passthrough", NUM_COLS)])
        cat_mask = [True] * len(CAT_COLS) + [False] * len(NUM_COLS)
        reg = HistGradientBoostingRegressor(
            categorical_features=cat_mask, max_iter=500, learning_rate=0.05,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
            random_state=seed)
        return make_pipeline(pre, reg)
    if name == "mlp":
        from sklearn.impute import SimpleImputer
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CAT_COLS),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM_COLS)])
        reg = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300,
                           early_stopping=True, validation_fraction=0.1,
                           n_iter_no_change=15, random_state=seed)
        return make_pipeline(pre, reg)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", choices=["split_temporal", "split_random"], required=True)
    ap.add_argument("--ref-date", default=None, help="YYYY-MM-DD price download date")
    args = ap.parse_args()
    d, out = Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True)

    ref = args.ref_date or (d / "download_date.txt").read_text().strip()
    ref_date = pd.Timestamp(ref)

    cards = pd.read_csv(d / "cards.csv", parse_dates=["release_date"])
    sp = pd.read_csv(d / "splits.csv")[["card_id", args.split]].rename(columns={args.split: "split"})
    df = cards.merge(sp, on="card_id")
    df = df[df.split != "none"].reset_index(drop=True)
    df["y"] = np.log1p(df.price_market)
    X = make_features(df, ref_date)

    # eBay hold-out target = median sold price per card (separate real-world test)
    ebay_y = None
    if (d / "ebay_cards.csv").exists():
        e = pd.read_csv(d / "ebay_cards.csv").set_index("card_id").ebay_median_usd
        ebay_y = np.log1p(df.card_id.map(e))

    idx = {s: df.index[df.split == s] for s in ["train", "val", "test", "ebay"]}
    print({k: len(v) for k, v in idx.items()}, "ref_date", ref)
    ytr = df.loc[idx["train"], "y"].to_numpy()

    eval_sets = {"val": (idx["val"], df.y), "test": (idx["test"], df.y)}
    if ebay_y is not None and len(idx["ebay"]):
        eval_sets["ebay"] = (idx["ebay"], ebay_y)

    rows, preds = [], []
    for name in ["rarity_median", "hgb", "mlp"]:
        for seed in (SEEDS if name != "rarity_median" else [0]):
            m = RarityMedian() if name == "rarity_median" else build_model(name, seed)
            m.fit(X.loc[idx["train"]], ytr)
            for ev, (ii, yy) in eval_sets.items():
                p = m.predict(X.loc[ii])
                r = metrics(yy.loc[ii].to_numpy(), p)
                rows.append(dict(model=name, seed=seed, eval=ev, n=len(ii), **r))
                preds.append(pd.DataFrame(dict(card_id=df.loc[ii, "card_id"].values, model=name,
                                               seed=seed, eval=ev, y_true_log=yy.loc[ii].values,
                                               y_pred_log=p)))
            print(f"done {name} seed {seed}")

    res = pd.DataFrame(rows)
    res.to_csv(out / f"baseline_runs_{args.split}.csv", index=False)
    pd.concat(preds).to_csv(out / f"baseline_preds_{args.split}.csv", index=False)
    cols = ["mae_log", "spearman", "medape", "within_2x"]
    summ = res.groupby(["eval", "model"])[cols].agg(["mean", "std"]).round(3)
    summ.to_csv(out / f"baseline_summary_{args.split}.csv")
    print("\n" + summ.to_string())


if __name__ == "__main__":
    main()
