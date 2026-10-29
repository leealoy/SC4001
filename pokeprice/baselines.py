"""Identity-aware metadata baselines for Pokemon card price prediction.

Primary cohort: Pokemon cards with a National Pokedex identity.
Target: log1p(TCGplayer market price in USD), standard printing only.

Feature sets make the central controlled comparison explicit:
    identity           Pokemon identity only
    metadata           structured metadata, excluding identity
    identity_metadata  identity plus the same structured metadata

The temporal split is the main experiment. All preprocessing is fitted on its
training rows only. Live eBay asking prices are not labels and are not scored.

Usage:
    python pokeprice/baselines.py --split split_temporal
    python pokeprice/baselines.py --split split_random
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

try:  # support both `python pokeprice/baselines.py` and package imports
    from .evaluation import metrics
except ImportError:
    from evaluation import metrics

SEEDS = [0, 1, 2]

IDENTITY_COL = "model_identity"
METADATA_CAT_COLS = [
    "rarity", "supertype", "subtypes", "types", "artist", "set_id", "series",
    "regulation_mark",
]
METADATA_NUM_COLS = [
    "hp", "number_rel", "printed_total", "n_attacks", "n_abilities",
    "retreat_cost", "age_years",
]
FEATURE_SETS = {
    "identity": ([IDENTITY_COL], []),
    "metadata": (METADATA_CAT_COLS, METADATA_NUM_COLS),
    "identity_metadata": ([IDENTITY_COL, *METADATA_CAT_COLS], METADATA_NUM_COLS),
}


def is_pokemon(series):
    """Handle both correctly decoded and legacy mojibake supertype strings."""
    return series.fillna("").astype(str).str.lower().str.startswith("pok")


def validate_identity_schema(df):
    required = {"national_pokedex_numbers", "pokemon_identity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"cards.csv is missing {sorted(missing)}; re-run fetch_cards.py to "
            "collect National Pokedex identities before running these baselines")


def make_features(df, ref_date):
    """Turn cards.csv rows into unfitted model features."""
    X = pd.DataFrame(index=df.index)

    # The optional all-card sensitivity cohort gives non-Pokemon cards a
    # namespaced subject identity rather than a fictitious Pokedex number.
    pokemon = is_pokemon(df.supertype)
    X[IDENTITY_COL] = (
        "nonpokemon:" + df.supertype.fillna("NA").astype(str)
        + ":" + df.name.fillna("NA").astype(str))
    X.loc[pokemon, IDENTITY_COL] = (
        "dex:" + df.loc[pokemon, "pokemon_identity"].astype(str))

    for col in METADATA_CAT_COLS:
        X[col] = df[col].fillna("NA").astype(str)
    X["hp"] = pd.to_numeric(df["hp"], errors="coerce")
    number = pd.to_numeric(
        df["number"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    X["number_rel"] = number / df["printed_total"].replace(0, np.nan)
    for col in ["printed_total", "n_attacks", "n_abilities", "retreat_cost"]:
        X[col] = pd.to_numeric(df[col], errors="coerce")
    X["age_years"] = (
        ref_date - pd.to_datetime(df["release_date"])
    ).dt.days / 365.25
    return X


class GroupMedian:
    """Training-only group median with a global fallback for unseen groups."""

    def __init__(self, column):
        self.column = column

    def fit(self, X, y):
        values = pd.Series(np.asarray(y), index=X.index)
        self.median_by_group = values.groupby(X[self.column].values).median()
        self.global_median = float(np.median(y))
        return self

    def predict(self, X):
        return X[self.column].map(
            self.median_by_group).fillna(self.global_median).to_numpy()


def build_model(name, feature_set, seed):
    cat_cols, num_cols = FEATURE_SETS[feature_set]
    transformers = []
    if cat_cols:
        if name == "hgb":
            # TargetEncoder uses internal cross-fitting in fit_transform, so a
            # row is not encoded using its own target. Unknown test categories
            # fall back to the training target mean.
            cat = TargetEncoder(
                target_type="continuous", cv=5, shuffle=True,
                random_state=seed)
            transformers.append(("cat", cat, cat_cols))
        else:
            # Keep every training identity distinct. Grouping rare identities
            # would weaken the claimed identity control. Rare non-identity
            # metadata values may still share an infrequent bucket.
            if IDENTITY_COL in cat_cols:
                transformers.append((
                    "identity", OneHotEncoder(handle_unknown="ignore"),
                    [IDENTITY_COL]))
            metadata_cat = [col for col in cat_cols if col != IDENTITY_COL]
            if metadata_cat:
                transformers.append((
                    "cat", OneHotEncoder(
                        handle_unknown="ignore", min_frequency=5),
                    metadata_cat))
    if num_cols:
        if name == "hgb":
            num = "passthrough"  # HistGradientBoosting handles NaN natively.
        else:
            from sklearn.impute import SimpleImputer
            num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        transformers.append(("num", num, num_cols))

    preprocessor = ColumnTransformer(transformers)
    if name == "hgb":
        regressor = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20, random_state=seed)
    elif name == "mlp":
        regressor = MLPRegressor(
            hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15, random_state=seed)
    else:
        raise ValueError(name)
    return make_pipeline(preprocessor, regressor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument(
        "--split", choices=["split_temporal", "split_random"], required=True)
    parser.add_argument(
        "--cohort", choices=["pokemon", "all"], default="pokemon",
        help="Pokemon-only is primary; all cards is a sensitivity analysis")
    parser.add_argument(
        "--feature-sets", nargs="+", choices=list(FEATURE_SETS),
        default=list(FEATURE_SETS))
    parser.add_argument(
        "--ref-date", default=None, help="YYYY-MM-DD price snapshot date")
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(exist_ok=True)
    ref = args.ref_date or (data_dir / "download_date.txt").read_text().strip()
    ref_date = pd.Timestamp(ref)

    cards = pd.read_csv(data_dir / "cards.csv", parse_dates=["release_date"])
    validate_identity_schema(cards)
    seen_col = f"identity_seen_in_{args.split.removeprefix('split_')}_train"
    split_cols = ["card_id", args.split, seen_col]
    splits = pd.read_csv(data_dir / "splits.csv", usecols=split_cols).rename(
        columns={args.split: "split", seen_col: "identity_seen_in_train"})
    df = cards.merge(splits, on="card_id")
    df = df[df.split != "none"].copy()
    unresolved = is_pokemon(df.supertype) & df.pokemon_identity.isna()
    if unresolved.any():
        print(
            f"excluding {unresolved.sum()} Pokemon cards without a National "
            "Pokedex identity; report this exclusion")
        df = df[~unresolved].copy()
    if args.cohort == "pokemon":
        df = df[is_pokemon(df.supertype)].copy()
    df = df.reset_index(drop=True)
    df["y"] = np.log1p(df.price_market)
    X = make_features(df, ref_date)

    indices = {
        part: df.index[df.split == part] for part in ["train", "val", "test"]}
    sizes = {key: len(value) for key, value in indices.items()}
    if any(size == 0 for size in sizes.values()):
        raise ValueError(f"empty split after cohort filtering: {sizes}")
    print(sizes, "ref_date", ref)
    y_train = df.loc[indices["train"], "y"].to_numpy()

    specs = []
    if "metadata" in args.feature_sets:
        specs.append(("rarity_median", "metadata", GroupMedian("rarity"), [0]))
    if "identity" in args.feature_sets:
        specs.append(
            ("identity_median", "identity", GroupMedian(IDENTITY_COL), [0]))
    for feature_set in args.feature_sets:
        for family in ["hgb", "mlp"]:
            specs.append((family, feature_set, None, SEEDS))

    rows, predictions = [], []
    for family, feature_set, fixed_model, seeds in specs:
        cat_cols, num_cols = FEATURE_SETS[feature_set]
        feature_cols = [*cat_cols, *num_cols]
        for seed in seeds:
            model = fixed_model or build_model(family, feature_set, seed)
            model.fit(X.loc[indices["train"], feature_cols], y_train)
            model_id = (
                family if family.endswith("_median")
                else f"{family}_{feature_set}")
            for evaluation in ["val", "test"]:
                ii = indices[evaluation]
                pred = model.predict(X.loc[ii, feature_cols])
                score = metrics(df.loc[ii, "y"].to_numpy(), pred)
                rows.append(dict(
                    model=model_id, family=family, feature_set=feature_set,
                    seed=seed, eval=evaluation, n=len(ii), **score))
                predictions.append(pd.DataFrame({
                    "card_id": df.loc[ii, "card_id"].to_numpy(),
                    "set_id": df.loc[ii, "set_id"].to_numpy(),
                    "pokemon_identity": df.loc[ii, "pokemon_identity"].to_numpy(),
                    "identity_seen_in_train": df.loc[
                        ii, "identity_seen_in_train"].to_numpy(),
                    "model": model_id,
                    "family": family,
                    "feature_set": feature_set,
                    "seed": seed,
                    "eval": evaluation,
                    "y_true_log": df.loc[ii, "y"].to_numpy(),
                    "y_pred_log": pred,
                }))
            print(f"done {model_id} seed {seed}")

    result = pd.DataFrame(rows)
    pred_df = pd.concat(predictions, ignore_index=True)
    stem = f"{args.split}_{args.cohort}"
    result.to_csv(out_dir / f"baseline_runs_{stem}.csv", index=False)
    pred_df.to_csv(out_dir / f"baseline_preds_{stem}.csv", index=False)
    metric_cols = ["mae_log", "spearman", "medape", "within_2x"]
    summary = result.groupby(["eval", "model"])[metric_cols].agg(
        ["mean", "std"]).round(3)
    summary.to_csv(out_dir / f"baseline_summary_{stem}.csv")
    print("\n" + summary.to_string())


if __name__ == "__main__":
    main()
