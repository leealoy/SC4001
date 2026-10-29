"""Evaluate prediction CSVs from metadata, image and fusion models uniformly.

Required columns:
    card_id, set_id, model, seed, eval, y_true_log, y_pred_log

Recommended columns:
    pokemon_identity, identity_seen_in_train, image_condition, feature_set

The optional paired comparison averages each model's seeds per card, aligns the
same cards with a reference model, and bootstraps whole sets rather than treating
cards from the same set as independent.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:  # support both direct script execution and `python -m pokeprice...`
    from .evaluation import metrics
except ImportError:
    from evaluation import metrics

REQUIRED = {
    "card_id", "set_id", "model", "seed", "eval", "y_true_log", "y_pred_log",
}
METRIC_COLS = ["mae_log", "spearman", "medape", "within_2x"]


def score_groups(frame, group_cols):
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(n=len(group), **metrics(group.y_true_log, group.y_pred_log))
        rows.append(row)
    return pd.DataFrame(rows)


def add_identity_status(frame):
    frame = frame.copy()
    if "identity_seen_in_train" not in frame:
        frame["identity_status"] = "not_recorded"
        return frame
    seen = frame.identity_seen_in_train.astype("string").str.lower()
    frame["identity_status"] = np.select(
        [seen.isin(["true", "1"]), seen.isin(["false", "0"])],
        ["seen", "unseen"], default="not_applicable")
    return frame


def paired_cluster_bootstrap(frame, reference, n_boot, seed):
    """Paired MAE deltas with whole-set bootstrap confidence intervals."""
    averaged = (frame.groupby(
        ["eval", "model", "card_id", "set_id"], as_index=False)
        .agg(y_true_log=("y_true_log", "first"),
             y_pred_log=("y_pred_log", "mean")))
    reference_rows = averaged[averaged.model == reference].rename(
        columns={"y_pred_log": "reference_pred"})
    if reference_rows.empty:
        raise ValueError(f"reference model {reference!r} is not present")

    rng = np.random.default_rng(seed)
    rows = []
    for evaluation in sorted(averaged["eval"].unique()):
        ref = reference_rows[reference_rows["eval"] == evaluation][
            ["card_id", "set_id", "y_true_log", "reference_pred"]]
        for model in sorted(set(averaged.model) - {reference}):
            candidate = averaged[
                (averaged["eval"] == evaluation) & (averaged.model == model)]
            paired = candidate.merge(
                ref, on=["card_id", "set_id"], suffixes=("", "_reference"),
                validate="one_to_one")
            if len(paired) != len(candidate) or len(paired) != len(ref):
                raise ValueError(
                    f"{model} and {reference} do not predict identical cards in {evaluation}")
            if not np.allclose(paired.y_true_log, paired.y_true_log_reference):
                raise ValueError("models disagree on true labels")
            paired["loss_delta"] = (
                np.abs(paired.y_true_log - paired.y_pred_log)
                - np.abs(paired.y_true_log - paired.reference_pred))
            clusters = paired.set_id.drop_duplicates().to_numpy()
            boot = np.empty(n_boot)
            by_cluster = {
                cluster: paired.loc[paired.set_id == cluster, "loss_delta"].to_numpy()
                for cluster in clusters}
            for index in range(n_boot):
                sampled = rng.choice(clusters, size=len(clusters), replace=True)
                boot[index] = np.concatenate(
                    [by_cluster[cluster] for cluster in sampled]).mean()
            rows.append(dict(
                eval=evaluation, model=model, reference=reference,
                n_cards=len(paired), n_sets=len(clusters),
                mae_delta=float(paired.loss_delta.mean()),
                ci95_low=float(np.quantile(boot, 0.025)),
                ci95_high=float(np.quantile(boot, 0.975)),
            ))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+", help="one or more prediction CSVs")
    parser.add_argument("--out", default="results/evaluation")
    parser.add_argument("--reference", default=None, help="model id for paired MAE deltas")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.predictions]
    predictions = pd.concat(frames, ignore_index=True)
    missing = REQUIRED - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction files are missing columns: {sorted(missing)}")
    key = ["card_id", "model", "seed", "eval"]
    if predictions.duplicated(key).any():
        example = predictions.loc[predictions.duplicated(key, keep=False), key].head()
        raise ValueError(f"duplicate prediction keys found:\n{example}")
    predictions = add_identity_status(predictions)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_scores = score_groups(predictions, ["eval", "model", "seed"])
    summary = run_scores.groupby(["eval", "model"])[METRIC_COLS].agg(
        ["mean", "std"]).round(4)
    per_set = score_groups(predictions, ["eval", "model", "seed", "set_id"])
    by_identity = score_groups(
        predictions, ["eval", "model", "seed", "identity_status"])

    run_scores.to_csv(f"{out}_runs.csv", index=False)
    summary.to_csv(f"{out}_summary.csv")
    per_set.to_csv(f"{out}_per_set.csv", index=False)
    by_identity.to_csv(f"{out}_by_identity.csv", index=False)
    if args.reference:
        paired = paired_cluster_bootstrap(
            predictions, args.reference, args.bootstrap, args.seed)
        paired.to_csv(f"{out}_paired.csv", index=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
