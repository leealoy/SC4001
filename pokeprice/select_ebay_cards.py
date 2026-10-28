"""Choose, IN ADVANCE and at random, which cards to look up on eBay Sold Items.

Choosing cards before looking at eBay avoids selection bias (picking whatever
has convenient listings).  Cards are sampled with a fixed seed from strata of
rarity group x era, equal allocation per stratum, from cards that have a
TCGplayer price >= --min-price (cheaper cards rarely have >=3 sales in a short
window).  NOTE this price floor means the eBay set under-represents bulk
commons; state this in the report.

Output: data/ebay_target_cards.csv, sorted so you work through each stratum in
`order` (1, 2, 3, ...) and stop a stratum once it has --keep cards with >=3 sold
listings.  Candidates beyond --keep are spares for cards with too few sales.
The TCGplayer price is NOT shown to you while collecting: it is in cards.csv,
not in this worklist, so you do not anchor on it.

Usage: python pokeprice/select_ebay_cards.py --per-stratum 14 --seed 42
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ERAS = [("1999-2002 WotC", "1999-01-01", "2003-01-01"),
        ("2003-2012", "2003-01-01", "2013-01-01"),
        ("2013-2019", "2013-01-01", "2020-01-01"),
        ("2020-2023", "2020-01-01", "2024-01-01"),
        ("2024-2025", "2024-01-01", "2026-01-01")]


def rarity_group(r):
    r = str(r)
    if r in ("Common", "Uncommon"):
        return "common/uncommon"
    if r in ("Rare", "Rare Holo", "Double Rare", "Rare Holo EX", "Rare Holo GX", "Rare Holo V"):
        return "rare/holo"
    if r == "Promo":
        return "promo"
    return "ultra/special"          # Ultra, Illustration, Hyper, Secret, Rainbow, ...


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--per-stratum", type=int, default=14, help="candidates drawn per stratum")
    ap.add_argument("--keep", type=int, default=8, help="cards wanted per stratum (rest are spares)")
    ap.add_argument("--min-price", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-promos", action="store_true",
                    help="promos are excluded by default: sellers omit the /total so titles "
                         "cannot be matched reliably (report as a limitation)")
    args = ap.parse_args()
    d = Path(args.data)

    c = pd.read_csv(d / "cards.csv", parse_dates=["release_date"])
    c = c[c.price_market.notna() & (c.price_market >= args.min_price) & (c.supertype == "Pokémon")].copy()
    c["era"] = None
    for name, lo, hi in ERAS:
        c.loc[(c.release_date >= lo) & (c.release_date < hi), "era"] = name
    c = c[c.era.notna()]
    c["rarity_group"] = c.rarity.map(rarity_group)
    if not args.include_promos:
        c = c[c.rarity_group != "promo"]
    c["stratum"] = c.rarity_group + " | " + c.era

    rng = np.random.default_rng(args.seed)
    parts = []
    for s, g in c.groupby("stratum"):
        g = g.iloc[rng.permutation(len(g))][:args.per_stratum].copy()
        g["order"] = range(1, len(g) + 1)
        parts.append(g)
    out = pd.concat(parts)
    out["number_full"] = out.number.astype(str) + "/" + out.printed_total.astype("Int64").astype(str)
    # string to paste into eBay search; avoids words our filters reject
    out["search_query"] = out.name + " " + out.number_full + " " + out.set_name
    out["is_spare"] = out.order > args.keep
    for col in ["n_sold_found", "notes"]:
        out[col] = ""
    out = out.sort_values(["stratum", "order"])
    cols = ["stratum", "order", "is_spare", "card_id", "search_query", "name", "set_name",
            "number_full", "rarity", "release_date", "n_sold_found", "notes"]
    out[cols].to_csv(d / "ebay_target_cards.csv", index=False)

    print(f"{len(out)} candidates in {out.stratum.nunique()} strata "
          f"(wanted {args.keep}/stratum -> {out.stratum.nunique() * args.keep} cards, "
          f">=3 sales each -> >= {3 * out.stratum.nunique() * args.keep} listings)")
    print(out.groupby("stratum").size().to_string())
    small = c.groupby("stratum").size()
    print("strata smaller than requested:", small[small < args.per_stratum].to_dict() or "none")


if __name__ == "__main__":
    main()
