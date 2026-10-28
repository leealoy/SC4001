"""Clean manually collected eBay sold listings and match them to card IDs.

Input : data/ebay_sold.csv   (collected BY HAND from ebay.com -> Sold Items;
                              we do not scrape eBay, see README)
        data/cards.csv       (from fetch_cards.py)
Output: data/ebay_cards.csv        one row per card: median sold price (USD), n_sales
        data/ebay_clean.csv        listings that passed every filter and were matched
        data/ebay_rejected.csv     listings removed by a filter (with reason)
        data/ebay_unmatched.csv    passed filters but no unique card match (with reason)

Required columns in ebay_sold.csv:
    item_id, title, sold_price_usd, sold_date (YYYY-MM-DD)
Optional: shipping_usd, image_file (seller photo, for the robustness experiment),
          target_card_id (the card you were searching for when you copied the row;
          used only to *audit* the title matcher, never as the label).

Price used = item price only (shipping excluded).

Usage: python pokeprice/build_ebay.py --start 2026-09-01 --end 2026-10-15
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --- listing filters: (reason, regex on lowercase title) -------------------
REJECT_PATTERNS = [
    ("graded", r"\b(psa|bgs|cgc|sgc|beckett|ace|gma|graded|slab(bed)?)\b|\bgem ?mint ?10\b"),
    ("lot_or_bundle", r"\b(lot|bundle|bulk|set of|x ?\d{1,3}\b|\d+ ?cards|playset|job lot|choose your|pick your)\b"),
    ("fake_or_proxy", r"\b(proxy|custom|fan ?made|replica|reprint|fake|orica|metal card|gold card|novelty|unofficial)\b"),
    ("sealed_product", r"\b(booster|box|etb|elite trainer|tin|blister|sealed|pack|deck|binder|display|collection box)\b"),
    ("non_english", r"\b(japanese|japan|jpn|korean|chinese|german|french|italian|spanish|portuguese|thai|indonesian)\b"),
    ("damaged", r"\b(damaged|played|heavily|hp\b|poor|creased|crease|water ?damage|bent|torn|marked|inked|ripped|dmg)\b"),
    ("variant", r"\b(1st ed(ition)?|first ed(ition)?|shadowless|reverse ?holo|rev ?holo|stamped|staff|prerelease|error|misprint|cosmos|cracked ice|jumbo|oversize)\b"),
]
# a handful of very common phrases that look like rejects but are fine
ALLOW = re.compile(r"\bnear mint\b|\bnm\b|\bmint\b|\blp\b|\blightly played\b")


def number_candidates(title):
    """Pull card-number tokens from a title: '025/165', '25/165', 'SWSH123', 'TG05'."""
    t = title.upper()
    out = []
    for m in re.finditer(r"\b([A-Z]{0,4}\d{1,3})\s*/\s*([A-Z]{0,4}\d{1,3})\b", t):
        out.append((m.group(1), m.group(2)))
    for m in re.finditer(r"\b(SWSH\d{1,3}|SM\d{1,3}|XY\d{1,3}|BW\d{1,3}|SVP\s?\d{1,3}|TG\d{1,2}|GG\d{1,2})\b", t):
        out.append((m.group(1).replace(" ", ""), None))
    return out


def norm_num(s):
    """'025' -> '25', 'TG05' -> 'TG5' so titles match the API's number format."""
    return re.sub(r"(?<=\D)0+(?=\d)|^0+(?=\d)", "", str(s).upper())


def tokens(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def build_index(cards):
    """Index cards by normalised number for fast title matching."""
    cards = cards.copy()
    cards["num_norm"] = cards["number"].astype(str).map(norm_num)
    return {k: g for k, g in cards.groupby("num_norm")}


def match_title(title, index):
    """Return (card_id, None) on a unique match, else (None, reason)."""
    cands = number_candidates(title)
    if not cands:
        return None, "no_card_number_in_title"
    tl = tokens(title)
    best = []
    for num, total in cands:
        g = index.get(norm_num(num))
        if g is None:
            continue
        for r in g.itertuples():
            name_toks = tokens(r.name) - {"ex", "v", "vmax", "vstar", "gx", "the"}
            if not name_toks or not name_toks <= tl:
                continue
            score = 1
            if total is not None and pd.notna(r.printed_total) and norm_num(total) == norm_num(int(r.printed_total)):
                score += 2  # '25/165' agrees with the set's printed total
            if tokens(r.set_name) <= tl:
                score += 2  # set name is spelled out in the title
            best.append((score, r.card_id))
    if not best:
        return None, "number_or_name_not_found"
    best.sort(reverse=True)
    top = [cid for s, cid in best if s == best[0][0]]
    if len(set(top)) > 1:
        return None, "ambiguous_set"
    # a bare number+name hit with no set evidence is too weak
    if best[0][0] < 3:
        return None, "weak_set_evidence"
    return top[0], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--start", required=True, help="sale window start YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="sale window end YYYY-MM-DD")
    ap.add_argument("--min-sales", type=int, default=3)
    ap.add_argument("--outlier-ratio", type=float, default=2.5,
                    help="drop a sale if price/median or median/price exceeds this")
    args = ap.parse_args()
    d = Path(args.data)

    raw = pd.read_csv(d / "ebay_sold.csv")
    cards = pd.read_csv(d / "cards.csv")
    n0 = len(raw)
    raw["sold_date"] = pd.to_datetime(raw["sold_date"], errors="coerce")
    raw["reject_reason"] = ""

    # 1. basic validity + date window + duplicates
    def mark(mask, why):
        raw.loc[mask & (raw.reject_reason == ""), "reject_reason"] = why

    mark(raw.sold_price_usd.isna() | (raw.sold_price_usd <= 0), "bad_price")
    mark(raw.sold_date.isna() | (raw.sold_date < args.start) | (raw.sold_date > args.end), "outside_date_window")
    mark(raw.duplicated("item_id", keep="first"), "duplicate_item_id")

    # 2. title filters
    low = raw.title.fillna("").str.lower()
    for why, pat in REJECT_PATTERNS:
        mark(low.map(lambda t: re.search(pat, t) is not None), why)

    rejected = raw[raw.reject_reason != ""].copy()
    kept = raw[raw.reject_reason == ""].copy()

    # 3. match to card ids
    index = build_index(cards)
    res = [match_title(t, index) for t in kept.title.fillna("")]
    kept["card_id"] = [r[0] for r in res]
    kept["unmatched_reason"] = [r[1] for r in res]
    unmatched = kept[kept.card_id.isna()].copy()
    matched = kept[kept.card_id.notna()].copy()

    # audit the matcher against the card we were searching for (if provided)
    if "target_card_id" in matched.columns and matched.target_card_id.notna().any():
        a = matched[matched.target_card_id.notna()]
        acc = (a.card_id == a.target_card_id).mean()
        print(f"matcher agrees with target_card_id on {acc:.1%} of {len(a)} audited rows")
        matched.loc[a.index[a.card_id != a.target_card_id], "audit_mismatch"] = True

    # 4. per-card outlier removal around the median (ratio test, robust for n>=3)
    med = matched.groupby("card_id").sold_price_usd.transform("median")
    ratio = np.maximum(matched.sold_price_usd / med, med / matched.sold_price_usd)
    out_mask = ratio > args.outlier_ratio
    out = matched[out_mask].assign(reject_reason="price_outlier")
    matched = matched[~out_mask]
    rejected = pd.concat([rejected, out])

    # 5. aggregate; require >= min_sales surviving sales per card
    agg = (matched.groupby("card_id").sold_price_usd
           .agg(ebay_median_usd="median", n_sales="size", ebay_min="min", ebay_max="max")
           .reset_index())
    few = agg[agg.n_sales < args.min_sales]
    agg = agg[agg.n_sales >= args.min_sales]
    matched = matched[matched.card_id.isin(agg.card_id)]
    # attach api metadata for convenience
    agg = agg.merge(cards[["card_id", "name", "set_id", "rarity", "release_date", "price_market"]],
                    on="card_id", how="left")

    agg.to_csv(d / "ebay_cards.csv", index=False)
    matched.to_csv(d / "ebay_clean.csv", index=False)
    rejected.to_csv(d / "ebay_rejected.csv", index=False)
    unmatched.to_csv(d / "ebay_unmatched.csv", index=False)

    print(f"listings in:            {n0}")
    print(f"rejected by filters:    {len(rejected)}")
    print(rejected.reject_reason.value_counts().to_string())
    print(f"unmatched to a card:    {len(unmatched)}")
    print(unmatched.unmatched_reason.value_counts().to_string())
    print(f"cards with <{args.min_sales} sales dropped: {len(few)}")
    print(f"FINAL: {len(agg)} cards, {len(matched)} sales -> {d/'ebay_cards.csv'}")
    print("REVIEW ebay_rejected.csv and ebay_unmatched.csv by hand before using the result.")


if __name__ == "__main__":
    main()
