"""Take a timestamped snapshot of active eBay Browse API asking prices.

These are unsold listings, not realised prices. The outputs are suitable only
for a clearly labelled exploratory external-domain analysis; they are never a
replacement for the TCGplayer target used to train or select models.

By default the script queries the preselected card IDs in
data/ebay_target_cards.csv and requests fresh API responses. Credentials are
read from the repository .env by dashboard_server.load_env.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:  # support direct script execution and package imports
    from .dashboard_server import Browse, browse_query, load_env, summarise_active
except ImportError:
    from dashboard_server import Browse, browse_query, load_env, summarise_active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--targets", default="data/ebay_target_cards.csv")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--use-cache", action="store_true",
        help="allow responses cached by the dashboard (recorded in output)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    load_env(Path(".env"))
    client = Browse()
    if not client.configured:
        raise RuntimeError("EBAY_CLIENT_ID and EBAY_CLIENT_SECRET are required")

    cards = pd.read_csv(data_dir / "cards.csv")
    targets = pd.read_csv(args.targets, usecols=["card_id"]).drop_duplicates()
    selected = targets.merge(cards, on="card_id", how="left", validate="one_to_one")
    if selected.name.isna().any():
        missing = selected.loc[selected.name.isna(), "card_id"].tolist()
        raise ValueError(f"target card IDs missing from cards.csv: {missing}")

    retrieved = datetime.now(timezone.utc).replace(microsecond=0)
    snapshot_id = retrieved.strftime("%Y%m%dT%H%M%SZ")
    listing_rows, card_rows = [], []
    for card in selected.to_dict(orient="records"):
        query = browse_query(card)
        response, cached = client.search(
            query, limit=args.limit, use_cache=args.use_cache)
        summary = summarise_active(response, card)
        card_rows.append(dict(
            snapshot_id=snapshot_id, retrieved_at_utc=retrieved.isoformat(),
            card_id=card["card_id"], query=query, response_cached=cached,
            n_returned=summary["n_returned"], n_kept=summary["n_kept"],
            n_buy_it_now=summary["n_buy_it_now"],
            lowest_ask_usd=summary["lowest_ask"],
            median_ask_usd=summary["median_ask"],
        ))
        for item in summary["items"]:
            listing_rows.append(dict(
                snapshot_id=snapshot_id, retrieved_at_utc=retrieved.isoformat(),
                card_id=card["card_id"], query=query, response_cached=cached,
                **item,
            ))

    listings = pd.DataFrame(listing_rows)
    summaries = pd.DataFrame(card_rows)
    listings.to_csv(data_dir / f"ebay_live_listings_{snapshot_id}.csv", index=False)
    summaries.to_csv(data_dir / f"ebay_live_cards_{snapshot_id}.csv", index=False)
    print(
        f"snapshot {snapshot_id}: {len(summaries)} cards, "
        f"{len(listings)} returned listings; asking prices only")


if __name__ == "__main__":
    main()
