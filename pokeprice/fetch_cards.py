"""Download Pokemon card metadata, TCGplayer prices and official images.

Source: Pokemon TCG API (https://docs.pokemontcg.io), v2.

Outputs (under --out, default data/):
    sets.csv         one row per set (id, name, series, releaseDate, ...)
    cards.csv        one row per card with metadata + price columns
    raw/page_XXXX.json   raw API pages (makes the script resumable)
    images/<card_id>.png official small card image (only with --images)

The script is resumable: pages already saved in raw/ are not requested again,
and images already on disk are skipped.  Set POKEMONTCG_API_KEY to raise the
rate limit (optional; a free key can be requested from the API site).

Usage:
    python pokeprice/fetch_cards.py                 # metadata + prices
    python pokeprice/fetch_cards.py --images        # also download images
"""
import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

try:  # support direct script execution and package imports
    from .path_utils import card_image_filename
except ImportError:
    from path_utils import card_image_filename

API = "https://api.pokemontcg.io/v2"
PAGE_SIZE = 250  # API maximum
# TCGplayer price keys that correspond to a card's *standard* printing.
# Preference order: plain non-foil first, then foil-only cards (e.g. most
# modern rares only exist as holofoil).
STANDARD_KEYS = ["normal", "holofoil", "unlimited", "unlimitedHolofoil"]


def get_json(session, url, params=None, retries=20):
    """GET with exponential backoff; the public API throws 5xx / timeouts often."""
    delay = 3
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code} (attempt {attempt}/{retries}); retry in {delay}s")
        except (requests.RequestException, ValueError) as e:
            print(f"  {type(e).__name__} (attempt {attempt}/{retries}); retry in {delay}s")
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise RuntimeError(f"giving up on {url} {params}")


def fetch_sets(session, out):
    data = get_json(session, f"{API}/sets", {"orderBy": "releaseDate"})["data"]
    rows = [
        dict(set_id=s["id"], set_name=s["name"], series=s.get("series"),
             release_date=s.get("releaseDate"), printed_total=s.get("printedTotal"),
             total=s.get("total"))
        for s in data
    ]
    df = pd.DataFrame(rows)
    df["release_date"] = pd.to_datetime(df["release_date"], format="%Y/%m/%d")
    df.to_csv(out / "sets.csv", index=False)
    print(f"sets: {len(df)}")
    return df


def fetch_pages(session, raw_dir):
    """Download every card page, skipping pages already cached on disk."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    page, total_count = 1, None
    while True:
        f = raw_dir / f"page_{page:04d}.json"
        if f.exists():
            payload = json.loads(f.read_text())
        else:
            payload = get_json(session, f"{API}/cards",
                               {"page": page, "pageSize": PAGE_SIZE})
            f.write_text(json.dumps(payload))
            time.sleep(0.5)  # be polite
        total_count = payload.get("totalCount", total_count)
        n = len(payload["data"])
        print(f"page {page}: {n} cards (total {total_count})")
        if n == 0 or page * PAGE_SIZE >= total_count:
            break
        page += 1


def pick_price(card):
    """Return (variant, market, low, mid, high) for the standard printing, or Nones.

    Only the *standard* printing is used (normal, else holofoil; vintage sets use "unlimited" for the non-1st-edition print).  Reverse
    holo / 1st edition variants are ignored so the target is one price per
    card.  `market` is the TCGplayer market price in USD.
    """
    prices = (card.get("tcgplayer") or {}).get("prices") or {}
    for key in STANDARD_KEYS:
        p = prices.get(key)
        if p and p.get("market"):
            return key, p["market"], p.get("low"), p.get("mid"), p.get("high")
    return None, None, None, None, None


def pokemon_identity(card):
    """Return a stable species identity from National Pokedex numbers.

    Printed names are not a safe identity: for example, ``Charizard``,
    ``Charizard ex`` and ``Charizard VMAX`` would be treated as unrelated.  The
    API's National Pokedex numbers group those mechanics under the same species.
    Multi-Pokemon cards retain a sorted, pipe-separated tuple.  Trainer and
    Energy cards intentionally have no Pokemon identity.
    """
    if not str(card.get("supertype", "")).lower().startswith("pok"):
        return None, None
    numbers = sorted({int(n) for n in (card.get("nationalPokedexNumbers") or [])})
    if not numbers:
        return None, None
    raw = "|".join(map(str, numbers))
    return raw, raw


def parse_cards(raw_dir):
    rows = []
    for f in sorted(raw_dir.glob("page_*.json")):
        for c in json.loads(f.read_text())["data"]:
            variant, market, low, mid, high = pick_price(c)
            pokedex_numbers, identity = pokemon_identity(c)
            rows.append(dict(
                card_id=c["id"], name=c["name"], supertype=c.get("supertype"),
                subtypes="|".join(c.get("subtypes") or []),
                hp=c.get("hp"), types="|".join(c.get("types") or []),
                national_pokedex_numbers=pokedex_numbers,
                pokemon_identity=identity,
                evolves_from=c.get("evolvesFrom"),
                rarity=c.get("rarity"), artist=c.get("artist"),
                number=c.get("number"), set_id=c["set"]["id"],
                set_name=c["set"]["name"], series=c["set"].get("series"),
                release_date=c["set"].get("releaseDate"),
                printed_total=c["set"].get("printedTotal"),
                n_attacks=len(c.get("attacks") or []),
                n_abilities=len(c.get("abilities") or []),
                retreat_cost=c.get("convertedRetreatCost"),
                regulation_mark=c.get("regulationMark"),
                price_variant=variant, price_market=market,
                price_low=low, price_mid=mid, price_high=high,
                price_updated=(c.get("tcgplayer") or {}).get("updatedAt"),
                image_url=(c.get("images") or {}).get("small"),
                image_url_large=(c.get("images") or {}).get("large"),
                image_file=card_image_filename(c["id"]),
            ))
    df = pd.DataFrame(rows).drop_duplicates("card_id")
    df["release_date"] = pd.to_datetime(df["release_date"], format="%Y/%m/%d")
    return df


def download_images(session, df, img_dir, size="small"):
    img_dir.mkdir(parents=True, exist_ok=True)
    col = "image_url" if size == "small" else "image_url_large"
    todo = df[df["price_market"].notna()]
    ok = fail = 0
    for i, (cid, url) in enumerate(zip(todo["card_id"], todo[col])):
        dest = img_dir / card_image_filename(cid)
        if dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            ok += 1
        except requests.RequestException:
            fail += 1
        if i % 500 == 0:
            print(f"  images: {i}/{len(todo)} (new {ok}, failed {fail})")
    print(f"images done: new {ok}, failed {fail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--images", action="store_true", help="also download card images")
    ap.add_argument("--image-size", choices=["small", "large"], default="small")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    if os.environ.get("POKEMONTCG_API_KEY"):
        session.headers["X-Api-Key"] = os.environ["POKEMONTCG_API_KEY"]

    fetch_sets(session, out)
    fetch_pages(session, out / "raw")
    df = parse_cards(out / "raw")
    df.to_csv(out / "cards.csv", index=False)
    # Record when prices were downloaded; baselines.py uses it as REF_DATE.
    (out / "download_date.txt").write_text(date.today().isoformat())

    priced = df["price_market"].notna().sum()
    print(f"\ncards: {len(df)}  with standard-printing price: {priced}")
    print(df.loc[df.price_market.notna(), "price_variant"].value_counts().to_string())
    if args.images:
        download_images(session, df, out / "images", args.image_size)


if __name__ == "__main__":
    sys.exit(main())
