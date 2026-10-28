"""Local dashboard for checking the hand-collected eBay sold prices.

Serves dashboard.html and a small JSON API over the project's own CSV files, so it
needs no network access except for card images (images.pokemontcg.io):

    data/cards.csv, data/sets.csv        card metadata (from fetch_cards.py)
    data/ebay_cards.csv                  median sold price per card   (build_ebay.py)
    data/ebay_clean.csv                  sales that passed the filters (build_ebay.py)
    data/ebay_rejected.csv               removed by a filter, with reason
    data/ebay_unmatched.csv              passed filters, no unique card match

Active (unsold) listings come live from the eBay Browse API.  These are ASKING
prices, not sale prices, and are shown separately from the sold data.  The Browse
API needs only the default OAuth scope, so ordinary production keys work.  Keys
(never committed; .env is git-ignored) go in the repo-root .env:
    EBAY_CLIENT_ID=...        # App ID
    EBAY_CLIENT_SECRET=...    # Cert ID
    EBAY_ENV=production       # or "sandbox"

TCGplayer price columns in cards.csv are never sent to the page: the dashboard
shows eBay prices only.  The CSVs are re-read whenever build_ebay.py rewrites
them, so re-run it and refresh the page.

Endpoints:
    GET /api/status                      what data is loaded
    GET /api/sets                        sets for the filter
    GET /api/cards?q=&set=&rarity=&sort=&only_ebay=1&page=1
    GET /api/ebay/card?id=<card_id>      one card's sold sales + audit rows (CSV)
    GET /api/ebay/active?id=<card_id>    one card's active listings (Browse API)
    GET /api/ebay/review                 all rejected / unmatched listings

Usage:
    python pokeprice/dashboard_server.py          # then open http://127.0.0.1:8765/
"""
import argparse
import base64
import hashlib
import json
import os
import re
import statistics
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

from build_ebay import REJECT_PATTERNS, norm_num, number_candidates, tokens

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE_DIR = ROOT / "data" / "ebay_api_cache"
CACHE_TTL = 6 * 3600  # seconds; asking prices move, but the daily call quota is 5,000
CATEGORY = "183454"  # Toys & Hobbies > CCG Individual Cards
# card-name words too generic to require in a listing title
WEAK_NAME_TOKENS = {"ex", "v", "vmax", "vstar", "gx", "the"}
PAGE_SIZE = 24
# card metadata sent to the page (no TCGplayer price columns)
CARD_COLS = ["card_id", "name", "supertype", "subtypes", "rarity", "artist", "number",
             "set_id", "set_name", "series", "release_date", "printed_total",
             "image_url", "image_url_large"]
SORTS = {
    "newest": (["release_date", "num_sort"], [False, True]),
    "oldest": (["release_date", "num_sort"], [True, True]),
    "name": (["name", "release_date"], [True, False]),
    "ebay_price": (["ebay_median_usd", "release_date"], [False, False]),
    "ebay_sales": (["n_sales", "release_date"], [False, False]),
}


def records(df):
    """DataFrame -> list of dicts with NaN as null."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def load_env(path):
    """Minimal KEY=VALUE .env reader; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


class EbayError(Exception):
    pass


class Browse:
    """eBay Browse API client (active listings) with an application token."""

    def __init__(self):
        self.client_id = os.environ.get("EBAY_CLIENT_ID", "")
        self.client_secret = os.environ.get("EBAY_CLIENT_SECRET", "")
        sandbox = os.environ.get("EBAY_ENV", "production").lower() == "sandbox"
        self.base = "https://api.sandbox.ebay.com" if sandbox else "https://api.ebay.com"
        self.env = "sandbox" if sandbox else "production"
        self.token, self.token_exp = None, 0
        self.session = requests.Session()

    @property
    def configured(self):
        return bool(self.client_id and self.client_secret)

    def get_token(self):
        if self.token and time.time() < self.token_exp - 60:
            return self.token
        basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        r = self.session.post(
            f"{self.base}/identity/v1/oauth2/token",
            headers={"Authorization": f"Basic {basic}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials",
                  "scope": "https://api.ebay.com/oauth/api_scope"}, timeout=30)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code != 200:
            raise EbayError(f"{body.get('error', f'HTTP {r.status_code}')}: "
                            f"{body.get('error_description', r.text[:200])}")
        self.token = body["access_token"]
        self.token_exp = time.time() + int(body.get("expires_in", 7200))
        return self.token

    def search(self, q, limit=50):
        """item_summary/search for Buy It Now and auction listings (cached on disk)."""
        key = hashlib.sha1(f"browse|{self.env}|{q}|{limit}".encode()).hexdigest()
        f = CACHE_DIR / f"{key}.json"
        if f.exists() and time.time() - f.stat().st_mtime < CACHE_TTL:
            return json.loads(f.read_text()), True
        r = self.session.get(
            f"{self.base}/buy/browse/v1/item_summary/search",
            headers={"Authorization": f"Bearer {self.get_token()}",
                     "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"},
            params={"q": q, "category_ids": CATEGORY, "limit": limit,
                    "filter": "buyingOptions:{FIXED_PRICE|AUCTION},priceCurrency:USD"},
            timeout=30)
        if r.status_code != 200:
            raise EbayError(f"Browse API HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(data))
        return data, False


# reprint sets that reuse the original card numbers (e.g. Charizard 4/102)
REPRINT_MARKERS = re.compile(r"\bcelebrations?\b|\bclassic collection\b|\bbase set 2\b")


def reject_reason(title, name, number, total=None, set_name=""):
    """Why a listing is not a raw English single of this card (None = keep).

    Same title filters as build_ebay.py, plus: the card's number and name must
    appear in the title, a '4/102'-style total must match the set's printed total
    when known, and reprint-set titles are rejected for the original card.
    """
    low = title.lower()
    for why, pat in REJECT_PATTERNS:
        if re.search(pat, low):
            return why
    if REPRINT_MARKERS.search(low) and not REPRINT_MARKERS.search(str(set_name).lower()):
        return "reprint_set"
    want = norm_num(number)
    cands = number_candidates(title)
    if cands:
        hits = [t for n, t in cands if norm_num(n) == want]
        if not hits:
            return "card_number_missing"
        if pd.notna(total) and total and all(t is not None and norm_num(t) != norm_num(int(total)) for t in hits):
            return "set_total_mismatch"
    # no '25/165'-style number in the title: accept a standalone '#25' / '25' token
    elif not re.search(rf"(?<![\w/$.]){re.escape(str(number))}(?![\w/.])", title, re.I):
        return "card_number_missing"
    if not (tokens(name) - WEAK_NAME_TOKENS) <= tokens(title):
        return "name_missing"
    return None


def browse_query(card):
    number, total = str(card["number"]), card.get("printed_total")
    if pd.notna(total) and number.isdigit():
        return f"{card['name']} {number}/{int(total)}"
    # no printed total (some new sets): the set name narrows the search instead
    return f"{card['name']} {number} {card['set_name']}"


def summarise_active(data, card):
    name, number = card["name"], card["number"]
    items = []
    for it in data.get("itemSummaries") or []:
        opts = it.get("buyingOptions") or []
        # auctions carry the current bid; Buy It Now carries the asking price
        price = it.get("currentBidPrice") if "AUCTION" in opts and "FIXED_PRICE" not in opts else it.get("price")
        if not price or price.get("currency") != "USD":
            continue
        ship = (it.get("shippingOptions") or [{}])[0].get("shippingCost") or {}
        title = it.get("title", "")
        items.append(dict(
            title=title, price=float(price["value"]),
            kind="auction" if "AUCTION" in opts and "FIXED_PRICE" not in opts else "buy_it_now",
            bids=it.get("bidCount"), ends=it.get("itemEndDate"),
            shipping=float(ship["value"]) if ship.get("value") is not None else None,
            condition=it.get("condition"), url=it.get("itemWebUrl"),
            image=(it.get("image") or {}).get("imageUrl"),
            reject=reject_reason(title, name, number, card.get("printed_total"), card.get("set_name", ""))))
    items.sort(key=lambda s: s["price"])
    bin_prices = [s["price"] for s in items if s["reject"] is None and s["kind"] == "buy_it_now"]
    return dict(
        n_returned=len(items), n_kept=sum(s["reject"] is None for s in items),
        n_buy_it_now=len(bin_prices),
        lowest_ask=min(bin_prices) if bin_prices else None,
        median_ask=statistics.median(bin_prices) if bin_prices else None,
        items=items)


class Data:
    """Loads the CSVs, reloading any file whose modification time changed."""

    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.mtimes = {}
        self.frames = {}

    def _get(self, name):
        f = self.dir / name
        if not f.exists():
            self.frames.pop(name, None)
            self.mtimes.pop(name, None)
            return None
        m = f.stat().st_mtime
        if self.mtimes.get(name) != m:
            self.frames[name] = pd.read_csv(f, dtype={"item_id": str, "number": str})
            self.mtimes[name] = m
        return self.frames[name]

    def cards(self):
        cards = self._get("cards.csv")
        if cards is None:
            raise FileNotFoundError(f"{self.dir / 'cards.csv'} not found; run fetch_cards.py first")
        cards = cards[CARD_COLS].copy()
        # numeric part of the card number, so '2' sorts before '10'
        cards["num_sort"] = pd.to_numeric(cards["number"].str.extract(r"(\d+)")[0], errors="coerce")
        agg = self._get("ebay_cards.csv")
        if agg is not None:
            cards = cards.merge(agg[["card_id", "ebay_median_usd", "n_sales", "ebay_min", "ebay_max"]],
                                on="card_id", how="left")
            last = self.last_sold()
            if last is not None:
                cards = cards.merge(last, on="card_id", how="left")
        for col in ["ebay_median_usd", "n_sales", "ebay_min", "ebay_max", "last_sold_usd", "last_sold_date"]:
            if col not in cards:
                cards[col] = None
        return cards

    def last_sold(self):
        """Most recent clean sale per card (latest sold_date; ties -> first row)."""
        clean = self._get("ebay_clean.csv")
        if clean is None or clean.empty:
            return None
        c = clean.sort_values("sold_date", ascending=False, kind="stable")
        c = c.drop_duplicates("card_id")
        return c[["card_id", "sold_price_usd", "sold_date"]].rename(
            columns={"sold_price_usd": "last_sold_usd", "sold_date": "last_sold_date"})

    def status(self):
        def n(name):
            df = self._get(name)
            return None if df is None else len(df)
        cards = self._get("cards.csv")
        return dict(
            data_dir=str(self.dir),
            cards=None if cards is None else len(cards),
            ebay_cards=n("ebay_cards.csv"), ebay_clean=n("ebay_clean.csv"),
            ebay_rejected=n("ebay_rejected.csv"), ebay_unmatched=n("ebay_unmatched.csv"),
            ebay_raw=n("ebay_sold.csv"))

    def sets(self):
        sets = self._get("sets.csv")
        if sets is None:
            return []
        return records(sets.sort_values("release_date", ascending=False)[["set_id", "set_name", "release_date"]])

    def search(self, p):
        cards = self.cards()
        q = p.get("q", "").strip().lower()
        if q:
            cards = cards[cards["name"].str.lower().str.contains(q, regex=False, na=False)
                          | (cards["card_id"].str.lower() == q)]
        if p.get("set"):
            cards = cards[cards["set_id"] == p["set"]]
        if p.get("rarity"):
            cards = cards[cards["rarity"] == p["rarity"]]
        if p.get("only_ebay") == "1":
            cards = cards[cards["n_sales"].notna()]
        by, asc = SORTS.get(p.get("sort"), SORTS["newest"])
        cards = cards.sort_values(by, ascending=asc, na_position="last", kind="stable")
        page = max(1, int(p.get("page", 1) or 1))
        start = (page - 1) * PAGE_SIZE
        out = cards.iloc[start:start + PAGE_SIZE].drop(columns="num_sort")
        return dict(total=len(cards), page=page, page_size=PAGE_SIZE, cards=records(out))

    def card(self, card_id):
        cards = self._get("cards.csv")
        row = None if cards is None else cards[cards["card_id"] == card_id]
        if row is None or row.empty:
            raise KeyError(f"unknown card_id {card_id!r}")
        return row.iloc[0]

    def card_sales(self, card_id):
        """Clean sales for a card, plus every audit row that names it as the target."""
        out = dict(card_id=card_id, clean=[], rejected=[], unmatched=[], mislabelled=[])
        clean = self._get("ebay_clean.csv")
        if clean is not None:
            c = clean[clean["card_id"] == card_id].sort_values("sold_date", ascending=False)
            out["clean"] = records(c)
            # rows the matcher gave to a *different* card than the one being searched for
            if "target_card_id" in clean:
                out["mislabelled"] = records(clean[(clean["target_card_id"] == card_id)
                                                   & (clean["card_id"] != card_id)])
        for name, key in [("ebay_rejected.csv", "rejected"), ("ebay_unmatched.csv", "unmatched")]:
            df = self._get(name)
            if df is not None and "target_card_id" in df:
                out[key] = records(df[df["target_card_id"] == card_id].sort_values("sold_date", ascending=False))
        return out

    def review(self):
        out = {}
        for name, key, reason in [("ebay_rejected.csv", "rejected", "reject_reason"),
                                  ("ebay_unmatched.csv", "unmatched", "unmatched_reason")]:
            df = self._get(name)
            if df is None:
                out[key] = None
                continue
            out[key] = dict(counts=df[reason].fillna("?").value_counts().to_dict(),
                            rows=records(df.sort_values(reason)))
        return out


def make_handler(data, browse):
    def browse_status():
        out = dict(configured=browse.configured, env=browse.env, token_ok=False, error=None)
        if not browse.configured:
            out["error"] = "EBAY_CLIENT_ID / EBAY_CLIENT_SECRET not set in the repo-root .env"
            return out
        try:
            browse.get_token()
            out["token_ok"] = True
        except (EbayError, requests.RequestException) as e:
            out["error"] = str(e)
        return out

    def active(card_id):
        card = data.card(card_id)
        q = browse_query(card)
        try:
            res, cached = browse.search(q)
        except (EbayError, requests.RequestException) as e:
            raise EbayError(f"{e} (query: {q})")
        return dict(query=q, cached=cached, **summarise_active(res, card))

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(HERE), **kw)

        def send_json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            p = {k: v[0] for k, v in parse_qs(u.query).items()}
            routes = {
                "/api/status": lambda: dict(data.status(), browse=browse_status()),
                "/api/sets": data.sets,
                "/api/cards": lambda: data.search(p),
                "/api/ebay/card": lambda: data.card_sales(p.get("id", "")),
                "/api/ebay/active": lambda: active(p.get("id", "")),
                "/api/ebay/review": data.review,
            }
            if u.path in routes:
                try:
                    return self.send_json(200, routes[u.path]())
                except FileNotFoundError as e:
                    return self.send_json(503, {"error": str(e)})
                except EbayError as e:
                    return self.send_json(502, {"error": str(e)})
                except (KeyError, ValueError) as e:
                    return self.send_json(500, {"error": f"{type(e).__name__}: {e}"})
            if u.path == "/":
                self.path = "/dashboard.html"
            return super().do_GET()

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE.parent / "data"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    load_env(ROOT / ".env")
    data = Data(args.data)
    browse = Browse()
    print(json.dumps(data.status(), indent=1))
    print(f"eBay Browse API ({browse.env}): credentials {'found' if browse.configured else 'MISSING'}")
    print(f"dashboard: http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(data, browse)).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
