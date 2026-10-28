"""Turn text you COPIED BY HAND from an eBay "Sold items" results page into rows
of data/ebay_sold.csv.

This script does not touch eBay.  You search for one card on ebay.com (Sold items
filter on), select the results page, copy it, paste it into a text file, and run
this parser on that file.  It only saves retyping.

Workflow for one card from data/ebay_target_cards.csv:
    1. ebay.com -> paste the card's `search_query` -> tick "Sold items" (+ "Completed").
    2. Select the listings text on the page (Cmd/Ctrl-A), copy, paste into paste.txt.
    3. python pokeprice/parse_ebay_paste.py paste.txt --card-id base1-4

Rows are appended to data/ebay_sold.csv (created if missing).  Duplicates (same
title, price and date) are skipped.  `target_card_id` is recorded only to audit the
title matcher in build_ebay.py; it is never used as a label.

Expected layout of each listing in the pasted text (eBay's wording can change, so
check the printed summary and open the CSV if the counts look wrong):
    Sold  Oct 12, 2026            <- starts a listing
    <title>
    <condition line, e.g. Pre-Owned / Brand New>
    $3.50                         <- first price after the title = item price
    ... + $1.25 delivery / Free delivery ...
Listings priced as a range ("$3.00 to $5.00") are skipped; they are not single sales.
"""
import argparse
import hashlib
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

SOLD = re.compile(r"^\s*Sold\s+(?:on\s+)?([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})\s*$")
PRICE = re.compile(r"^\s*(?:US\s*)?\$\s?([\d,]+(?:\.\d{1,2})?)\s*$")
SHIP = re.compile(r"\+\s*(?:US\s*)?\$\s?([\d,]+(?:\.\d{1,2})?)")
NOISE = re.compile(r"^(new listing|opens in a new window or tab|sponsored|pre-owned|brand new|"
                   r"open box|used|ungraded|best offer accepted|or best offer|"
                   r"free (shipping|delivery|returns)|.*\bwatchers?\b|.*\bsold\b$)", re.I)


def parse(text):
    """Return a list of dicts (title, price, date, shipping) from pasted page text."""
    lines = [l.rstrip() for l in text.splitlines()]
    rows, i = [], 0
    while i < len(lines):
        m = SOLD.match(lines[i])
        if not m:
            i += 1
            continue
        date = datetime.strptime(m.group(1), "%b %d, %Y").date().isoformat()
        j, title, price, ship, range_price = i + 1, None, None, None, False
        # scan until the next "Sold <date>" line
        while j < len(lines) and not SOLD.match(lines[j]):
            s = lines[j].strip()
            if s and title is None and not NOISE.match(s) and not PRICE.match(s):
                title = re.sub(r"\s*Opens in a new window or tab\s*$", "", s, flags=re.I)
            elif title and price is None and PRICE.match(s):
                price = float(PRICE.match(s).group(1).replace(",", ""))
            elif title and price is None and re.search(r"\$.*\bto\b.*\$", s):
                range_price = True
            elif price is not None and ship is None:
                sm = SHIP.search(s)
                if sm:
                    ship = float(sm.group(1).replace(",", ""))
                elif re.search(r"free (shipping|delivery)", s, re.I):
                    ship = 0.0
            j += 1
        if title and price is not None and not range_price:
            rows.append(dict(title=title, sold_price_usd=price, sold_date=date, shipping_usd=ship))
        i = j
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paste", help="text file with the pasted page")
    ap.add_argument("--card-id", default="", help="card you searched for (audit only)")
    ap.add_argument("--out", default="data/ebay_sold.csv")
    args = ap.parse_args()

    rows = parse(Path(args.paste).read_text(encoding="utf-8"))
    new = pd.DataFrame(rows, columns=["title", "sold_price_usd", "sold_date", "shipping_usd"])
    if new.empty:
        print("0 listings parsed: the pasted text does not match the expected layout "
              "(see the docstring). Nothing written.")
        return
    new["target_card_id"] = args.card_id
    new["image_file"] = ""
    # the page text has no item number, so make a stable id from the content
    new["item_id"] = ["p" + hashlib.sha1(f"{t}|{p}|{d}".encode()).hexdigest()[:12]
                      for t, p, d in zip(new.title, new.sold_price_usd, new.sold_date)]
    cols = ["item_id", "title", "sold_price_usd", "sold_date", "shipping_usd",
            "image_file", "target_card_id"]
    new = new[cols]

    out = Path(args.out)
    if out.exists():
        old = pd.read_csv(out, dtype=str)
        new = new[~new.item_id.isin(old.item_id)]
        new = pd.concat([old, new.astype(str)], ignore_index=True)
    out.parent.mkdir(exist_ok=True)
    new.to_csv(out, index=False)
    print(f"parsed {len(rows)} listings from {args.paste}; {args.out} now has {len(new)} rows")


if __name__ == "__main__":
    main()
