"""Cross-platform paths shared by collection and modelling scripts."""
from urllib.parse import quote


def card_image_filename(card_id):
    """Return a reversible, Windows-safe PNG filename for a card ID.

    Most IDs are unchanged. Reserved filename characters such as the question
    mark in ``ex10-?`` are percent-encoded (``ex10-%3F.png``).
    """
    return f"{quote(str(card_id), safe='-_.')}.png"

