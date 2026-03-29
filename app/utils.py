# Core utility functions: zip extraction, topping evaluation, budget evaluation, hold detection, price extraction

import re
from typing import Optional

from app.models import PizzaOrder


def extract_zip_code(delivery_address: str) -> str:
    """Extract a 5-digit zip code from a delivery address string.

    Args:
        delivery_address: A string containing a delivery address with an embedded 5-digit zip code.

    Returns:
        A string of exactly 5 digits representing the zip code.

    Raises:
        ValueError: If no 5-digit zip code is found in the address.
    """
    match = re.search(r"\b(\d{5})\b", delivery_address)
    if not match:
        raise ValueError("No 5-digit zip code found in delivery address")
    return match.group(1)


def evaluate_topping_substitution(offered_topping: str, order: PizzaOrder) -> str:
    """Decide whether to accept or reject an offered topping substitution.

    Checks the blacklist (no_go_toppings) first, then the whitelist
    (acceptable_topping_subs). Anything not on either list is rejected.
    Matching is case-insensitive with substring support.

    Args:
        offered_topping: The topping name offered by the restaurant employee.
        order: The PizzaOrder containing whitelist and blacklist.

    Returns:
        "reject" if the topping is blacklisted or not on the whitelist,
        "accept" if the topping is on the whitelist.
    """
    offered = offered_topping.lower().strip()

    # Check blacklist first
    for no_go in order.no_go_toppings:
        if no_go.lower() in offered or offered in no_go.lower():
            return "reject"

    # Check whitelist
    for acceptable in order.acceptable_topping_subs:
        if acceptable.lower() in offered or offered in acceptable.lower():
            return "accept"

    # Not on either list — reject
    return "reject"


def evaluate_budget(
    running_total: float,
    next_item_price: float,
    budget_max: float,
    is_drink: bool,
    skip_drink_if_over: bool,
) -> str:
    """Determine budget action for the next item.

    Args:
        running_total: Current accumulated total (>= 0).
        next_item_price: Price of the item to add (> 0).
        budget_max: Maximum allowed budget (> 0).
        is_drink: Whether the item is a drink.
        skip_drink_if_over: Whether to skip the drink instead of going over budget.

    Returns:
        "proceed" if total stays within budget,
        "skip_drink" if drink would exceed budget and skip flag is true,
        "over_budget" otherwise.
    """
    if running_total + next_item_price <= budget_max:
        return "proceed"

    if is_drink and skip_drink_if_over:
        return "skip_drink"

    return "over_budget"


def detect_hold_end(transcript: str) -> bool:
    """Detect when a human employee picks up after hold.

    Args:
        transcript: Raw transcript text from Deepgram STT (may include hold music artifacts).

    Returns:
        True if the transcript indicates a human employee has picked up,
        False for hold music noise, short artifacts, or silence.
    """
    text = transcript.lower().strip()

    # Ignore short transcripts (STT noise / hold music artifacts)
    if len(text) < 5:
        return False

    # Known conversational greeting patterns indicating a human picked up
    greeting_patterns = [
        "what can i get",
        "how can i help",
        "thanks for calling",
        "thanks for holding",
        "what would you like",
        "hey there",
        "hi there",
        "what can i do for you",
    ]

    for pattern in greeting_patterns:
        if pattern in text:
            return True

    # Fallback: 4+ words likely means human speech, not hold music
    if len(text.split()) >= 4:
        return True

    return False


def extract_price(text: str) -> Optional[float]:
    """Extract a price from text, matching $XX.XX or XX.XX patterns.

    Args:
        text: Raw text that may contain a price (e.g., employee speech transcript).

    Returns:
        The numeric float value of the first price found, or None if no price is present.
    """
    match = re.search(r"\$?(\d+\.\d{2})\b", text)
    if not match:
        return None
    return float(match.group(1))
