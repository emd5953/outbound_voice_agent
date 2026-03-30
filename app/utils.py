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
        "get started",
        "ready to order",
        "go ahead",
        "hello",
        "welcome",
    ]

    # IVR / hold phrases to IGNORE (not a human pickup)
    ignore_patterns = [
        "please hold",
        "connect you",
        "team member",
        "hold while",
        "transferring",
        "one moment",
        "press 1",
        "press 2",
        "press 3",
        "say the name",
        "callback number",
        "zip code",
        "is that correct",
        "got it",
        "say yes",
        "say no",
        "start over",
        "to confirm",
        "please call back",
        "try again",
    ]

    for pattern in ignore_patterns:
        if pattern in text:
            return False

    for pattern in greeting_patterns:
        if pattern in text:
            return True

    # Fallback: 4+ words likely means human speech, not hold music
    if len(text.split()) >= 4:
        return True

    return False


def extract_price(text: str) -> Optional[float]:
    """Extract a price from text, matching $XX.XX, XX.XX, or word-form prices.

    Handles both digit patterns ("$18.50", "18.50") and word-form prices
    from STT transcription ("eighteen fifty", "twenty eight ninety eight").

    Args:
        text: Raw text that may contain a price (e.g., employee speech transcript).

    Returns:
        The numeric float value of the first price found, or None if no price is present.
    """
    # Try digit patterns first
    match = re.search(r"\$?(\d+\.\d{2})\b", text)
    if match:
        return float(match.group(1))

    # Word-form number mapping
    _ONES = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19,
    }
    _TENS = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    }

    def _word_to_num(word: str) -> Optional[int]:
        """Convert a single number word or compound like 'twenty eight' to int."""
        word = word.strip().lower()
        if word in _ONES:
            return _ONES[word]
        if word in _TENS:
            return _TENS[word]
        # Try compound: "twenty eight"
        parts = word.split()
        if len(parts) == 2 and parts[0] in _TENS and parts[1] in _ONES:
            return _TENS[parts[0]] + _ONES[parts[1]]
        return None

    def _words_to_price(text_lower: str) -> Optional[float]:
        """Try to extract a dollars.cents price from word-form numbers in text."""
        # Tokenize and convert each word to a number
        words = re.findall(r"[a-z]+", text_lower)
        nums = []
        i = 0
        while i < len(words):
            # Try two-word compound first (e.g. "twenty eight")
            if i + 1 < len(words):
                two_word = words[i] + " " + words[i + 1]
                val = _word_to_num(two_word)
                if val is not None:
                    nums.append(val)
                    i += 2
                    continue
            # Single word
            val = _word_to_num(words[i])
            if val is not None:
                nums.append(val)
            else:
                # Non-number word breaks a sequence — if we have nums, try to parse
                if len(nums) >= 2:
                    dollars, cents = nums[-2], nums[-1]
                    if 0 < cents <= 99:
                        return round(dollars + cents / 100.0, 2)
                nums = []
            i += 1

        # Check remaining nums at end of string
        if len(nums) >= 2:
            dollars, cents = nums[-2], nums[-1]
            if 0 < cents <= 99:
                return round(dollars + cents / 100.0, 2)

        return None

    text_lower = text.lower()
    result = _words_to_price(text_lower)
    if result is not None:
        return result

    # Try single number with "dollars" or "bucks"
    all_words = list(_ONES.keys()) + list(_TENS.keys())
    word_pattern = "|".join(sorted(all_words, key=len, reverse=True))
    compound = rf"(?:(?:{word_pattern})\s+(?:{word_pattern})|(?:{word_pattern}))"
    single_pattern = rf"({compound})\s+(?:dollars?|bucks?)"
    m = re.search(single_pattern, text_lower)
    if m:
        val = _word_to_num(m.group(1))
        if val is not None:
            return float(val)

    return None
