# Call orchestrator: main state machine driver wiring all components together

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional

from app.models import CallResult, OrderContext, OrderPayload
from app.state_machine import CallState, ConversationPhase, handle_ivr_transcript
from app.utils import detect_hold_end, evaluate_budget, evaluate_topping_substitution, extract_price

if TYPE_CHECKING:
    from app.conversation_engine import ConversationEngine
    from app.media_handler import MediaStreamHandler

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# TTS text normalization — make numbers sound natural over the phone
# ------------------------------------------------------------------

import re as _re

_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

_ONES_W = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS_W = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _int_to_words(n: int) -> str:
    """Convert an integer 0-9999 to natural English words."""
    if n < 0:
        return "negative " + _int_to_words(-n)
    if n < 20:
        return _ONES_W[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS_W[t] + (" " + _ONES_W[o] if o else "")
    if n < 1000:
        h, rem = divmod(n, 100)
        return _ONES_W[h] + " hundred" + (" " + _int_to_words(rem) if rem else "")
    if n < 10000:
        th, rem = divmod(n, 1000)
        return _ONES_W[th] + " thousand" + (" " + _int_to_words(rem) if rem else "")
    return str(n)


def _price_to_words(match: _re.Match) -> str:
    """Convert $XX.XX or XX.XX price to natural speech like 'eighteen fifty'."""
    val = match.group(0).lstrip("$")
    try:
        dollars, cents = val.split(".")
        d = int(dollars)
        c = int(cents)
        if c == 0:
            return _int_to_words(d) + " dollars"
        return _int_to_words(d) + " " + _int_to_words(c)
    except (ValueError, IndexError):
        return match.group(0)


def _phone_to_words(match: _re.Match) -> str:
    """Convert 10-digit phone number to spoken digit groups."""
    digits = match.group(0)
    # "512-555-0147" → "five one two, five five five, zero one four seven"
    clean = _re.sub(r"\D", "", digits)
    if len(clean) == 10:
        parts = [clean[:3], clean[3:6], clean[6:]]
        return ", ".join(" ".join(_DIGIT_WORDS[d] for d in part) for part in parts)
    # Fallback: spell out each digit
    return " ".join(_DIGIT_WORDS.get(d, d) for d in clean)


def _normalize_for_tts(text: str) -> str:
    """Normalize text for natural TTS pronunciation.

    - $18.50 → "eighteen fifty"
    - 2L → "two liter"
    - 12 count → "twelve count"
    - Phone numbers → digit-by-digit groups
    - Order numbers like 4412 → "forty four twelve"
    """
    # Prices: $18.50 or 18.50
    text = _re.sub(r"\$?\d+\.\d{2}\b", _price_to_words, text)

    # Volume: 2L, 1L
    text = _re.sub(r"\b2L\b", "two liter", text)
    text = _re.sub(r"\b1L\b", "one liter", text)

    # Crust types: remove hyphens that TTS might mangle
    text = text.replace("hand-tossed", "hand tossed")
    text = text.replace("Hand-tossed", "hand tossed")

    # Counts: 12 count, 6 count, etc.
    def _count_to_words(m: _re.Match) -> str:
        return _int_to_words(int(m.group(1))) + " count"
    text = _re.sub(r"\b(\d+)\s*count\b", _count_to_words, text, flags=_re.IGNORECASE)

    # Phone numbers (10 digits, possibly with dashes/spaces)
    text = _re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", _phone_to_words, text)

    # Standalone 3-4 digit numbers (likely order numbers) — say naturally
    # e.g. 4412 → "forty four twelve", 512 → "five twelve"
    def _short_num_to_words(m: _re.Match) -> str:
        n = int(m.group(0))
        if 100 <= n <= 9999:
            # Split into two-digit pairs for natural reading
            # 4412 → "forty four twelve", 2305 → "twenty three oh five"
            s = m.group(0)
            if len(s) == 4:
                hi = int(s[:2])
                lo = int(s[2:])
                hi_w = _int_to_words(hi)
                if lo == 0:
                    lo_w = "hundred"
                elif lo < 10:
                    lo_w = "oh " + _int_to_words(lo)
                else:
                    lo_w = _int_to_words(lo)
                return hi_w + " " + lo_w
            elif len(s) == 3:
                hi = int(s[0])
                lo = int(s[1:])
                if lo < 10:
                    return _int_to_words(hi) + " oh " + _int_to_words(lo)
                return _int_to_words(hi) + " " + _int_to_words(lo)
        return _int_to_words(n)

    # Only convert standalone numbers not already handled (not part of prices/phones)
    text = _re.sub(r"(?<!\d)\b(\d{3,4})\b(?!\d|\.)", _short_num_to_words, text)

    # 5-digit zip codes — read as digit groups: 78745 → "seven eight seven four five"
    def _zip_to_words(m: _re.Match) -> str:
        return " ".join(_DIGIT_WORDS[d] for d in m.group(0))
    text = _re.sub(r"(?<!\d)\b(\d{5})\b(?!\d)", _zip_to_words, text)

    return text


# IVR states where handle_ivr_transcript should be called
_IVR_STATES = frozenset(
    {
        CallState.IVR_WELCOME,
        CallState.IVR_NAME,
        CallState.IVR_CALLBACK,
        CallState.IVR_ZIPCODE,
        CallState.IVR_CONFIRM,
    }
)


class CallOrchestrator:
    """Finite state machine driving the entire call lifecycle.

    Manages IVR navigation, hold detection, conversation routing,
    and structured call termination.
    """

    def __init__(self, order: OrderPayload, context: OrderContext) -> None:
        self.order = order
        self.context = context
        self.state = CallState.INITIATING
        self.conversation_phase = ConversationPhase.GREETING
        self.ivr_retry_count: int = 0
        self.transcript_history: list[dict[str, str]] = []
        self.running_total: float = 0.0

        # External component references (set after construction)
        self._media_handler: Optional[MediaStreamHandler] = None
        self._conversation_engine: Optional[ConversationEngine] = None
        self._twilio_client: Any = None

        # Debounce: accumulate transcripts before responding
        self._pending_transcript: str = ""
        self._debounce_task: Optional[asyncio.Task] = None
        self._DEBOUNCE_SECONDS: float = 0.4  # wait for employee to finish

        # Hold cooldown: ignore transcripts briefly after entering hold
        self._hold_cooldown_until: float = 0.0
        self.call_sid: str = ""
        self.stream_sid: str = ""

        # Call completion signalling
        self._completion_event = asyncio.Event()
        self._call_result: Optional[CallResult] = None

    # ------------------------------------------------------------------
    # Component wiring
    # ------------------------------------------------------------------

    def set_media_handler(self, handler: MediaStreamHandler) -> None:
        """Attach the media stream handler after construction."""
        self._media_handler = handler

    def set_conversation_engine(self, engine: ConversationEngine) -> None:
        """Attach the conversation engine after construction."""
        self._conversation_engine = engine

    def set_twilio_client(self, client: Any) -> None:
        """Attach the Twilio REST client for DTMF operations."""
        self._twilio_client = client

    # ------------------------------------------------------------------
    # Transcript routing
    # ------------------------------------------------------------------

    async def handle_transcript(self, text: str) -> None:
        """Route an incoming transcript based on the current call state.

        - IVR states → deterministic IVR handler (no debounce)
        - ON_HOLD   → hold-end detection
        - CONVERSATION → debounced: accumulate text, wait for silence, then process
        """
        if self.state in _IVR_STATES:
            # IVR needs immediate responses — no debounce
            self.transcript_history.append({"role": "employee", "text": text})
            await handle_ivr_transcript(self, text)

        elif self.state == CallState.ON_HOLD:
            # Ignore transcripts during cooldown (IVR may still be finishing)
            import time
            if time.monotonic() < self._hold_cooldown_until:
                return
            self.transcript_history.append({"role": "employee", "text": text})
            if detect_hold_end(text):
                logger.info("Human detected — transitioning to CONVERSATION")
                self.state = CallState.CONVERSATION
                await self._debounced_conversation(text)

        elif self.state == CallState.CONVERSATION:
            await self._debounced_conversation(text)

    async def _debounced_conversation(self, text: str) -> None:
        """Accumulate transcript text and wait for a pause before responding.
        
        This prevents the agent from cutting off the employee mid-sentence
        when Deepgram sends multiple is_final chunks for one utterance.
        """
        if text.strip():
            if self._pending_transcript:
                self._pending_transcript += " " + text.strip()
            else:
                self._pending_transcript = text.strip()

        # Cancel any existing debounce timer
        if self._debounce_task is not None and not self._debounce_task.done():
            self._debounce_task.cancel()

        # Start a new timer
        self._debounce_task = asyncio.create_task(self._debounce_fire())

    async def _debounce_fire(self) -> None:
        """Wait for silence, then process the accumulated transcript."""
        await asyncio.sleep(self._DEBOUNCE_SECONDS)
        
        transcript = self._pending_transcript.strip()
        self._pending_transcript = ""
        
        if not transcript:
            return
        
        self.transcript_history.append({"role": "employee", "text": transcript})
        await self.handle_conversation(transcript)

    # ------------------------------------------------------------------
    # Conversation handler (stub — implemented in task 9.2)
    # ------------------------------------------------------------------

    async def handle_conversation(self, transcript: str) -> None:
        """Handle a transcript during the CONVERSATION state.

        Implements bot detection, price extraction, topping substitution,
        side fallback, drink budget logic, and LLM-driven responses.
        """
        # Ensure transcript is in history (may already be added by debounce)
        if not any(
            e["role"] == "employee" and e["text"] == transcript
            for e in self.transcript_history[-3:]
        ):
            self.transcript_history.append({"role": "employee", "text": transcript})

        phase = self.conversation_phase
        order = self.order
        ctx = self.context
        text_lower = transcript.lower()

        # --- Bot detection ---
        bot_patterns = [
            "are you a robot",
            "is this a bot",
            "is this automated",
            "am i talking to a computer",
            "is this a real person",
            "are you real",
            "you sound like a robot",
            "you a bot",
        ]
        if any(p in text_lower for p in bot_patterns):
            await self.speak("Sorry about that, have a good one.")
            await self.end_call("detected_as_bot")
            return

        # --- Price extraction ---
        price = extract_price(transcript)
        if price is not None:
            self.record_price(phase, price)

        # --- Extract delivery time ---
        if ctx.delivery_time is None:
            dt = self._extract_delivery_time(text_lower)
            if dt:
                ctx.delivery_time = dt
                logger.info("Extracted delivery time: %s", dt)

        # --- Extract order number ---
        if ctx.order_number is None:
            on = self._extract_order_number(text_lower)
            if on:
                ctx.order_number = on
                logger.info("Extracted order number: %s", on)

        # --- Over-budget check for pizza + side ---
        if (
            price is not None
            and phase in (ConversationPhase.PIZZA_ORDER, ConversationPhase.SIDE_ORDER)
            and ctx.running_total > order.budget_max
        ):
            await self.speak("Ah shoot, that's over my budget. Sorry about that, I'll have to cancel. Thanks though.")
            await self.end_call("over_budget")
            return

        # --- Phase-specific business logic ---
        response: str | None = None

        if phase == ConversationPhase.PIZZA_ORDER:
            # Check if the pizza itself can't be ordered (core item unavailable)
            nothing_available_kw = [
                "we can't make that",
                "we don't have pizza",
                "no pizza",
                "can't do pizza",
                "don't sell pizza",
                "we don't make pizza",
                "can't make any pizza",
                "we're closed",
                "can't do that order",
            ]
            if any(kw in text_lower for kw in nothing_available_kw):
                await self.speak("Ah no worries. Thanks anyway, have a good one.")
                await self.end_call("nothing_available")
                return

            # Check for topping unavailability and offered substitution
            unavailable_kw = ["out of", "don't have", "unavailable", "we're out", "we ran out"]
            offer_kw = ["how about", "we could do", "what about", "instead", "want"]
            if any(kw in text_lower for kw in unavailable_kw):
                has_offer = any(kw in text_lower for kw in offer_kw)
                if has_offer:
                    # They're offering a substitute — evaluate it
                    decision = evaluate_topping_substitution(transcript, order.pizza)
                    if decision == "accept":
                        response = await self._llm_response(
                            transcript, "Accept the offered substitute. Say 'yeah that works' or similar."
                        )
                    elif decision == "reject":
                        response = await self._llm_response(
                            transcript,
                            f"Decline what they offered and ask what else they have. "
                            f"You'd accept any of these: {', '.join(order.pizza.acceptable_topping_subs)}. "
                            f"Never accept: {', '.join(order.pizza.no_go_toppings)}. "
                            f"Do NOT suggest a specific topping — just ask what they have.",
                        )
                else:
                    # They just said something is unavailable, no offer yet
                    # DON'T suggest a sub — ask what they can do instead
                    response = await self._llm_response(
                        transcript,
                        "A topping is unavailable. Ask what they can do instead. "
                        "Do NOT suggest a specific replacement yourself. Just ask 'what else you got?' or similar.",
                    )
            elif any(kw in text_lower for kw in offer_kw):
                # Employee offering a substitute without the unavailable keyword
                decision = evaluate_topping_substitution(transcript, order.pizza)
                if decision == "accept":
                    response = await self._llm_response(
                        transcript, "Accept the offered substitute. Say 'yeah that works' or similar."
                    )
                else:
                    response = await self._llm_response(
                        transcript,
                        f"Decline and ask what else they have. "
                        f"You'd accept: {', '.join(order.pizza.acceptable_topping_subs)}. "
                        f"Do NOT suggest a specific topping.",
                    )

        elif phase == ConversationPhase.SIDE_ORDER:
            unavailable_kw = ["out of", "don't have", "unavailable", "we're out", "no wings", "no buffalo"]
            offered_kw = ["we do have", "we got", "we have", "how about", "what about"]

            has_unavailable = any(kw in text_lower for kw in unavailable_kw)
            has_offer = any(kw in text_lower for kw in offered_kw)

            if has_unavailable and has_offer:
                # Employee says something is unavailable but offers an alternative
                offered_backup = None
                for backup in order.side.backup_options:
                    if backup.lower() in text_lower:
                        # Check if the backup appears after an offer keyword
                        for ok in offered_kw:
                            if ok in text_lower:
                                ok_pos = text_lower.index(ok)
                                bp_pos = text_lower.index(backup.lower())
                                if bp_pos > ok_pos:
                                    offered_backup = backup
                                    break
                    if offered_backup:
                        break

                if offered_backup:
                    ctx.side_original = order.side.first_choice
                    ctx.side_description = offered_backup
                    response = await self._llm_response(
                        transcript, f"Accept {offered_backup}. Say 'yeah that works' or 'sure'."
                    )
                else:
                    # All mentioned items are unavailable
                    remaining = [b for b in order.side.backup_options if b.lower() not in text_lower]
                    if remaining:
                        response = await self._llm_response(
                            transcript, f"Ask for {remaining[0]} instead."
                        )
                    elif order.side.if_all_unavailable == "skip":
                        ctx.side_skipped = True
                        response = await self._llm_response(
                            transcript, "All side options unavailable. Skip the side and move on."
                        )

            elif has_unavailable:
                # Everything mentioned is unavailable — try backups not mentioned
                remaining_backups = [b for b in order.side.backup_options if b.lower() not in text_lower]
                if remaining_backups:
                    response = await self._llm_response(
                        transcript, f"Ask for {remaining_backups[0]} instead."
                    )
                else:
                    if order.side.if_all_unavailable == "skip":
                        ctx.side_skipped = True
                        response = await self._llm_response(
                            transcript, "All side options unavailable. Skip the side and move on."
                        )

            elif has_offer:
                # Employee offering something — check if it's a backup
                for backup in order.side.backup_options:
                    if backup.lower() in text_lower:
                        ctx.side_original = order.side.first_choice
                        ctx.side_description = backup
                        response = await self._llm_response(
                            transcript, f"Accept {backup}. Say 'yeah that works' or similar."
                        )
                        break

        elif phase == ConversationPhase.DRINK_ORDER:
            if price is not None:
                budget_action = evaluate_budget(
                    ctx.running_total - price,  # price already recorded above
                    price,
                    order.budget_max,
                    is_drink=True,
                    skip_drink_if_over=order.drink.skip_if_over_budget,
                )
                if budget_action == "skip_drink":
                    ctx.drink_skipped = True
                    ctx.drink_skip_reason = "over_budget"
                    ctx.drink_price = None
                    ctx.running_total -= price
                    response = await self._llm_response(
                        transcript,
                        "Actually skip the drink — it's over budget. Say 'actually never mind on the drink, that's everything'.",
                    )
                elif budget_action == "over_budget":
                    await self.speak("Actually that's gonna be over my budget. Sorry, I'll have to pass. Thanks though.")
                    await self.end_call("over_budget")
                    return

            # Check for drink unavailability
            unavailable_kw = ["out of", "don't have", "unavailable", "we're out"]
            if any(kw in text_lower for kw in unavailable_kw):
                # Try alternatives in order
                found_alt = False
                for alt in order.drink.alternatives:
                    if alt.lower() not in text_lower:
                        response = await self._llm_response(
                            transcript, f"Ask for {alt} instead."
                        )
                        found_alt = True
                        break
                if not found_alt:
                    ctx.drink_skipped = True
                    ctx.drink_skip_reason = "all_unavailable"

        elif phase == ConversationPhase.PRICE_COLLECTION:
            # Extract total if mentioned
            if price is not None and ctx.running_total == 0:
                ctx.running_total = price

            # Check if we have everything we need — but DON'T deliver special
            # instructions here. Let the phase advance to SPECIAL_INSTRUCTIONS
            # where we handle it deterministically.

        elif phase == ConversationPhase.SPECIAL_INSTRUCTIONS:
            if not ctx.special_instructions_delivered:
                # Deliver special instructions via LLM for natural delivery
                response = await self._llm_response(
                    transcript,
                    f"You MUST mention the delivery instructions right now: \"{order.special_instructions}\". "
                    f"Say it casually like 'oh one more thing — {order.special_instructions}'. "
                    f"Then wait for them to acknowledge.",
                )
                ctx.special_instructions_delivered = True
            else:
                # Employee acknowledged — say thanks and hang up
                response = await self._llm_response(
                    transcript, "Say thanks and have a good one. Keep it short."
                )
                if response:
                    self.transcript_history.append({"role": "agent", "text": response})
                await self.end_call("completed")
                return

        elif phase == ConversationPhase.CLOSING:
            response = await self._llm_response(transcript, "Say thanks and bye. Keep it short.")
            self.transcript_history.append({"role": "agent", "text": response})
            await self.end_call("completed")
            return

        # --- "Anything else?" handling ---
        anything_else_kw = ["anything else", "that it", "that everything", "that all", "is that all"]
        if any(kw in text_lower for kw in anything_else_kw):
            # Check what we still need based on CURRENT phase (after possible advancement above)
            current_phase = self.conversation_phase
            if current_phase == ConversationPhase.SIDE_ORDER and ctx.side_price is None and not ctx.side_skipped:
                side_desc = order.side.first_choice.replace("12 count", "twelve count").replace("6 count", "six count")
                response = await self._llm_response(
                    transcript,
                    f"Yes, also order: {side_desc}. Ask for the price.",
                )
            elif current_phase == ConversationPhase.DRINK_ORDER and ctx.drink_price is None and not ctx.drink_skipped:
                drink_desc = order.drink.first_choice.replace("2L", "two liter").replace("1L", "one liter")
                response = await self._llm_response(
                    transcript,
                    f"Yes, also a {drink_desc}.",
                )
            elif current_phase in (ConversationPhase.PRICE_COLLECTION, ConversationPhase.SPECIAL_INSTRUCTIONS):
                response = await self._llm_response(
                    transcript,
                    "That's everything. Ask what the total is, how long for delivery, and the order number.",
                )
            elif phase == ConversationPhase.PIZZA_ORDER and ctx.pizza_price is not None:
                # Pizza done, need to order side
                side_desc = order.side.first_choice.replace("12 count", "twelve count").replace("6 count", "six count")
                response = await self._llm_response(
                    transcript,
                    f"Yes, also order: {side_desc}. Ask for the price.",
                )
            elif phase == ConversationPhase.SIDE_ORDER and (ctx.side_price is not None or ctx.side_skipped):
                # Side done, need drink
                drink_desc = order.drink.first_choice.replace("2L", "two liter").replace("1L", "one liter")
                response = await self._llm_response(
                    transcript,
                    f"Yes, also a {drink_desc}.",
                )
            elif phase == ConversationPhase.DRINK_ORDER and (ctx.drink_price is not None or ctx.drink_skipped):
                # All items ordered
                response = await self._llm_response(
                    transcript,
                    "That's everything. Ask what the total is.",
                )
            elif ctx.pizza_price is not None and (ctx.side_price is not None or ctx.side_skipped) and (ctx.drink_price is not None or ctx.drink_skipped):
                response = await self._llm_response(
                    transcript,
                    "That's everything. Ask what the total is, how long for delivery, and the order number.",
                )

        # --- Default: LLM response ---
        if response is None:
            response = await self._llm_response(transcript)

        # _llm_response now streams directly to TTS, so only use speak()
        # for responses that were built without _llm_response (none currently).
        # Log the full response to transcript history.
        if response:
            self.transcript_history.append({"role": "agent", "text": response})

        # --- Phase advancement ---
        await self.maybe_advance_phase()

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_delivery_time(text_lower: str) -> str | None:
        """Extract delivery time estimate from employee speech."""
        import re

        # Word-form numbers for STT transcripts
        _NUM_WORDS = {
            "five": "5", "ten": "10", "fifteen": "15", "twenty": "20",
            "twenty five": "25", "thirty": "30", "thirty five": "35",
            "forty": "40", "forty five": "45", "fifty": "50",
            "fifty five": "55", "sixty": "60",
        }

        # "35-40 minutes", "35 to 40 minutes"
        m = re.search(r"(\d+\s*(?:to|-)\s*\d+)\s*minutes?", text_lower)
        if m:
            return m.group(0).strip()
        # "35 minutes", "about 35 minutes"
        m = re.search(r"(\d+)\s*minutes?", text_lower)
        if m:
            return m.group(0).strip()

        # Word-form: "thirty five minutes", "about twenty minutes"
        # Try two-word compounds first, then single words
        for word, digit in sorted(_NUM_WORDS.items(), key=lambda x: -len(x[0])):
            pattern = rf"(?:about\s+)?{word}\s+minutes?"
            m = re.search(pattern, text_lower)
            if m:
                return m.group(0).strip()

        # "about an hour", "an hour", "an hour and a half"
        if "hour" in text_lower:
            m = re.search(r"(?:about\s+)?(?:an?\s+)?hour(?:\s+and\s+a\s+half)?", text_lower)
            if m:
                return m.group(0).strip()
        return None

    @staticmethod
    def _extract_order_number(text_lower: str) -> str | None:
        """Extract order/confirmation number from employee speech.
        
        Only extracts when the transcript explicitly mentions 'order number'
        or 'confirmation number' to avoid false positives from prices.
        """
        import re

        # Must have explicit "order number" or "confirmation number" context
        if "order" not in text_lower and "confirmation" not in text_lower:
            return None

        # Direct digit pattern: "order number is 4412", "order number 4412"
        m = re.search(r"(?:order|confirmation)\s*(?:number|#|num)?\s*(?:is\s+)?(\d+)", text_lower)
        if m:
            return m.group(1)

        # "number is 4412" after order/confirmation context
        m = re.search(r"number\s+(?:is\s+)?(\d+)", text_lower)
        if m:
            return m.group(1)

        # Word-form number mapping
        _ONES = {
            "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
            "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
            "eighteen": 18, "nineteen": 19,
        }
        _TENS = {
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        }

        def _word_to_num(word: str) -> int | None:
            w = word.strip().lower()
            if w in _ONES:
                return _ONES[w]
            if w in _TENS:
                return _TENS[w]
            parts = w.split()
            if len(parts) == 2 and parts[0] in _TENS and parts[1] in _ONES:
                return _TENS[parts[0]] + _ONES[parts[1]]
            return None

        # Try to find word-form numbers after "order/confirmation number"
        m = re.search(r"(?:order|confirmation)\s*(?:number|#|num)?\s*(?:is\s+)?(.*)", text_lower)
        remainder = m.group(1).strip() if m else text_lower

        words = re.findall(r"[a-z]+", remainder)
        nums = []
        i = 0
        while i < len(words):
            if i + 1 < len(words):
                two_word = words[i] + " " + words[i + 1]
                val = _word_to_num(two_word)
                if val is not None:
                    nums.append(val)
                    i += 2
                    continue
            val = _word_to_num(words[i])
            if val is not None:
                nums.append(val)
                i += 1
            else:
                if nums:
                    break  # stop at first non-number word
                i += 1
        if nums:
            return "".join(str(n) for n in nums)

        return None

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    async def _llm_response(
        self, employee_text: str, extra_instruction: str | None = None
    ) -> str:
        """Generate an LLM response via the conversation engine, streaming
        each sentence fragment through TTS as it arrives.

        Returns the full concatenated response text for transcript logging.
        """
        if self._conversation_engine is None:
            return ""

        full_text = ""
        async for fragment in self._conversation_engine.generate_response_streamed(
            employee_text=employee_text,
            phase=self.conversation_phase,
            context=self.context,
            transcript_history=self.transcript_history,
            extra_instruction=extra_instruction,
        ):
            full_text += fragment
            # Stream each fragment to TTS immediately — don't wait for full response
            await self._speak_raw(fragment)

        return full_text

    async def _speak_raw(self, text: str) -> None:
        """Synthesize and send a text fragment without adding to transcript.

        Used by the streaming pipeline to send individual chunks. The caller
        is responsible for logging the full response to transcript_history.
        """
        from app.media_handler import synthesize_and_send

        text = _normalize_for_tts(text)
        if not text.strip():
            return

        if self._media_handler is not None:
            await synthesize_and_send(
                text=text,
                stream_sid=self._media_handler.stream_sid,
                twilio_ws=self._media_handler.twilio_ws,
            )

    # ------------------------------------------------------------------
    # Price recording
    # ------------------------------------------------------------------

    def record_price(self, phase: ConversationPhase, price: float) -> None:
        """Store price for the current phase and accumulate running_total."""
        ctx = self.context
        if phase == ConversationPhase.PIZZA_ORDER and ctx.pizza_price is None:
            ctx.pizza_price = price
            ctx.running_total += price
        elif phase == ConversationPhase.SIDE_ORDER and ctx.side_price is None:
            ctx.side_price = price
            ctx.running_total += price
        elif phase == ConversationPhase.DRINK_ORDER and ctx.drink_price is None:
            ctx.drink_price = price
            ctx.running_total += price

        # Check budget after recording
        if ctx.running_total > self.order.budget_max:
            # Only flag over_budget for non-drink items (drink has its own logic)
            if phase != ConversationPhase.DRINK_ORDER:
                logger.warning(
                    "Running total $%.2f exceeds budget $%.2f",
                    ctx.running_total,
                    self.order.budget_max,
                )

    # ------------------------------------------------------------------
    # Phase advancement
    # ------------------------------------------------------------------

    async def maybe_advance_phase(self) -> None:
        """Transition to the next conversation phase when conditions are met."""
        phase = self.conversation_phase
        ctx = self.context

        # Count how many employee utterances we've had in conversation
        conv_count = sum(
            1 for e in self.transcript_history
            if e["role"] == "employee" and e.get("text", "").strip()
        )

        if phase == ConversationPhase.GREETING:
            # Advance after the agent has responded to the first greeting.
            # If the greeting itself contained an order prompt (e.g. "Thanks
            # for calling, what can I get started for you?"), skip straight
            # to PIZZA_ORDER so we don't deadlock waiting for the employee
            # to re-ask.
            if conv_count >= 1:
                # Check if any employee utterance already contains an order prompt
                order_keywords = [
                    "what can i get", "what would you like",
                    "what are you looking", "what do you want",
                    "looking to order", "what'll it be",
                    "what can i do for you", "ready to order",
                    "go ahead with your order", "what are we ordering",
                    "start your order", "what do you need",
                    "what'll you have", "get started",
                ]
                last_employee = ""
                for e in reversed(self.transcript_history):
                    if e["role"] == "employee" and e.get("text", "").strip():
                        last_employee = e["text"].lower()
                        break
                if any(kw in last_employee for kw in order_keywords):
                    # Employee already asked what we want — jump to ordering
                    self.conversation_phase = ConversationPhase.PIZZA_ORDER
                else:
                    self.conversation_phase = ConversationPhase.DELIVERY_INFO

        elif phase == ConversationPhase.DELIVERY_INFO:
            last_employee = ""
            for e in reversed(self.transcript_history):
                if e["role"] == "employee" and e.get("text", "").strip():
                    last_employee = e["text"].lower()
                    break
            # Only advance when the employee is clearly asking what they want to order.
            # "order" alone is too broad — matches "name on the order", "order started", etc.
            order_keywords = ["what can i get", "what would you like",
                              "what are you looking", "what do you want", "looking to order",
                              "what'll it be", "what can i do for you",
                              "ready to order", "go ahead with your order",
                              "what are we ordering", "start your order",
                              "what do you need", "what'll you have", "get started"]
            if any(kw in last_employee for kw in order_keywords) or conv_count >= 6:
                self.conversation_phase = ConversationPhase.PIZZA_ORDER

        elif phase == ConversationPhase.PIZZA_ORDER and ctx.pizza_price is not None:
            self.conversation_phase = ConversationPhase.SIDE_ORDER

        elif phase == ConversationPhase.SIDE_ORDER and (
            ctx.side_price is not None or ctx.side_skipped
        ):
            self.conversation_phase = ConversationPhase.DRINK_ORDER

        elif phase == ConversationPhase.DRINK_ORDER and (
            ctx.drink_price is not None or ctx.drink_skipped
        ):
            self.conversation_phase = ConversationPhase.PRICE_COLLECTION

        elif phase == ConversationPhase.PRICE_COLLECTION:
            # Need delivery time AND order number to advance
            if ctx.delivery_time is not None and ctx.order_number is not None:
                self.conversation_phase = ConversationPhase.SPECIAL_INSTRUCTIONS

        elif phase == ConversationPhase.SPECIAL_INSTRUCTIONS:
            if ctx.special_instructions_delivered:
                self.conversation_phase = ConversationPhase.CLOSING

        # --- Completed hangup: CLOSING phase reached ---
        # Don't auto-hangup here — let the CLOSING/SPECIAL_INSTRUCTIONS
        # handler manage the actual hangup after saying goodbye.

    # ------------------------------------------------------------------
    # Actions: DTMF, speak, end call
    # ------------------------------------------------------------------

    async def send_dtmf(self, digits: str) -> None:
        """Send DTMF tones through the audio stream.

        Generates DTMF tone audio and sends it through the Twilio WebSocket
        media stream. This avoids updating the call's TwiML which would
        kill the bidirectional stream.
        """
        import audioop
        import math
        import struct

        logger.info("Sending DTMF: %s", digits)
        self.transcript_history.append({"role": "agent", "text": f"[DTMF: {digits}]"})

        if self._media_handler is None:
            return

        # DTMF frequency pairs
        DTMF_FREQS = {
            "1": (697, 1209), "2": (697, 1336), "3": (697, 1477),
            "4": (770, 1209), "5": (770, 1336), "6": (770, 1477),
            "7": (852, 1209), "8": (852, 1336), "9": (852, 1477),
            "0": (941, 1336), "*": (941, 1209), "#": (941, 1477),
        }

        sample_rate = 8000
        tone_duration = 0.15  # seconds per digit
        gap_duration = 0.05   # gap between digits

        import base64
        import json

        for digit in digits:
            freqs = DTMF_FREQS.get(digit)
            if freqs is None:
                continue

            f1, f2 = freqs
            num_samples = int(sample_rate * tone_duration)
            samples = []
            for i in range(num_samples):
                t = i / sample_rate
                val = 0.5 * (math.sin(2 * math.pi * f1 * t) + math.sin(2 * math.pi * f2 * t))
                # Convert to 16-bit PCM
                samples.append(int(val * 16383))

            # Pack as 16-bit linear PCM
            pcm_data = struct.pack(f"<{len(samples)}h", *samples)
            # Convert to mulaw
            mulaw_data = audioop.lin2ulaw(pcm_data, 2)

            # Send as base64 media event
            audio_b64 = base64.b64encode(mulaw_data).decode("utf-8")
            media_event = json.dumps({
                "event": "media",
                "streamSid": self._media_handler.stream_sid,
                "media": {"payload": audio_b64},
            })
            await self._media_handler.twilio_ws.send_text(media_event)

            # Gap between digits
            if gap_duration > 0:
                gap_samples = int(sample_rate * gap_duration)
                silence = audioop.lin2ulaw(b"\x00\x00" * gap_samples, 2)
                gap_b64 = base64.b64encode(silence).decode("utf-8")
                gap_event = json.dumps({
                    "event": "media",
                    "streamSid": self._media_handler.stream_sid,
                    "media": {"payload": gap_b64},
                })
                await self._media_handler.twilio_ws.send_text(gap_event)

    async def speak(self, text: str) -> None:
        """Synthesize text via Cartesia TTS and send audio to Twilio.

        Normalizes text for natural TTS pronunciation before synthesis.
        Delegates to the media handler's synthesize_and_send function.
        """
        from app.media_handler import synthesize_and_send

        text = _normalize_for_tts(text)
        logger.info("Speaking: %s", text[:80])
        self.transcript_history.append({"role": "agent", "text": text})

        if self._media_handler is not None:
            await synthesize_and_send(
                text=text,
                stream_sid=self._media_handler.stream_sid,
                twilio_ws=self._media_handler.twilio_ws,
            )

    async def end_call(self, outcome: str) -> None:
        """Terminate the call with the given outcome.

        Sets state to HANGUP, builds the CallResult, hangs up the Twilio
        call, and signals completion so wait_for_completion() can return.
        """
        logger.info("Ending call with outcome: %s", outcome)
        self.state = CallState.HANGUP

        # Actually hang up the Twilio call
        if self._twilio_client and self.call_sid:
            try:
                self._twilio_client.calls(self.call_sid).update(status="completed")
                logger.info("Twilio call %s hung up", self.call_sid)
            except Exception as exc:
                logger.warning("Failed to hang up Twilio call: %s", exc)

        self._call_result = self._build_result(outcome)
        self._completion_event.set()

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    async def wait_for_completion(self) -> CallResult:
        """Block until the call ends and return the structured result."""
        await self._completion_event.wait()
        assert self._call_result is not None
        return self._call_result

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _build_result(self, outcome: str) -> CallResult:
        """Assemble a CallResult from whatever data has been collected."""
        ctx = self.context

        pizza_data = None
        if ctx.pizza_ordered or ctx.pizza_price is not None:
            pizza_data = {
                "description": ctx.pizza_description,
                "substitutions": ctx.pizza_substitutions,
                "price": ctx.pizza_price,
            }

        side_data = None
        if ctx.side_ordered or ctx.side_price is not None:
            side_data = {
                "description": ctx.side_description,
                "original_choice": ctx.side_original,
                "price": ctx.side_price,
            }
        elif ctx.side_skipped:
            side_data = {"skipped": True}

        drink_data = None
        if ctx.drink_ordered or ctx.drink_price is not None:
            drink_data = {
                "description": ctx.drink_description,
                "price": ctx.drink_price,
            }
        elif ctx.drink_skipped:
            drink_data = {
                "skipped": True,
                "reason": ctx.drink_skip_reason or "over_budget",
            }

        return CallResult(
            outcome=outcome,
            pizza=pizza_data,
            side=side_data,
            drink=drink_data,
            total=ctx.running_total if ctx.running_total > 0 else None,
            delivery_time=ctx.delivery_time,
            order_number=ctx.order_number,
            special_instructions_delivered=ctx.special_instructions_delivered,
        )
