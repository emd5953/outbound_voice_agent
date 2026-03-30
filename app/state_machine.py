# Call state machine: CallState, ConversationPhase enums and IVR handling

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from app.utils import extract_zip_code

if TYPE_CHECKING:
    from app.call_orchestrator import CallOrchestrator


class CallState(Enum):
    """Finite states for the entire call lifecycle."""

    INITIATING = "initiating"
    IVR_WELCOME = "ivr_welcome"
    IVR_NAME = "ivr_name"
    IVR_CALLBACK = "ivr_callback"
    IVR_ZIPCODE = "ivr_zipcode"
    IVR_CONFIRM = "ivr_confirm"
    ON_HOLD = "on_hold"
    CONVERSATION = "conversation"
    HANGUP = "hangup"


class ConversationPhase(Enum):
    """Phases within the CONVERSATION state, progressed in order."""

    GREETING = "greeting"
    DELIVERY_INFO = "delivery_info"
    PIZZA_ORDER = "pizza_order"
    SIDE_ORDER = "side_order"
    DRINK_ORDER = "drink_order"
    PRICE_COLLECTION = "price_collection"
    SPECIAL_INSTRUCTIONS = "special_instructions"
    CLOSING = "closing"


async def handle_ivr_transcript(
    orchestrator: CallOrchestrator, transcript: str
) -> None:
    """Route IVR prompts to the correct response action.

    Deterministic IVR navigation — no LLM involved.
    Each state expects a specific prompt and responds with the exact bare value.

    Args:
        orchestrator: The call orchestrator holding current state, order data,
                      and retry count.
        transcript: Raw transcript text from Deepgram STT.
    """
    text = transcript.lower().strip()
    state = orchestrator.state
    order = orchestrator.order

    # --- Retry / failure detection (applies to all IVR states) ---
    if "didn't understand" in text or "try again" in text:
        orchestrator.ivr_retry_count += 1
        if orchestrator.ivr_retry_count >= 3:
            await orchestrator.end_call("ivr_failure")
        # Otherwise stay in current state; IVR will repeat the prompt
        return

    if "call back later" in text:
        await orchestrator.end_call("ivr_failure")
        return

    # --- State-specific prompt matching ---
    # Deepgram may transcribe digits as words ("one" not "1") and may
    # split a single IVR prompt across multiple transcript chunks.
    if state == CallState.IVR_WELCOME:
        # IVR may split "Thanks for calling, press 1 for delivery" across
        # multiple transcript chunks. Match broadly on "press 1/one" alone
        # since that's the action we need to take regardless.
        if "press 1" in text or "press one" in text:
            await orchestrator.send_dtmf("1")
            orchestrator.state = CallState.IVR_NAME
        elif "delivery" in text and ("1" in text or "one" in text):
            await orchestrator.send_dtmf("1")
            orchestrator.state = CallState.IVR_NAME

    elif state == CallState.IVR_NAME:
        if "name" in text and ("say" in text or "order" in text):
            await orchestrator.speak(order.customer_name)
            orchestrator.state = CallState.IVR_CALLBACK

    elif state == CallState.IVR_CALLBACK:
        if "callback" in text or "phone number" in text or "digit" in text:
            await orchestrator.send_dtmf(order.phone_number)
            orchestrator.state = CallState.IVR_ZIPCODE

    elif state == CallState.IVR_ZIPCODE:
        if "zip" in text:
            zip_code = extract_zip_code(order.delivery_address)
            await orchestrator.speak(zip_code)
            orchestrator.state = CallState.IVR_CONFIRM

    elif state == CallState.IVR_CONFIRM:
        if "correct" in text or "is that right" in text or "confirm" in text:
            import asyncio
            await asyncio.sleep(0.5)
            await orchestrator.speak("yes")
            orchestrator.state = CallState.ON_HOLD
            import time
            orchestrator._hold_cooldown_until = time.monotonic() + 1.5
        elif "incorrect" in text or "wrong" in text or "start over" in text or "try again" in text:
            # IVR rejected our info — restart from name
            orchestrator.state = CallState.IVR_NAME
