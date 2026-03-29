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
        self._DEBOUNCE_SECONDS: float = 1.5  # wait for employee to finish

        # Twilio call metadata
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
        ]
        if any(p in text_lower for p in bot_patterns):
            await self.speak("Sorry about that, have a good one.")
            await self.end_call("detected_as_bot")
            return

        # --- Price extraction ---
        price = extract_price(transcript)
        if price is not None:
            self.record_price(phase, price)

        # --- Over-budget check for pizza + side ---
        if (
            price is not None
            and phase in (ConversationPhase.PIZZA_ORDER, ConversationPhase.SIDE_ORDER)
            and ctx.running_total > order.budget_max
        ):
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
                "not available",
                "can't make any pizza",
                "we're closed",
                "can't do that order",
            ]
            if any(kw in text_lower for kw in nothing_available_kw):
                await self.end_call("nothing_available")
                return

            unavailable_kw = ["out of", "don't have", "unavailable", "we're out"]
            if any(kw in text_lower for kw in unavailable_kw):
                # Employee says a topping is unavailable — check for offered sub
                decision = evaluate_topping_substitution(transcript, order.pizza)
                if decision == "accept":
                    response = await self._llm_response(
                        transcript, f"Accept the offered substitute."
                    )
                else:
                    response = await self._llm_response(
                        transcript,
                        f"Decline and ask what else they have. "
                        f"Acceptable: {order.pizza.acceptable_topping_subs}",
                    )

        elif phase == ConversationPhase.SIDE_ORDER:
            unavailable_kw = ["out of", "don't have", "unavailable", "we're out"]
            if any(kw in text_lower for kw in unavailable_kw):
                # Try backup options in order
                found_backup = False
                for backup in order.side.backup_options:
                    if backup.lower() not in text_lower:
                        response = await self._llm_response(
                            transcript, f"Ask for {backup} instead."
                        )
                        found_backup = True
                        break
                if not found_backup:
                    if order.side.if_all_unavailable == "skip":
                        ctx.side_skipped = True
                        response = await self._llm_response(
                            transcript, "All side options unavailable. Skip the side."
                        )

        elif phase == ConversationPhase.SPECIAL_INSTRUCTIONS:
            # Deliver special instructions and mark as done
            if not ctx.special_instructions_delivered:
                response = await self._llm_response(
                    transcript,
                    f"Deliver these special instructions: \"{order.special_instructions}\". "
                    f"Then say thanks and goodbye.",
                )
                ctx.special_instructions_delivered = True

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
                    # Undo the recorded drink price
                    ctx.drink_price = None
                    ctx.running_total -= price
                    response = await self._llm_response(
                        transcript,
                        "Actually, let's skip the drink. That's everything.",
                    )
                elif budget_action == "over_budget":
                    await self.end_call("over_budget")
                    return

        # --- Default: LLM response ---
        if response is None:
            response = await self._llm_response(transcript)

        await self.speak(response)

        # --- Phase advancement ---
        await self.maybe_advance_phase()

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    async def _llm_response(
        self, employee_text: str, extra_instruction: str | None = None
    ) -> str:
        """Generate an LLM response via the conversation engine."""
        if self._conversation_engine is None:
            return ""
        return await self._conversation_engine.generate_response(
            employee_text=employee_text,
            phase=self.conversation_phase,
            context=self.context,
            transcript_history=self.transcript_history,
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
        """Transition to the next conversation phase when conditions are met.
        
        GREETING and DELIVERY_INFO advance based on transcript count to avoid
        jumping ahead before the employee has asked for info.
        """
        phase = self.conversation_phase
        ctx = self.context

        # Count how many employee utterances we've had in conversation
        conv_count = sum(
            1 for e in self.transcript_history
            if e["role"] == "employee" and e.get("text", "").strip()
        )

        if phase == ConversationPhase.GREETING:
            # Stay in greeting for the first exchange, then move to delivery info
            if conv_count >= 2:
                self.conversation_phase = ConversationPhase.DELIVERY_INFO

        elif phase == ConversationPhase.DELIVERY_INFO:
            # Stay in delivery info until the employee asks about the order
            # (indicated by enough back-and-forth or keywords)
            last_employee = ""
            for e in reversed(self.transcript_history):
                if e["role"] == "employee" and e.get("text", "").strip():
                    last_employee = e["text"].lower()
                    break
            order_keywords = ["order", "what can i get", "what would you like",
                              "what are you looking", "what do you want", "looking to order"]
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
            if ctx.delivery_time is not None and ctx.order_number is not None:
                self.conversation_phase = ConversationPhase.SPECIAL_INSTRUCTIONS

        elif phase == ConversationPhase.SPECIAL_INSTRUCTIONS:
            if ctx.special_instructions_delivered:
                self.conversation_phase = ConversationPhase.CLOSING

        # --- Completed hangup: CLOSING phase reached ---
        if self.conversation_phase == ConversationPhase.CLOSING:
            await self.end_call("completed")

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

        Delegates to the media handler's synthesize_and_send function.
        """
        from app.media_handler import synthesize_and_send

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

        Sets state to HANGUP, builds the CallResult, and signals
        completion so wait_for_completion() can return.
        """
        logger.info("Ending call with outcome: %s", outcome)
        self.state = CallState.HANGUP

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
