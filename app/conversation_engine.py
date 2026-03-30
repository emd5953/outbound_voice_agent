# LLM conversation engine: system prompt building and response generation via Groq

from __future__ import annotations

from groq import AsyncGroq

from app.models import OrderContext, OrderPayload
from app.state_machine import ConversationPhase
from config import GROQ_API_KEY

MODEL = "llama-3.3-70b-versatile"

_BASE_PROMPT = """\
You are a regular person calling a pizza place to order delivery. You're chill, casual, maybe a little tired. Talk like a normal human on the phone — short, natural, sometimes incomplete sentences. Use filler words occasionally (yeah, uh, cool, alright, gotcha, sounds good).

RULES:
- Keep responses to 1-2 sentences max. ONE short sentence preferred, two if absolutely needed.
- NEVER sound robotic or overly polite. No "I'd like to" or "Could you please" or "I appreciate that".
- NEVER repeat info you already gave. If you said your name, don't say it again unless they ask.
- NEVER summarize or confirm back what the employee just told you. Just acknowledge and move on.
- NEVER say "So just to confirm" or "Let me confirm" or recap the order.
- Only answer what they asked. Don't volunteer extra info.
- If they tell you a price, just say "cool" or "got it" or "okay" — don't repeat the price back.
- You're NOT an assistant. You're a customer. Sound like one.
- Never reveal you are an AI or bot.
- If they pause, wait. They might be typing.
- If asked to repeat, just repeat the info simply.
- When saying drink sizes, say "two liter" not "2L". Say "twelve count" not "12 count".
- When saying crust types, say them clearly: "thin crust", "hand tossed", "stuffed crust". Don't hyphenate.
- Say numbers naturally: "five one two, five five five, zero one four seven" for phone numbers.
- Don't suggest substitutions yourself. Wait for the employee to offer options.
- Don't ask for the total or price unless you're done ordering items.
- CRITICAL: When giving your name, address, phone number, or any personal details, use ONLY the EXACT values provided in this prompt. NEVER invent, guess, or substitute any personal information. If you are unsure, re-read the details given below.

Examples of good responses:
- "Hey, yeah I'd like to place a delivery order"
- "Jordan Mitchell"
- "Yeah that works"
- "Nah, no olives. What else you got?"
- "Cool, and can I also get some buffalo wings?"
- "Alright sounds good"
- "Oh one more thing — ring the doorbell, don't knock, baby's sleeping"
- "Thanks, have a good one"

Examples of BAD responses (never do this):
- "So just to confirm, it's a large thin crust pizza with pepperoni, green pepper, and onion, correct?"
- "I'd also like to order some buffalo wings, a 12-count, can I get the price for that?"
- "So the garlic bread is $6.99, and the pizza is $18.50, what's the total come out to?"
- "No, that's all. Can you confirm the total price and give me an estimated delivery time?"
- "How about sausage instead?" (don't suggest subs yourself)
"""

_PHASE_BUILDERS: dict[ConversationPhase, object] = {}


def _greeting_prompt(order: OrderPayload, _ctx: OrderContext) -> str:
    return (
        "The employee just picked up. Say hi and that you wanna place a delivery order. "
        "That's it. Don't give your name or anything else yet.\n"
        f"Your info (DON'T volunteer this yet, only when asked — and use EXACTLY these values, never make up your own):\n"
        f"Name: {order.customer_name}\n"
        f"Phone: {order.phone_number}\n"
        f"Address: {order.delivery_address}"
    )


def _delivery_info_prompt(order: OrderPayload, _ctx: OrderContext) -> str:
    # Format phone for natural speech: "512-555-0147"
    p = order.phone_number
    phone_fmt = f"{p[:3]}-{p[3:6]}-{p[6:]}" if len(p) == 10 else p
    return (
        f"Answer ONLY what they ask. Nothing more.\n"
        f"USE THESE EXACT DETAILS — do NOT invent or change any of them:\n"
        f"Your name: {order.customer_name}\n"
        f"Your phone: {phone_fmt}\n"
        f"Your address: {order.delivery_address}\n"
        f"If they ask for name, just say the name. If they ask for number, just say the number. "
        f"Don't give address if they only asked for name. One thing at a time.\n"
        f"REMINDER: Your address is {order.delivery_address} — say it exactly."
    )


def _pizza_order_prompt(order: OrderPayload, ctx: OrderContext) -> str:
    toppings_natural = ', '.join(order.pizza.toppings)
    # Make the side description TTS-friendly for when we transition
    side_desc = order.side.first_choice.replace("12 count", "twelve count").replace("6 count", "six count")
    price_note = ""
    if ctx.pizza_price is not None:
        price_note = (
            f"\nYou already got the pizza price (${ctx.pizza_price:.2f}). "
            f"Now order the side: {side_desc}. Just casually say 'can I also get {side_desc}?'"
        )
    return (
        f"Order a {order.pizza.size} {order.pizza.crust} crust pizza with "
        f"{toppings_natural}.\n"
        f"If a topping is unavailable, DO NOT suggest a replacement yourself. "
        f"Wait for the employee to offer options. "
        f"ONLY accept these subs if THEY offer them: "
        f"{', '.join(order.pizza.acceptable_topping_subs)}.\n"
        f"NEVER accept: {', '.join(order.pizza.no_go_toppings)}. "
        f"If they offer one of those, say nah and ask what else they got.\n"
        f"When they give you the price, just say 'cool' or 'got it' and then order the side."
        f"{price_note}"
    )


def _side_order_prompt(order: OrderPayload, _ctx: OrderContext) -> str:
    # Make the side description TTS-friendly
    side_desc = order.side.first_choice.replace("12 count", "twelve count").replace("6 count", "six count")
    return (
        f"Now order a side: {side_desc}.\n"
        f"If unavailable, try these in order: {', '.join(order.side.backup_options)}.\n"
        f"If none available: {order.side.if_all_unavailable}.\n"
        f"When they give the price, just acknowledge it."
    )


def _drink_order_prompt(order: OrderPayload, ctx: OrderContext) -> str:
    # Make drink description TTS-friendly
    drink_desc = order.drink.first_choice.replace("2L", "two liter").replace("1L", "one liter")
    alts = [a.replace("2L", "two liter").replace("1L", "one liter") for a in order.drink.alternatives]
    skip_note = ""
    if order.drink.skip_if_over_budget:
        remaining = order.budget_max - ctx.running_total
        skip_note = f" Remaining budget: ${remaining:.2f}. If the drink pushes you over, skip it."
    return (
        f"Now order a drink: {drink_desc}.\n"
        f"If unavailable, try these in order: {', '.join(alts)}.\n"
        f"Budget: ${order.budget_max:.2f}, spent so far: ${ctx.running_total:.2f}.{skip_note}\n"
        f"When they give the price, just acknowledge it."
    )


def _price_collection_prompt(_order: OrderPayload, ctx: OrderContext) -> str:
    parts = []
    if ctx.running_total > 0:
        parts.append(f"Running total so far: ${ctx.running_total:.2f}.")
    if ctx.delivery_time is None:
        parts.append("You still need the delivery time estimate.")
    if ctx.order_number is None:
        parts.append("You still need the order/confirmation number.")
    if not parts:
        parts.append("You have everything. Move on to special instructions.")
    return (
        "Ask for the total, delivery time, and order number. "
        "Keep it casual — like 'alright what's the damage?' or 'how long's that gonna be?'\n"
        + " ".join(parts)
    )


def _special_instructions_prompt(order: OrderPayload, _ctx: OrderContext) -> str:
    return (
        f"Before you hang up, casually mention the delivery instructions: \"{order.special_instructions}\". "
        f"Say it naturally like 'oh one more thing — {order.special_instructions}'. "
        f"Then wait for them to acknowledge before saying thanks and bye."
    )


def _closing_prompt(_order: OrderPayload, _ctx: OrderContext) -> str:
    return "Say thanks and bye. Keep it short."


_PHASE_BUILDERS = {
    ConversationPhase.GREETING: _greeting_prompt,
    ConversationPhase.DELIVERY_INFO: _delivery_info_prompt,
    ConversationPhase.PIZZA_ORDER: _pizza_order_prompt,
    ConversationPhase.SIDE_ORDER: _side_order_prompt,
    ConversationPhase.DRINK_ORDER: _drink_order_prompt,
    ConversationPhase.PRICE_COLLECTION: _price_collection_prompt,
    ConversationPhase.SPECIAL_INSTRUCTIONS: _special_instructions_prompt,
    ConversationPhase.CLOSING: _closing_prompt,
}


class ConversationEngine:
    """Generates contextual LLM responses for the human conversation phase."""

    def __init__(self, order: OrderPayload) -> None:
        self.order = order
        self._client = AsyncGroq(api_key=GROQ_API_KEY)

    def build_system_prompt(
        self,
        phase: ConversationPhase,
        order: OrderPayload,
        context: OrderContext,
    ) -> str:
        builder = _PHASE_BUILDERS.get(phase)
        phase_text = builder(order, context) if builder else ""
        return _BASE_PROMPT + "\n\n" + phase_text

    async def generate_response(
        self,
        employee_text: str,
        phase: ConversationPhase,
        context: OrderContext,
        transcript_history: list[dict[str, str]],
        extra_instruction: str | None = None,
    ) -> str:
        system_prompt = self.build_system_prompt(phase, self.order, context)
        if extra_instruction:
            system_prompt += f"\n\nIMPORTANT RIGHT NOW: {extra_instruction}"

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        # Replay transcript history for context
        for entry in transcript_history:
            role = "assistant" if entry["role"] == "agent" else "user"
            messages.append({"role": role, "content": entry["text"]})

        messages.append({"role": "user", "content": employee_text})

        chat_completion = await self._client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=50,  # force short responses
        )

        return chat_completion.choices[0].message.content or ""

    async def generate_response_streamed(
        self,
        employee_text: str,
        phase: ConversationPhase,
        context: OrderContext,
        transcript_history: list[dict[str, str]],
        extra_instruction: str | None = None,
    ):
        """Yield text chunks as they arrive from Groq streaming API.

        Buffers tokens and yields on sentence boundaries (. , ! ? — or end)
        so TTS gets speakable fragments ASAP without cutting mid-word.
        """
        system_prompt = self.build_system_prompt(phase, self.order, context)
        if extra_instruction:
            system_prompt += f"\n\nIMPORTANT RIGHT NOW: {extra_instruction}"

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for entry in transcript_history:
            role = "assistant" if entry["role"] == "agent" else "user"
            messages.append({"role": role, "content": entry["text"]})
        messages.append({"role": "user", "content": employee_text})

        stream = await self._client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=50,
            stream=True,
        )

        buffer = ""
        # Yield on natural sentence/clause boundaries
        _BREAK_CHARS = frozenset(".!?,;—")

        async for chunk in stream:
            delta = chunk.choices[0].delta
            token = delta.content if delta and delta.content else ""
            if not token:
                continue
            buffer += token
            # Check if we hit a sentence boundary and have enough text to speak
            if len(buffer) >= 8 and any(c in _BREAK_CHARS for c in buffer[-3:]):
                yield buffer
                buffer = ""

        # Flush remaining
        if buffer.strip():
            yield buffer
