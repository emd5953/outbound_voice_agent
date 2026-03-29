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
- Keep responses to ONE short sentence. Two max if absolutely needed.
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
"""

_PHASE_BUILDERS: dict[ConversationPhase, object] = {}


def _greeting_prompt(order: OrderPayload, _ctx: OrderContext) -> str:
    return (
        "The employee just picked up. Say hi and that you wanna place a delivery order. "
        "That's it. Don't give your name or anything else yet."
    )


def _delivery_info_prompt(order: OrderPayload, _ctx: OrderContext) -> str:
    return (
        f"Answer ONLY what they ask. Nothing more.\n"
        f"Your name: {order.customer_name}\n"
        f"Your phone: {order.phone_number}\n"
        f"Your address: {order.delivery_address}\n"
        f"If they ask for name, just say the name. If they ask for number, just say the number. "
        f"Don't give address if they only asked for name. One thing at a time."
    )


def _pizza_order_prompt(order: OrderPayload, _ctx: OrderContext) -> str:
    return (
        f"Order a {order.pizza.size} {order.pizza.crust} with "
        f"{', '.join(order.pizza.toppings)}.\n"
        f"If a topping is unavailable, ONLY accept these subs: "
        f"{', '.join(order.pizza.acceptable_topping_subs)}.\n"
        f"NEVER accept: {', '.join(order.pizza.no_go_toppings)}. "
        f"If they offer one of those, say nah and ask what else they got.\n"
        f"When they give you the price, just say 'cool' or 'got it' and move on."
    )


def _side_order_prompt(order: OrderPayload, _ctx: OrderContext) -> str:
    return (
        f"Now order a side: {order.side.first_choice}.\n"
        f"If unavailable, try these in order: {', '.join(order.side.backup_options)}.\n"
        f"If none available: {order.side.if_all_unavailable}.\n"
        f"When they give the price, just acknowledge it."
    )


def _drink_order_prompt(order: OrderPayload, ctx: OrderContext) -> str:
    skip_note = (
        " If the drink pushes you over budget, skip it."
        if order.drink.skip_if_over_budget
        else ""
    )
    return (
        f"Now order a drink: {order.drink.first_choice}.\n"
        f"If unavailable, try: {', '.join(order.drink.alternatives)}.\n"
        f"Budget: ${order.budget_max:.2f}, spent so far: ${ctx.running_total:.2f}.{skip_note}\n"
        f"When they give the price, just acknowledge it."
    )


def _price_collection_prompt(_order: OrderPayload, ctx: OrderContext) -> str:
    return (
        f"Ask what the total comes to. Also ask how long for delivery and the order number. "
        f"Keep it casual — like 'alright what's the total?' or 'how long's that gonna be?'"
    )


def _special_instructions_prompt(order: OrderPayload, _ctx: OrderContext) -> str:
    return (
        f"Before you hang up, casually mention: \"{order.special_instructions}\". "
        f"Then say thanks and bye. Keep it natural."
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
    ) -> str:
        system_prompt = self.build_system_prompt(phase, self.order, context)

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        # Replay transcript history for context
        for entry in transcript_history:
            role = "assistant" if entry["role"] == "agent" else "user"
            messages.append({"role": role, "content": entry["text"]})

        messages.append({"role": "user", "content": employee_text})

        chat_completion = await self._client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=80,  # force short responses
        )

        return chat_completion.choices[0].message.content or ""
