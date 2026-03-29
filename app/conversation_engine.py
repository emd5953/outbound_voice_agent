# LLM conversation engine: system prompt building and response generation via Groq

from __future__ import annotations

from groq import AsyncGroq

from app.models import OrderContext, OrderPayload
from app.state_machine import ConversationPhase
from config import GROQ_API_KEY

MODEL = "llama-3.3-70b-versatile"

_BASE_PROMPT = (
    "You are a customer calling a pizza restaurant to place a delivery order. "
    "Speak naturally and concisely like a real person on the phone. "
    "Never reveal you are an AI. Keep responses short — 1-2 sentences max. "
    "Wait for the employee to finish speaking before responding. "
    "If the employee pauses, wait — they may be typing. "
    "If asked to repeat something, just repeat it without frustration."
)

_PHASE_BUILDERS: dict[ConversationPhase, object] = {}  # populated below


def _greeting_prompt(order: OrderPayload, _ctx: OrderContext) -> str:
    return (
        f"You're placing a delivery order. "
        f"Name: {order.customer_name}. "
        f"Phone: {order.phone_number}. "
        f"Address: {order.delivery_address}."
    )


def _delivery_info_prompt(order: OrderPayload, _ctx: OrderContext) -> str:
    return (
        f"Provide delivery details when asked. "
        f"Name: {order.customer_name}. "
        f"Phone: {order.phone_number}. "
        f"Address: {order.delivery_address}."
    )


def _pizza_order_prompt(order: OrderPayload, _ctx: OrderContext) -> str:
    return (
        f"Order a {order.pizza.size} {order.pizza.crust} pizza with "
        f"{', '.join(order.pizza.toppings)}. "
        f"If a topping is unavailable, accept ONLY these substitutes: "
        f"{', '.join(order.pizza.acceptable_topping_subs)}. "
        f"NEVER accept these toppings: {', '.join(order.pizza.no_go_toppings)}. "
        f"If offered a no-go topping, politely decline and ask what else they have. "
        f"Ask for the price of the pizza."
    )


def _side_order_prompt(order: OrderPayload, _ctx: OrderContext) -> str:
    return (
        f"Order: {order.side.first_choice}. "
        f"If unavailable, try in order: {', '.join(order.side.backup_options)}. "
        f"If all unavailable: {order.side.if_all_unavailable}. "
        f"Ask for the price."
    )


def _drink_order_prompt(order: OrderPayload, ctx: OrderContext) -> str:
    skip_note = (
        "Skip drink if it would exceed budget."
        if order.drink.skip_if_over_budget
        else ""
    )
    return (
        f"Order: {order.drink.first_choice}. "
        f"If unavailable, try: {', '.join(order.drink.alternatives)}. "
        f"Current running total: ${ctx.running_total:.2f}. "
        f"Budget max: ${order.budget_max:.2f}. "
        f"{skip_note} "
        f"Ask for the price."
    )


def _price_collection_prompt(_order: OrderPayload, ctx: OrderContext) -> str:
    return (
        f"Confirm the total price. Running total so far: ${ctx.running_total:.2f}. "
        f"Ask for the delivery time estimate and order number."
    )


def _special_instructions_prompt(order: OrderPayload, _ctx: OrderContext) -> str:
    return (
        f'Before finishing, tell the employee: "{order.special_instructions}". '
        f"Then say thanks and goodbye."
    )


def _closing_prompt(_order: OrderPayload, _ctx: OrderContext) -> str:
    return "Thank the employee and say goodbye politely."


# Register phase builders
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
        """Build a phase-specific system prompt with order data and rules."""
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
        """Call Groq API with Llama 3.3 70B and return the agent reply."""
        system_prompt = self.build_system_prompt(phase, self.order, context)

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        # Replay transcript history so the LLM has full conversation context
        for entry in transcript_history:
            role = "assistant" if entry["role"] == "agent" else "user"
            messages.append({"role": role, "content": entry["text"]})

        # Append the latest employee utterance
        messages.append({"role": "user", "content": employee_text})

        chat_completion = await self._client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )

        return chat_completion.choices[0].message.content or ""
