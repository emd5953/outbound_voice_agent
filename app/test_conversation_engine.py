# Tests for ConversationEngine: system prompt building and generate_response

from __future__ import annotations

import pytest

from app.conversation_engine import ConversationEngine, _BASE_PROMPT
from app.models import (
    DrinkOrder,
    OrderContext,
    OrderPayload,
    PizzaOrder,
    SideOrder,
)
from app.state_machine import ConversationPhase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_order() -> OrderPayload:
    return OrderPayload(
        customer_name="Jordan Mitchell",
        phone_number="5125550147",
        delivery_address="4821 Elm Street, Apt 3B, Austin, TX 78745",
        pizza=PizzaOrder(
            size="large",
            crust="thin",
            toppings=["pepperoni", "mushroom", "green pepper"],
            acceptable_topping_subs=["sausage", "bacon", "onion"],
            no_go_toppings=["olives", "anchovies"],
        ),
        side=SideOrder(
            first_choice="buffalo wings",
            backup_options=["garlic bread", "breadsticks"],
            if_all_unavailable="skip",
        ),
        drink=DrinkOrder(
            first_choice="2L Coke",
            alternatives=["2L Pepsi", "2L Sprite"],
            skip_if_over_budget=True,
        ),
        budget_max=45.00,
        special_instructions="Ring doorbell, don't knock",
    )


@pytest.fixture
def sample_context(sample_order: OrderPayload) -> OrderContext:
    return OrderContext(
        customer_name=sample_order.customer_name,
        phone_number=sample_order.phone_number,
        delivery_address=sample_order.delivery_address,
        zip_code="78745",
    )


@pytest.fixture
def engine(sample_order: OrderPayload) -> ConversationEngine:
    return ConversationEngine(order=sample_order)


# ---------------------------------------------------------------------------
# build_system_prompt tests
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    """Verify phase-specific system prompts contain required data."""

    def test_base_prompt_always_present(
        self, engine: ConversationEngine, sample_order: OrderPayload, sample_context: OrderContext
    ):
        for phase in ConversationPhase:
            prompt = engine.build_system_prompt(phase, sample_order, sample_context)
            assert _BASE_PROMPT in prompt

    def test_never_reveal_ai_instruction(
        self, engine: ConversationEngine, sample_order: OrderPayload, sample_context: OrderContext
    ):
        prompt = engine.build_system_prompt(
            ConversationPhase.GREETING, sample_order, sample_context
        )
        assert "never reveal you are an ai" in prompt.lower()

    def test_short_response_instruction(
        self, engine: ConversationEngine, sample_order: OrderPayload, sample_context: OrderContext
    ):
        prompt = engine.build_system_prompt(
            ConversationPhase.GREETING, sample_order, sample_context
        )
        assert "1-2 sentences" in prompt.lower()

    def test_greeting_contains_customer_info(
        self, engine: ConversationEngine, sample_order: OrderPayload, sample_context: OrderContext
    ):
        prompt = engine.build_system_prompt(
            ConversationPhase.GREETING, sample_order, sample_context
        )
        assert sample_order.customer_name in prompt
        assert sample_order.phone_number in prompt
        assert sample_order.delivery_address in prompt

    def test_pizza_order_contains_toppings_and_rules(
        self, engine: ConversationEngine, sample_order: OrderPayload, sample_context: OrderContext
    ):
        prompt = engine.build_system_prompt(
            ConversationPhase.PIZZA_ORDER, sample_order, sample_context
        )
        for t in sample_order.pizza.toppings:
            assert t in prompt
        for s in sample_order.pizza.acceptable_topping_subs:
            assert s in prompt
        for ng in sample_order.pizza.no_go_toppings:
            assert ng in prompt

    def test_side_order_contains_choices(
        self, engine: ConversationEngine, sample_order: OrderPayload, sample_context: OrderContext
    ):
        prompt = engine.build_system_prompt(
            ConversationPhase.SIDE_ORDER, sample_order, sample_context
        )
        assert sample_order.side.first_choice in prompt
        for b in sample_order.side.backup_options:
            assert b in prompt

    def test_drink_order_contains_budget_context(
        self, engine: ConversationEngine, sample_order: OrderPayload, sample_context: OrderContext
    ):
        sample_context.running_total = 25.49
        prompt = engine.build_system_prompt(
            ConversationPhase.DRINK_ORDER, sample_order, sample_context
        )
        assert "$25.49" in prompt
        assert f"${sample_order.budget_max:.2f}" in prompt
        # Drink name is TTS-friendly ("two liter" instead of "2L")
        assert "coke" in prompt.lower()

    def test_special_instructions_in_prompt(
        self, engine: ConversationEngine, sample_order: OrderPayload, sample_context: OrderContext
    ):
        prompt = engine.build_system_prompt(
            ConversationPhase.SPECIAL_INSTRUCTIONS, sample_order, sample_context
        )
        assert sample_order.special_instructions in prompt


# ---------------------------------------------------------------------------
# generate_response tests (mocked Groq API)
# ---------------------------------------------------------------------------

class TestGenerateResponse:
    """Verify generate_response builds correct messages and calls Groq."""

    @pytest.mark.asyncio
    async def test_generate_response_returns_string(
        self, engine: ConversationEngine, sample_order: OrderPayload, sample_context: OrderContext, monkeypatch
    ):
        """Mock the Groq client to verify the method wires correctly."""

        class FakeChoice:
            class message:
                content = "Sure, I'd like a large thin crust pizza."

        class FakeCompletion:
            choices = [FakeChoice()]

        async def fake_create(**kwargs):
            # Verify model is correct
            assert kwargs["model"] == "llama-3.3-70b-versatile"
            # Verify system message is first
            assert kwargs["messages"][0]["role"] == "system"
            # Verify employee text is last user message
            assert kwargs["messages"][-1]["role"] == "user"
            assert kwargs["messages"][-1]["content"] == "What can I get for you?"
            return FakeCompletion()

        monkeypatch.setattr(
            engine._client.chat.completions, "create", fake_create
        )

        result = await engine.generate_response(
            employee_text="What can I get for you?",
            phase=ConversationPhase.GREETING,
            context=sample_context,
            transcript_history=[],
        )
        assert result == "Sure, I'd like a large thin crust pizza."

    @pytest.mark.asyncio
    async def test_transcript_history_included(
        self, engine: ConversationEngine, sample_order: OrderPayload, sample_context: OrderContext, monkeypatch
    ):
        """Verify transcript history is passed as conversation messages."""
        captured_messages = []

        class FakeChoice:
            class message:
                content = "Yes, pepperoni please."

        class FakeCompletion:
            choices = [FakeChoice()]

        async def fake_create(**kwargs):
            captured_messages.extend(kwargs["messages"])
            return FakeCompletion()

        monkeypatch.setattr(
            engine._client.chat.completions, "create", fake_create
        )

        history = [
            {"role": "employee", "text": "What can I get for you?"},
            {"role": "agent", "text": "I'd like a large thin crust pizza."},
        ]

        await engine.generate_response(
            employee_text="What toppings?",
            phase=ConversationPhase.PIZZA_ORDER,
            context=sample_context,
            transcript_history=history,
        )

        # system + 2 history entries + 1 new user message = 4
        assert len(captured_messages) == 4
        assert captured_messages[0]["role"] == "system"
        assert captured_messages[1]["role"] == "user"  # employee -> user
        assert captured_messages[2]["role"] == "assistant"  # agent -> assistant
        assert captured_messages[3]["role"] == "user"
        assert captured_messages[3]["content"] == "What toppings?"
