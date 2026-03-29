# Integration tests with mocked external services (task 13.1, 13.2)

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.call_orchestrator import CallOrchestrator
from app.conversation_engine import ConversationEngine
from app.models import (
    CallResult,
    DrinkOrder,
    OrderContext,
    OrderPayload,
    PizzaOrder,
    SideOrder,
)
from app.server import app
from app.state_machine import CallState, ConversationPhase


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

def _make_payload_dict(**overrides) -> dict:
    defaults = dict(
        customer_name="Jordan Mitchell",
        phone_number="5125550147",
        delivery_address="4821 Elm Street, Austin, TX 78745",
        pizza=dict(
            size="large",
            crust="thin",
            toppings=["pepperoni", "mushroom", "green pepper"],
            acceptable_topping_subs=["sausage", "bacon", "onion"],
            no_go_toppings=["olives", "anchovies"],
        ),
        side=dict(
            first_choice="buffalo wings",
            backup_options=["garlic bread", "breadsticks"],
            if_all_unavailable="skip",
        ),
        drink=dict(
            first_choice="2L Coke",
            alternatives=["2L Pepsi", "2L Sprite"],
            skip_if_over_budget=True,
        ),
        budget_max=45.00,
        special_instructions="Ring doorbell, don't knock",
    )
    defaults.update(overrides)
    return defaults


def _make_order(**overrides) -> OrderPayload:
    return OrderPayload(**_make_payload_dict(**overrides))


def _make_context(order: OrderPayload) -> OrderContext:
    return OrderContext(
        customer_name=order.customer_name,
        phone_number=order.phone_number,
        delivery_address=order.delivery_address,
        zip_code="78745",
    )


def _make_orchestrator(**overrides) -> CallOrchestrator:
    order = _make_order(**overrides)
    ctx = _make_context(order)
    return CallOrchestrator(order=order, context=ctx)


# ------------------------------------------------------------------
# 13.1 — Integration: IVR navigation with mocked services
# ------------------------------------------------------------------

class TestIVRIntegration:
    """Simulate IVR prompts in sequence and verify exact DTMF/TTS responses."""

    @pytest.mark.asyncio
    async def test_full_ivr_happy_path(self):
        orch = _make_orchestrator()
        orch.state = CallState.IVR_WELCOME

        # Prompt 1: Welcome — press 1 for delivery
        await orch.handle_transcript(
            "Thank you for calling. Press 1 for delivery. Press 2 for carryout."
        )
        assert orch.state == CallState.IVR_NAME
        assert any("[DTMF: 1]" in e["text"] for e in orch.transcript_history)

        # Prompt 2: Name
        await orch.handle_transcript("Please say the name for the order.")
        assert orch.state == CallState.IVR_CALLBACK
        assert any(
            e["text"] == "Jordan Mitchell" and e["role"] == "agent"
            for e in orch.transcript_history
        )

        # Prompt 3: Callback number
        await orch.handle_transcript(
            "Please enter your 10-digit callback number."
        )
        assert orch.state == CallState.IVR_ZIPCODE
        assert any("[DTMF: 5125550147]" in e["text"] for e in orch.transcript_history)

        # Prompt 4: Zip code
        await orch.handle_transcript("Please say your delivery zip code.")
        assert orch.state == CallState.IVR_CONFIRM
        assert any(
            e["text"] == "78745" and e["role"] == "agent"
            for e in orch.transcript_history
        )

        # Prompt 5: Confirmation
        await orch.handle_transcript(
            "I heard Jordan Mitchell, 5125550147, zip code 78745. Is that correct?"
        )
        assert orch.state == CallState.ON_HOLD
        assert any(
            e["text"] == "yes" and e["role"] == "agent"
            for e in orch.transcript_history
        )

    @pytest.mark.asyncio
    async def test_ivr_retry_then_success(self):
        orch = _make_orchestrator()
        orch.state = CallState.IVR_NAME

        # First attempt fails
        await orch.handle_transcript("I didn't understand that, please try again.")
        assert orch.state == CallState.IVR_NAME
        assert orch.ivr_retry_count == 1

        # Second attempt succeeds
        await orch.handle_transcript("Please say the name for the order.")
        assert orch.state == CallState.IVR_CALLBACK

    @pytest.mark.asyncio
    async def test_ivr_max_retries_ends_call(self):
        orch = _make_orchestrator()
        orch.state = CallState.IVR_ZIPCODE

        for _ in range(3):
            await orch.handle_transcript("I didn't understand that, try again.")

        assert orch.state == CallState.HANGUP
        assert orch._call_result.outcome == "ivr_failure"


# ------------------------------------------------------------------
# 13.1 — Integration: Hold → Conversation transition
# ------------------------------------------------------------------

class TestHoldToConversationIntegration:
    """Verify state transitions from ON_HOLD through CONVERSATION."""

    @pytest.mark.asyncio
    async def test_hold_ignores_short_then_detects_human(self):
        orch = _make_orchestrator()
        orch.state = CallState.ON_HOLD

        # Short noise — should stay on hold
        await orch.handle_transcript("mm")
        assert orch.state == CallState.ON_HOLD

        # Human greeting — should transition
        await orch.handle_transcript("Hey thanks for calling, what can I get for you?")
        assert orch.state == CallState.CONVERSATION


# ------------------------------------------------------------------
# 13.1 — Integration: Conversation phase progression with mocked LLM
# ------------------------------------------------------------------

class TestConversationIntegration:
    """Full conversation flow with mocked Groq LLM responses."""

    @pytest.mark.asyncio
    async def test_happy_path_completed(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.GREETING

        # Mock the conversation engine
        engine = MagicMock(spec=ConversationEngine)
        engine.generate_response = AsyncMock(return_value="Sure thing.")
        orch.set_conversation_engine(engine)

        # Greeting → Delivery Info → Pizza Order
        await orch.handle_conversation("What can I get for you?")
        assert orch.conversation_phase == ConversationPhase.DELIVERY_INFO

        await orch.handle_conversation("Name for the order?")
        assert orch.conversation_phase == ConversationPhase.PIZZA_ORDER

        # Pizza price received → Side Order
        await orch.handle_conversation("That pizza is $18.50")
        assert orch.context.pizza_price == 18.50
        assert orch.conversation_phase == ConversationPhase.SIDE_ORDER

        # Side price received → Drink Order
        await orch.handle_conversation("Garlic bread is $6.99")
        assert orch.context.side_price == 6.99
        assert orch.conversation_phase == ConversationPhase.DRINK_ORDER

        # Drink price received → Price Collection
        await orch.handle_conversation("2L Coke is $3.49")
        assert orch.context.drink_price == 3.49
        assert orch.conversation_phase == ConversationPhase.PRICE_COLLECTION

        # Set delivery time and order number to advance
        orch.context.delivery_time = "35 minutes"
        orch.context.order_number = "4412"
        await orch.maybe_advance_phase()
        assert orch.conversation_phase == ConversationPhase.SPECIAL_INSTRUCTIONS

        # Special instructions delivered → Closing → completed
        await orch.handle_conversation("Anything else?")
        assert orch.context.special_instructions_delivered is True
        assert orch.state == CallState.HANGUP
        assert orch._call_result.outcome == "completed"

    @pytest.mark.asyncio
    async def test_nothing_available_outcome(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.PIZZA_ORDER

        engine = MagicMock(spec=ConversationEngine)
        engine.generate_response = AsyncMock(return_value="Oh no, sorry.")
        orch.set_conversation_engine(engine)

        await orch.handle_conversation("Sorry, we can't make that pizza right now")
        assert orch.state == CallState.HANGUP
        assert orch._call_result.outcome == "nothing_available"

    @pytest.mark.asyncio
    async def test_over_budget_outcome(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.PIZZA_ORDER

        engine = MagicMock(spec=ConversationEngine)
        engine.generate_response = AsyncMock(return_value="That's too much.")
        orch.set_conversation_engine(engine)

        # budget_max is 45.00
        await orch.handle_conversation("That pizza is $50.00")
        assert orch.state == CallState.HANGUP
        assert orch._call_result.outcome == "over_budget"

    @pytest.mark.asyncio
    async def test_detected_as_bot_outcome(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION

        await orch.handle_conversation("Hey, are you a robot?")
        assert orch.state == CallState.HANGUP
        assert orch._call_result.outcome == "detected_as_bot"

    @pytest.mark.asyncio
    async def test_drink_skipped_over_budget(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.DRINK_ORDER
        orch.context.running_total = 42.00

        engine = MagicMock(spec=ConversationEngine)
        engine.generate_response = AsyncMock(return_value="Let's skip the drink.")
        orch.set_conversation_engine(engine)

        await orch.handle_conversation("The Coke is $5.00")
        assert orch.context.drink_skipped is True
        assert orch.context.drink_skip_reason == "over_budget"
        assert orch.context.drink_price is None

    @pytest.mark.asyncio
    async def test_side_skipped_all_unavailable(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.SIDE_ORDER

        engine = MagicMock(spec=ConversationEngine)
        engine.generate_response = AsyncMock(return_value="No worries, skip the side.")
        orch.set_conversation_engine(engine)

        # Employee says everything is out — mention all backup options too
        await orch.handle_conversation(
            "We're out of buffalo wings, garlic bread, and breadsticks"
        )
        assert orch.context.side_skipped is True


# ------------------------------------------------------------------
# 13.1 — Integration: CallResult schema validation
# ------------------------------------------------------------------

class TestCallResultSchema:
    """Verify structured JSON output matches CallResult schema for each outcome."""

    def test_completed_result_schema(self):
        orch = _make_orchestrator()
        orch.context.pizza_ordered = True
        orch.context.pizza_description = "large thin pepperoni, onion, green pepper"
        orch.context.pizza_substitutions = {"mushroom": "onion"}
        orch.context.pizza_price = 18.50
        orch.context.side_ordered = True
        orch.context.side_description = "garlic bread"
        orch.context.side_original = "buffalo wings"
        orch.context.side_price = 6.99
        orch.context.drink_ordered = True
        orch.context.drink_description = "2L Coke"
        orch.context.drink_price = 3.49
        orch.context.running_total = 28.98
        orch.context.delivery_time = "35 minutes"
        orch.context.order_number = "4412"
        orch.context.special_instructions_delivered = True

        result = orch._build_result("completed")
        assert isinstance(result, CallResult)
        assert result.outcome == "completed"
        assert result.pizza is not None
        assert result.pizza["price"] == 18.50
        assert result.side is not None
        assert result.side["price"] == 6.99
        assert result.drink is not None
        assert result.drink["price"] == 3.49
        assert result.total == 28.98
        assert result.delivery_time == "35 minutes"
        assert result.order_number == "4412"
        assert result.special_instructions_delivered is True

        # Verify JSON serialization works
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["outcome"] == "completed"

    def test_nothing_available_result_schema(self):
        orch = _make_orchestrator()
        result = orch._build_result("nothing_available")
        assert result.outcome == "nothing_available"
        assert result.pizza is None
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["outcome"] == "nothing_available"

    def test_over_budget_result_schema(self):
        orch = _make_orchestrator()
        orch.context.pizza_price = 40.00
        orch.context.running_total = 40.00
        result = orch._build_result("over_budget")
        assert result.outcome == "over_budget"
        assert result.total == 40.00

    def test_detected_as_bot_result_schema(self):
        orch = _make_orchestrator()
        result = orch._build_result("detected_as_bot")
        assert result.outcome == "detected_as_bot"

    def test_drink_skipped_result_schema(self):
        orch = _make_orchestrator()
        orch.context.drink_skipped = True
        orch.context.drink_skip_reason = "over_budget"
        result = orch._build_result("completed")
        assert result.drink is not None
        assert result.drink["skipped"] is True
        assert result.drink["reason"] == "over_budget"


# ------------------------------------------------------------------
# 13.2 — Endpoint validation tests
# ------------------------------------------------------------------

class TestPlaceOrderEndpoint:
    """Test POST /place-order with valid and invalid payloads."""

    @pytest.mark.asyncio
    async def test_valid_payload_returns_200(self):
        """Valid payload should initiate a call and return 200."""
        payload = _make_payload_dict()

        # Mock Twilio client and call creation
        mock_call = MagicMock()
        mock_call.sid = "CA_TEST_123"

        mock_twilio = MagicMock()
        mock_twilio.calls.create.return_value = mock_call

        async def fake_wait():
            return CallResult(outcome="completed")

        with patch("app.server.TwilioClient", return_value=mock_twilio):
            with patch.object(
                CallOrchestrator,
                "wait_for_completion",
                side_effect=fake_wait,
            ):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post("/place-order", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["outcome"] == "completed"

    @pytest.mark.asyncio
    async def test_invalid_phone_number_returns_422(self):
        """Phone number not 10 digits should return validation error."""
        payload = _make_payload_dict(phone_number="123")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/place-order", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_zip_in_address_returns_422(self):
        """Address without a zip code should return 422."""
        payload = _make_payload_dict(delivery_address="123 Main St, No Zip Here")

        # Mock Twilio so it doesn't actually try to connect
        mock_twilio = MagicMock()
        with patch("app.server.TwilioClient", return_value=mock_twilio):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/place-order", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_overlapping_toppings_returns_422(self):
        """Toppings overlapping with no_go_toppings should return validation error."""
        payload = _make_payload_dict()
        payload["pizza"]["toppings"] = ["olives", "pepperoni"]
        payload["pizza"]["no_go_toppings"] = ["olives"]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/place-order", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_overlapping_subs_and_nogo_returns_422(self):
        """acceptable_topping_subs overlapping no_go_toppings should return 422."""
        payload = _make_payload_dict()
        payload["pizza"]["acceptable_topping_subs"] = ["olives", "bacon"]
        payload["pizza"]["no_go_toppings"] = ["olives"]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/place-order", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_negative_budget_returns_422(self):
        """Non-positive budget_max should return validation error."""
        payload = _make_payload_dict(budget_max=-10)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/place-order", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_zero_budget_returns_422(self):
        """Zero budget_max should return validation error (gt=0)."""
        payload = _make_payload_dict(budget_max=0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/place-order", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_no_call_initiated_on_invalid_payload(self):
        """Invalid payload should NOT initiate a Twilio call."""
        payload = _make_payload_dict(phone_number="bad")
        mock_twilio = MagicMock()

        with patch("app.server.TwilioClient", return_value=mock_twilio):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/place-order", json=payload)

        assert resp.status_code == 422
        mock_twilio.calls.create.assert_not_called()
