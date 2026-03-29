# Tests for CallOrchestrator (task 9.1)

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.call_orchestrator import CallOrchestrator
from app.models import (
    CallResult,
    DrinkOrder,
    OrderContext,
    OrderPayload,
    PizzaOrder,
    SideOrder,
)
from app.state_machine import CallState, ConversationPhase


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _make_order() -> OrderPayload:
    return OrderPayload(
        customer_name="John Doe",
        phone_number="5551234567",
        delivery_address="123 Main St, Springfield 62704",
        pizza=PizzaOrder(
            size="large",
            crust="thin",
            toppings=["pepperoni", "mushrooms"],
            acceptable_topping_subs=["olives", "onions"],
            no_go_toppings=["anchovies"],
        ),
        side=SideOrder(
            first_choice="garlic bread",
            backup_options=["breadsticks"],
            if_all_unavailable="skip",
        ),
        drink=DrinkOrder(
            first_choice="Coke",
            alternatives=["Sprite"],
            skip_if_over_budget=True,
        ),
        budget_max=40.00,
        special_instructions="Ring the doorbell twice",
    )


def _make_context(order: OrderPayload) -> OrderContext:
    return OrderContext(
        customer_name=order.customer_name,
        phone_number=order.phone_number,
        delivery_address=order.delivery_address,
        zip_code="62704",
    )


def _make_orchestrator() -> CallOrchestrator:
    order = _make_order()
    ctx = _make_context(order)
    return CallOrchestrator(order=order, context=ctx)


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

class TestInit:
    def test_initial_state_is_initiating(self):
        orch = _make_orchestrator()
        assert orch.state == CallState.INITIATING

    def test_initial_ivr_retry_count_is_zero(self):
        orch = _make_orchestrator()
        assert orch.ivr_retry_count == 0

    def test_initial_conversation_phase_is_greeting(self):
        orch = _make_orchestrator()
        assert orch.conversation_phase == ConversationPhase.GREETING

    def test_transcript_history_starts_empty(self):
        orch = _make_orchestrator()
        assert orch.transcript_history == []

    def test_order_and_context_stored(self):
        order = _make_order()
        ctx = _make_context(order)
        orch = CallOrchestrator(order=order, context=ctx)
        assert orch.order is order
        assert orch.context is ctx


# ------------------------------------------------------------------
# Component wiring
# ------------------------------------------------------------------

class TestWiring:
    def test_set_media_handler(self):
        orch = _make_orchestrator()
        handler = MagicMock()
        orch.set_media_handler(handler)
        assert orch._media_handler is handler

    def test_set_conversation_engine(self):
        orch = _make_orchestrator()
        engine = MagicMock()
        orch.set_conversation_engine(engine)
        assert orch._conversation_engine is engine


# ------------------------------------------------------------------
# handle_transcript routing
# ------------------------------------------------------------------

class TestHandleTranscript:
    @pytest.mark.asyncio
    async def test_ivr_state_routes_to_ivr_handler(self):
        orch = _make_orchestrator()
        orch.state = CallState.IVR_WELCOME

        with patch(
            "app.call_orchestrator.handle_ivr_transcript", new_callable=AsyncMock
        ) as mock_ivr:
            await orch.handle_transcript("Press 1 for delivery")
            mock_ivr.assert_called_once_with(orch, "Press 1 for delivery")

    @pytest.mark.asyncio
    async def test_on_hold_transitions_to_conversation_on_human(self):
        orch = _make_orchestrator()
        orch.state = CallState.ON_HOLD

        with patch.object(orch, "handle_conversation", new_callable=AsyncMock) as mock_conv:
            await orch.handle_transcript("Hey there, what can I get for you?")
            assert orch.state == CallState.CONVERSATION
            mock_conv.assert_called_once_with("Hey there, what can I get for you?")

    @pytest.mark.asyncio
    async def test_on_hold_ignores_short_transcript(self):
        orch = _make_orchestrator()
        orch.state = CallState.ON_HOLD

        await orch.handle_transcript("hi")
        assert orch.state == CallState.ON_HOLD

    @pytest.mark.asyncio
    async def test_conversation_state_routes_to_handle_conversation(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION

        with patch.object(orch, "handle_conversation", new_callable=AsyncMock) as mock_conv:
            await orch.handle_transcript("What size pizza?")
            mock_conv.assert_called_once_with("What size pizza?")

    @pytest.mark.asyncio
    async def test_transcript_recorded_in_history(self):
        orch = _make_orchestrator()
        orch.state = CallState.ON_HOLD

        await orch.handle_transcript("hi")
        assert len(orch.transcript_history) == 1
        assert orch.transcript_history[0] == {"role": "employee", "text": "hi"}

    @pytest.mark.asyncio
    async def test_multiple_transcripts_in_chronological_order(self):
        orch = _make_orchestrator()
        orch.state = CallState.ON_HOLD

        await orch.handle_transcript("hi")
        await orch.handle_transcript("hello there how can I help you today")

        # First transcript is short so ignored by hold detection (employee only)
        # Second triggers conversation: employee entry + agent response from speak()
        employee_entries = [e for e in orch.transcript_history if e["role"] == "employee"]
        assert len(employee_entries) == 2
        assert employee_entries[0]["text"] == "hi"
        assert employee_entries[1]["text"] == "hello there how can I help you today"


# ------------------------------------------------------------------
# send_dtmf
# ------------------------------------------------------------------

class TestSendDtmf:
    @pytest.mark.asyncio
    async def test_send_dtmf_records_in_history(self):
        orch = _make_orchestrator()
        orch.call_sid = "CA123"
        mock_client = MagicMock()
        orch.set_twilio_client(mock_client)

        await orch.send_dtmf("1")
        assert any("[DTMF: 1]" in e["text"] for e in orch.transcript_history)

    @pytest.mark.asyncio
    async def test_send_dtmf_calls_twilio(self):
        orch = _make_orchestrator()
        orch.call_sid = "CA123"
        mock_client = MagicMock()
        mock_call = MagicMock()
        mock_client.calls.return_value = mock_call
        orch.set_twilio_client(mock_client)

        await orch.send_dtmf("5551234567")

        mock_client.calls.assert_called_once_with("CA123")
        mock_call.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_dtmf_no_op_without_call_sid(self):
        orch = _make_orchestrator()
        orch.call_sid = ""
        mock_client = MagicMock()
        orch.set_twilio_client(mock_client)

        await orch.send_dtmf("1")
        mock_client.calls.assert_not_called()


# ------------------------------------------------------------------
# speak
# ------------------------------------------------------------------

class TestSpeak:
    @pytest.mark.asyncio
    async def test_speak_records_in_history(self):
        orch = _make_orchestrator()
        handler = MagicMock()
        handler.stream_sid = "MZ123"
        handler.twilio_ws = MagicMock()
        orch.set_media_handler(handler)

        with patch("app.media_handler.synthesize_and_send", new_callable=AsyncMock):
            await orch.speak("John Doe")

        assert any(e["text"] == "John Doe" for e in orch.transcript_history)

    @pytest.mark.asyncio
    async def test_speak_calls_synthesize_and_send(self):
        orch = _make_orchestrator()
        handler = MagicMock()
        handler.stream_sid = "MZ123"
        handler.twilio_ws = MagicMock()
        orch.set_media_handler(handler)

        with patch(
            "app.media_handler.synthesize_and_send", new_callable=AsyncMock
        ) as mock_tts:
            await orch.speak("yes")
            mock_tts.assert_called_once_with(
                text="yes",
                stream_sid="MZ123",
                twilio_ws=handler.twilio_ws,
            )

    @pytest.mark.asyncio
    async def test_speak_no_op_without_media_handler(self):
        orch = _make_orchestrator()
        # No media handler set — should not raise
        await orch.speak("hello")
        assert any(e["text"] == "hello" for e in orch.transcript_history)


# ------------------------------------------------------------------
# end_call
# ------------------------------------------------------------------

class TestEndCall:
    @pytest.mark.asyncio
    async def test_end_call_sets_hangup_state(self):
        orch = _make_orchestrator()
        await orch.end_call("completed")
        assert orch.state == CallState.HANGUP

    @pytest.mark.asyncio
    async def test_end_call_stores_outcome(self):
        orch = _make_orchestrator()
        await orch.end_call("over_budget")
        assert orch._call_result is not None
        assert orch._call_result.outcome == "over_budget"

    @pytest.mark.asyncio
    async def test_end_call_signals_completion(self):
        orch = _make_orchestrator()
        await orch.end_call("completed")
        assert orch._completion_event.is_set()


# ------------------------------------------------------------------
# wait_for_completion
# ------------------------------------------------------------------

class TestWaitForCompletion:
    @pytest.mark.asyncio
    async def test_returns_call_result_after_end(self):
        orch = _make_orchestrator()

        async def end_after_delay():
            await asyncio.sleep(0.05)
            await orch.end_call("completed")

        asyncio.create_task(end_after_delay())
        result = await orch.wait_for_completion()

        assert isinstance(result, CallResult)
        assert result.outcome == "completed"


# ------------------------------------------------------------------
# _build_result
# ------------------------------------------------------------------

class TestBuildResult:
    def test_empty_result_for_early_hangup(self):
        orch = _make_orchestrator()
        result = orch._build_result("ivr_failure")
        assert result.outcome == "ivr_failure"
        assert result.pizza is None
        assert result.side is None
        assert result.drink is None

    def test_partial_result_with_pizza_data(self):
        orch = _make_orchestrator()
        orch.context.pizza_ordered = True
        orch.context.pizza_description = "large thin pepperoni"
        orch.context.pizza_price = 18.99
        orch.context.running_total = 18.99

        result = orch._build_result("over_budget")
        assert result.pizza is not None
        assert result.pizza["price"] == 18.99
        assert result.total == 18.99

    def test_skipped_side_in_result(self):
        orch = _make_orchestrator()
        orch.context.side_skipped = True

        result = orch._build_result("completed")
        assert result.side is not None
        assert result.side["skipped"] is True


# ------------------------------------------------------------------
# handle_conversation — bot detection
# ------------------------------------------------------------------

class TestBotDetection:
    @pytest.mark.asyncio
    async def test_bot_detection_are_you_a_robot(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        await orch.handle_conversation("Hey, are you a robot?")
        assert orch.state == CallState.HANGUP
        assert orch._call_result is not None
        assert orch._call_result.outcome == "detected_as_bot"

    @pytest.mark.asyncio
    async def test_bot_detection_is_this_a_bot(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        await orch.handle_conversation("Wait, is this a bot?")
        assert orch._call_result.outcome == "detected_as_bot"

    @pytest.mark.asyncio
    async def test_bot_detection_is_this_automated(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        await orch.handle_conversation("is this automated")
        assert orch._call_result.outcome == "detected_as_bot"

    @pytest.mark.asyncio
    async def test_bot_detection_am_i_talking_to_a_computer(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        await orch.handle_conversation("am i talking to a computer right now")
        assert orch._call_result.outcome == "detected_as_bot"

    @pytest.mark.asyncio
    async def test_bot_detection_is_this_a_real_person(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        await orch.handle_conversation("is this a real person")
        assert orch._call_result.outcome == "detected_as_bot"

    @pytest.mark.asyncio
    async def test_bot_detection_speaks_goodbye(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        await orch.handle_conversation("are you a robot")
        agent_entries = [e for e in orch.transcript_history if e["role"] == "agent"]
        assert any("good" in e["text"].lower() for e in agent_entries)

    @pytest.mark.asyncio
    async def test_normal_text_no_bot_detection(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        await orch.handle_conversation("What size pizza do you want?")
        assert orch.state != CallState.HANGUP


# ------------------------------------------------------------------
# handle_conversation — price extraction
# ------------------------------------------------------------------

class TestPriceExtraction:
    @pytest.mark.asyncio
    async def test_pizza_price_recorded(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.PIZZA_ORDER
        await orch.handle_conversation("That'll be $18.99")
        assert orch.context.pizza_price == 18.99
        assert orch.context.running_total == 18.99

    @pytest.mark.asyncio
    async def test_side_price_recorded(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.SIDE_ORDER
        await orch.handle_conversation("The garlic bread is $6.50")
        assert orch.context.side_price == 6.50
        assert orch.context.running_total == 6.50

    @pytest.mark.asyncio
    async def test_no_price_no_change(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.PIZZA_ORDER
        await orch.handle_conversation("What toppings do you want?")
        assert orch.context.pizza_price is None
        assert orch.context.running_total == 0.0

    @pytest.mark.asyncio
    async def test_price_not_recorded_twice(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.PIZZA_ORDER
        orch.context.pizza_price = 15.00
        orch.context.running_total = 15.00
        await orch.handle_conversation("Actually that's $18.99")
        # Should not overwrite
        assert orch.context.pizza_price == 15.00
        assert orch.context.running_total == 15.00


# ------------------------------------------------------------------
# record_price
# ------------------------------------------------------------------

class TestRecordPrice:
    def test_record_pizza_price(self):
        orch = _make_orchestrator()
        orch.record_price(ConversationPhase.PIZZA_ORDER, 18.99)
        assert orch.context.pizza_price == 18.99
        assert orch.context.running_total == 18.99

    def test_record_side_price(self):
        orch = _make_orchestrator()
        orch.record_price(ConversationPhase.SIDE_ORDER, 6.50)
        assert orch.context.side_price == 6.50
        assert orch.context.running_total == 6.50

    def test_record_drink_price(self):
        orch = _make_orchestrator()
        orch.record_price(ConversationPhase.DRINK_ORDER, 3.49)
        assert orch.context.drink_price == 3.49
        assert orch.context.running_total == 3.49

    def test_accumulates_running_total(self):
        orch = _make_orchestrator()
        orch.record_price(ConversationPhase.PIZZA_ORDER, 18.99)
        orch.record_price(ConversationPhase.SIDE_ORDER, 6.50)
        orch.record_price(ConversationPhase.DRINK_ORDER, 3.49)
        assert abs(orch.context.running_total - 28.98) < 0.01

    def test_does_not_overwrite_existing_price(self):
        orch = _make_orchestrator()
        orch.record_price(ConversationPhase.PIZZA_ORDER, 18.99)
        orch.record_price(ConversationPhase.PIZZA_ORDER, 22.00)
        assert orch.context.pizza_price == 18.99
        assert orch.context.running_total == 18.99

    def test_ignores_unrelated_phase(self):
        orch = _make_orchestrator()
        orch.record_price(ConversationPhase.GREETING, 10.00)
        assert orch.context.running_total == 0.0


# ------------------------------------------------------------------
# maybe_advance_phase
# ------------------------------------------------------------------

class TestMaybeAdvancePhase:
    @pytest.mark.asyncio
    async def test_greeting_to_delivery_info(self):
        orch = _make_orchestrator()
        orch.conversation_phase = ConversationPhase.GREETING
        await orch.maybe_advance_phase()
        assert orch.conversation_phase == ConversationPhase.DELIVERY_INFO

    @pytest.mark.asyncio
    async def test_delivery_info_to_pizza_order(self):
        orch = _make_orchestrator()
        orch.conversation_phase = ConversationPhase.DELIVERY_INFO
        await orch.maybe_advance_phase()
        assert orch.conversation_phase == ConversationPhase.PIZZA_ORDER

    @pytest.mark.asyncio
    async def test_pizza_order_stays_without_price(self):
        orch = _make_orchestrator()
        orch.conversation_phase = ConversationPhase.PIZZA_ORDER
        await orch.maybe_advance_phase()
        assert orch.conversation_phase == ConversationPhase.PIZZA_ORDER

    @pytest.mark.asyncio
    async def test_pizza_order_to_side_order_with_price(self):
        orch = _make_orchestrator()
        orch.conversation_phase = ConversationPhase.PIZZA_ORDER
        orch.context.pizza_price = 18.99
        await orch.maybe_advance_phase()
        assert orch.conversation_phase == ConversationPhase.SIDE_ORDER

    @pytest.mark.asyncio
    async def test_side_order_stays_without_price_or_skip(self):
        orch = _make_orchestrator()
        orch.conversation_phase = ConversationPhase.SIDE_ORDER
        await orch.maybe_advance_phase()
        assert orch.conversation_phase == ConversationPhase.SIDE_ORDER

    @pytest.mark.asyncio
    async def test_side_order_to_drink_with_price(self):
        orch = _make_orchestrator()
        orch.conversation_phase = ConversationPhase.SIDE_ORDER
        orch.context.side_price = 6.50
        await orch.maybe_advance_phase()
        assert orch.conversation_phase == ConversationPhase.DRINK_ORDER

    @pytest.mark.asyncio
    async def test_side_order_to_drink_when_skipped(self):
        orch = _make_orchestrator()
        orch.conversation_phase = ConversationPhase.SIDE_ORDER
        orch.context.side_skipped = True
        await orch.maybe_advance_phase()
        assert orch.conversation_phase == ConversationPhase.DRINK_ORDER

    @pytest.mark.asyncio
    async def test_drink_order_to_price_collection_with_price(self):
        orch = _make_orchestrator()
        orch.conversation_phase = ConversationPhase.DRINK_ORDER
        orch.context.drink_price = 3.49
        await orch.maybe_advance_phase()
        assert orch.conversation_phase == ConversationPhase.PRICE_COLLECTION

    @pytest.mark.asyncio
    async def test_drink_order_to_price_collection_when_skipped(self):
        orch = _make_orchestrator()
        orch.conversation_phase = ConversationPhase.DRINK_ORDER
        orch.context.drink_skipped = True
        await orch.maybe_advance_phase()
        assert orch.conversation_phase == ConversationPhase.PRICE_COLLECTION

    @pytest.mark.asyncio
    async def test_price_collection_stays_without_both_fields(self):
        orch = _make_orchestrator()
        orch.conversation_phase = ConversationPhase.PRICE_COLLECTION
        orch.context.delivery_time = "30 minutes"
        await orch.maybe_advance_phase()
        assert orch.conversation_phase == ConversationPhase.PRICE_COLLECTION

    @pytest.mark.asyncio
    async def test_price_collection_to_special_instructions(self):
        orch = _make_orchestrator()
        orch.conversation_phase = ConversationPhase.PRICE_COLLECTION
        orch.context.delivery_time = "30 minutes"
        orch.context.order_number = "1234"
        await orch.maybe_advance_phase()
        assert orch.conversation_phase == ConversationPhase.SPECIAL_INSTRUCTIONS

    @pytest.mark.asyncio
    async def test_special_instructions_to_closing(self):
        orch = _make_orchestrator()
        orch.conversation_phase = ConversationPhase.SPECIAL_INSTRUCTIONS
        orch.context.special_instructions_delivered = True
        await orch.maybe_advance_phase()
        assert orch.conversation_phase == ConversationPhase.CLOSING
        # Transitioning to CLOSING triggers end_call("completed")
        assert orch.state == CallState.HANGUP
        assert orch._call_result is not None
        assert orch._call_result.outcome == "completed"

    @pytest.mark.asyncio
    async def test_special_instructions_stays_without_delivery(self):
        orch = _make_orchestrator()
        orch.conversation_phase = ConversationPhase.SPECIAL_INSTRUCTIONS
        await orch.maybe_advance_phase()
        assert orch.conversation_phase == ConversationPhase.SPECIAL_INSTRUCTIONS


# ------------------------------------------------------------------
# Drink budget logic
# ------------------------------------------------------------------

class TestDrinkBudgetLogic:
    @pytest.mark.asyncio
    async def test_drink_skipped_when_over_budget(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.DRINK_ORDER
        orch.context.running_total = 38.00  # budget_max is 40
        await orch.handle_conversation("The Coke is $5.00")
        assert orch.context.drink_skipped is True
        assert orch.context.drink_skip_reason == "over_budget"
        assert orch.context.drink_price is None
        # running_total should not include the drink
        assert orch.context.running_total == 38.00

    @pytest.mark.asyncio
    async def test_drink_proceeds_within_budget(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.DRINK_ORDER
        orch.context.running_total = 30.00
        await orch.handle_conversation("The Coke is $3.49")
        assert orch.context.drink_price == 3.49
        assert abs(orch.context.running_total - 33.49) < 0.01


# ------------------------------------------------------------------
# Hangup conditions (task 9.3)
# ------------------------------------------------------------------

class TestNothingAvailable:
    @pytest.mark.asyncio
    async def test_pizza_cant_be_made(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.PIZZA_ORDER
        await orch.handle_conversation("Sorry, we can't make that")
        assert orch.state == CallState.HANGUP
        assert orch._call_result.outcome == "nothing_available"

    @pytest.mark.asyncio
    async def test_no_pizza_available(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.PIZZA_ORDER
        await orch.handle_conversation("We don't have pizza right now")
        assert orch._call_result.outcome == "nothing_available"

    @pytest.mark.asyncio
    async def test_were_closed(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.PIZZA_ORDER
        await orch.handle_conversation("Sorry, we're closed")
        assert orch._call_result.outcome == "nothing_available"

    @pytest.mark.asyncio
    async def test_normal_pizza_conversation_no_hangup(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.PIZZA_ORDER
        await orch.handle_conversation("What size pizza do you want?")
        assert orch.state != CallState.HANGUP


class TestOverBudgetPizzaSide:
    @pytest.mark.asyncio
    async def test_pizza_price_exceeds_budget(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.PIZZA_ORDER
        # budget_max is 40.00
        await orch.handle_conversation("That pizza is $45.00")
        assert orch.state == CallState.HANGUP
        assert orch._call_result.outcome == "over_budget"

    @pytest.mark.asyncio
    async def test_side_price_pushes_over_budget(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.SIDE_ORDER
        orch.context.pizza_price = 35.00
        orch.context.running_total = 35.00
        await orch.handle_conversation("The garlic bread is $8.00")
        assert orch.state == CallState.HANGUP
        assert orch._call_result.outcome == "over_budget"

    @pytest.mark.asyncio
    async def test_pizza_price_within_budget_no_hangup(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.PIZZA_ORDER
        await orch.handle_conversation("That pizza is $18.99")
        assert orch.state != CallState.HANGUP


class TestCompletedHangup:
    @pytest.mark.asyncio
    async def test_closing_triggers_completed(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.SPECIAL_INSTRUCTIONS
        orch.context.special_instructions_delivered = True
        await orch.maybe_advance_phase()
        assert orch.state == CallState.HANGUP
        assert orch._call_result.outcome == "completed"


class TestSpecialInstructionsDelivery:
    @pytest.mark.asyncio
    async def test_special_instructions_marked_delivered(self):
        orch = _make_orchestrator()
        orch.state = CallState.CONVERSATION
        orch.conversation_phase = ConversationPhase.SPECIAL_INSTRUCTIONS
        await orch.handle_conversation("Anything else?")
        assert orch.context.special_instructions_delivered is True


class TestBuildResultDrinkSkipped:
    def test_drink_skipped_includes_reason(self):
        orch = _make_orchestrator()
        orch.context.drink_skipped = True
        orch.context.drink_skip_reason = "over_budget"
        result = orch._build_result("completed")
        assert result.drink is not None
        assert result.drink["skipped"] is True
        assert result.drink["reason"] == "over_budget"

    def test_drink_skipped_default_reason(self):
        orch = _make_orchestrator()
        orch.context.drink_skipped = True
        result = orch._build_result("completed")
        assert result.drink is not None
        assert result.drink["skipped"] is True
        assert result.drink["reason"] == "over_budget"
