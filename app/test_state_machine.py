import pytest
from dataclasses import dataclass, field
from typing import Optional

from app.models import OrderPayload
from app.state_machine import CallState, handle_ivr_transcript


# --- Lightweight fake orchestrator for testing (no circular import) ---

@dataclass
class FakeOrchestrator:
    """Minimal stand-in for CallOrchestrator used in unit tests."""

    state: CallState
    order: OrderPayload
    ivr_retry_count: int = 0

    # Recorded actions for assertions
    dtmf_sent: list[str] = field(default_factory=list)
    spoken: list[str] = field(default_factory=list)
    ended_with: Optional[str] = None

    async def send_dtmf(self, digits: str) -> None:
        self.dtmf_sent.append(digits)

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    async def end_call(self, outcome: str) -> None:
        self.ended_with = outcome


def _make_order(**overrides) -> OrderPayload:
    defaults = dict(
        customer_name="Jordan Mitchell",
        phone_number="5125550147",
        delivery_address="4821 Elm Street, Austin, TX 78745",
        pizza=dict(
            size="large",
            crust="thin",
            toppings=["pepperoni"],
            acceptable_topping_subs=["mushrooms"],
            no_go_toppings=["anchovies"],
        ),
        side=dict(
            first_choice="garlic bread",
            backup_options=["breadsticks"],
            if_all_unavailable="skip",
        ),
        drink=dict(
            first_choice="2L Coke",
            alternatives=["2L Pepsi"],
            skip_if_over_budget=True,
        ),
        budget_max=45.0,
        special_instructions="Ring doorbell",
    )
    defaults.update(overrides)
    return OrderPayload(**defaults)


class TestIVRWelcome:
    @pytest.mark.asyncio
    async def test_press_1_sends_dtmf_and_transitions(self):
        orch = FakeOrchestrator(state=CallState.IVR_WELCOME, order=_make_order())
        await handle_ivr_transcript(orch, "Press 1 for delivery, press 2 for pickup")
        assert orch.dtmf_sent == ["1"]
        assert orch.state == CallState.IVR_NAME

    @pytest.mark.asyncio
    async def test_unrecognised_prompt_no_action(self):
        orch = FakeOrchestrator(state=CallState.IVR_WELCOME, order=_make_order())
        await handle_ivr_transcript(orch, "Welcome to Joe's Pizza")
        assert orch.dtmf_sent == []
        assert orch.state == CallState.IVR_WELCOME


class TestIVRName:
    @pytest.mark.asyncio
    async def test_speaks_bare_customer_name(self):
        orch = FakeOrchestrator(state=CallState.IVR_NAME, order=_make_order())
        await handle_ivr_transcript(orch, "Please say the name for the order")
        assert orch.spoken == ["Jordan Mitchell"]
        assert orch.state == CallState.IVR_CALLBACK

    @pytest.mark.asyncio
    async def test_name_has_no_prefix(self):
        orch = FakeOrchestrator(
            state=CallState.IVR_NAME,
            order=_make_order(customer_name="Alice"),
        )
        await handle_ivr_transcript(orch, "Please say the name for the order")
        assert orch.spoken == ["Alice"]


class TestIVRCallback:
    @pytest.mark.asyncio
    async def test_sends_phone_as_dtmf(self):
        orch = FakeOrchestrator(state=CallState.IVR_CALLBACK, order=_make_order())
        await handle_ivr_transcript(orch, "Please enter your 10-digit callback number")
        assert orch.dtmf_sent == ["5125550147"]
        assert orch.state == CallState.IVR_ZIPCODE


class TestIVRZipcode:
    @pytest.mark.asyncio
    async def test_speaks_bare_zip_code(self):
        orch = FakeOrchestrator(state=CallState.IVR_ZIPCODE, order=_make_order())
        await handle_ivr_transcript(orch, "Please say your delivery zip code")
        assert orch.spoken == ["78745"]
        assert orch.state == CallState.IVR_CONFIRM


class TestIVRConfirm:
    @pytest.mark.asyncio
    async def test_speaks_yes_and_transitions_to_hold(self):
        orch = FakeOrchestrator(state=CallState.IVR_CONFIRM, order=_make_order())
        await handle_ivr_transcript(orch, "Is that correct?")
        assert orch.spoken == ["yes"]
        assert orch.state == CallState.ON_HOLD


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_didnt_understand_increments_retry(self):
        orch = FakeOrchestrator(state=CallState.IVR_NAME, order=_make_order())
        await handle_ivr_transcript(orch, "I didn't understand that")
        assert orch.ivr_retry_count == 1
        assert orch.state == CallState.IVR_NAME
        assert orch.ended_with is None

    @pytest.mark.asyncio
    async def test_try_again_increments_retry(self):
        orch = FakeOrchestrator(state=CallState.IVR_ZIPCODE, order=_make_order())
        await handle_ivr_transcript(orch, "Please try again")
        assert orch.ivr_retry_count == 1
        assert orch.state == CallState.IVR_ZIPCODE

    @pytest.mark.asyncio
    async def test_three_retries_ends_call(self):
        orch = FakeOrchestrator(
            state=CallState.IVR_CALLBACK, order=_make_order(), ivr_retry_count=2
        )
        await handle_ivr_transcript(orch, "I didn't understand that, try again")
        assert orch.ivr_retry_count == 3
        assert orch.ended_with == "ivr_failure"

    @pytest.mark.asyncio
    async def test_retry_preserves_state(self):
        for state in [
            CallState.IVR_WELCOME,
            CallState.IVR_NAME,
            CallState.IVR_CALLBACK,
            CallState.IVR_ZIPCODE,
            CallState.IVR_CONFIRM,
        ]:
            orch = FakeOrchestrator(state=state, order=_make_order())
            await handle_ivr_transcript(orch, "I didn't understand that")
            assert orch.state == state, f"State changed unexpectedly for {state}"


class TestCallBackLater:
    @pytest.mark.asyncio
    async def test_call_back_later_ends_call(self):
        orch = FakeOrchestrator(state=CallState.IVR_WELCOME, order=_make_order())
        await handle_ivr_transcript(orch, "Please call back later")
        assert orch.ended_with == "ivr_failure"
