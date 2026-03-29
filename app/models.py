# Order payload and result models (Pydantic)

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class PizzaOrder(BaseModel):
    size: str
    crust: str
    toppings: list[str]
    acceptable_topping_subs: list[str]
    no_go_toppings: list[str]


class SideOrder(BaseModel):
    first_choice: str
    backup_options: list[str]
    if_all_unavailable: str  # "skip" or other instruction


class DrinkOrder(BaseModel):
    first_choice: str
    alternatives: list[str]
    skip_if_over_budget: bool


class OrderPayload(BaseModel):
    customer_name: str
    phone_number: str = Field(pattern=r"^\d{10}$")
    delivery_address: str
    pizza: PizzaOrder
    side: SideOrder
    drink: DrinkOrder
    budget_max: float = Field(gt=0)
    special_instructions: str

    @model_validator(mode="after")
    def validate_topping_lists(self) -> "OrderPayload":
        toppings_set = {t.lower() for t in self.pizza.toppings}
        no_go_set = {t.lower() for t in self.pizza.no_go_toppings}
        subs_set = {t.lower() for t in self.pizza.acceptable_topping_subs}

        overlap_toppings = toppings_set & no_go_set
        if overlap_toppings:
            raise ValueError(
                f"pizza.toppings and pizza.no_go_toppings overlap: {overlap_toppings}"
            )

        overlap_subs = subs_set & no_go_set
        if overlap_subs:
            raise ValueError(
                f"pizza.acceptable_topping_subs and pizza.no_go_toppings overlap: {overlap_subs}"
            )

        return self


class CallResult(BaseModel):
    outcome: str  # "completed" | "nothing_available" | "over_budget" | "detected_as_bot"
    pizza: Optional[dict] = None
    side: Optional[dict] = None
    drink: Optional[dict] = None
    total: Optional[float] = None
    delivery_time: Optional[str] = None
    order_number: Optional[str] = None
    special_instructions_delivered: bool = False


@dataclass
class OrderContext:
    # Extracted from payload
    customer_name: str
    phone_number: str
    delivery_address: str
    zip_code: str  # Extracted from delivery_address

    # Pizza tracking
    pizza_ordered: bool = False
    pizza_description: str = ""
    pizza_substitutions: dict[str, str] = field(default_factory=dict)
    pizza_price: Optional[float] = None

    # Side tracking
    side_ordered: bool = False
    side_description: str = ""
    side_original: str = ""
    side_price: Optional[float] = None
    side_skipped: bool = False

    # Drink tracking
    drink_ordered: bool = False
    drink_description: str = ""
    drink_price: Optional[float] = None
    drink_skipped: bool = False
    drink_skip_reason: Optional[str] = None

    # Order summary
    running_total: float = 0.0
    delivery_time: Optional[str] = None
    order_number: Optional[str] = None
    special_instructions_delivered: bool = False
