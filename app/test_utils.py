import pytest
from app.models import PizzaOrder
from app.utils import extract_zip_code, evaluate_topping_substitution, evaluate_budget, detect_hold_end, extract_price


class TestExtractZipCode:
    def test_extracts_zip_from_full_address(self):
        assert extract_zip_code("123 Main St, Springfield, IL 62704") == "62704"

    def test_extracts_zip_at_end_of_string(self):
        assert extract_zip_code("456 Oak Ave, Portland OR 97201") == "97201"

    def test_extracts_first_zip_when_multiple(self):
        assert extract_zip_code("Apt 12345, 99 Elm St, NY 10001") == "12345"

    def test_raises_on_no_zip(self):
        with pytest.raises(ValueError, match="No 5-digit zip code found"):
            extract_zip_code("123 Main Street, No Zip Here")

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError):
            extract_zip_code("")

    def test_ignores_longer_digit_sequences(self):
        # 6-digit number should not match as a zip
        assert extract_zip_code("ID 1234567, City 90210") == "90210"

    def test_zip_only(self):
        assert extract_zip_code("90210") == "90210"


def _make_pizza_order(
    acceptable_subs=None,
    no_go=None,
) -> PizzaOrder:
    return PizzaOrder(
        size="large",
        crust="thin",
        toppings=["pepperoni"],
        acceptable_topping_subs=acceptable_subs or [],
        no_go_toppings=no_go or [],
    )


class TestEvaluateToppingSubstitution:
    def test_reject_blacklisted_topping(self):
        order = _make_pizza_order(no_go=["anchovies"])
        assert evaluate_topping_substitution("anchovies", order) == "reject"

    def test_accept_whitelisted_topping(self):
        order = _make_pizza_order(acceptable_subs=["mushrooms"])
        assert evaluate_topping_substitution("mushrooms", order) == "accept"

    def test_reject_topping_not_on_either_list(self):
        order = _make_pizza_order(acceptable_subs=["mushrooms"], no_go=["anchovies"])
        assert evaluate_topping_substitution("pineapple", order) == "reject"

    def test_case_insensitive_blacklist(self):
        order = _make_pizza_order(no_go=["Anchovies"])
        assert evaluate_topping_substitution("ANCHOVIES", order) == "reject"

    def test_case_insensitive_whitelist(self):
        order = _make_pizza_order(acceptable_subs=["Mushrooms"])
        assert evaluate_topping_substitution("MUSHROOMS", order) == "accept"

    def test_substring_match_blacklist(self):
        order = _make_pizza_order(no_go=["anchovy"])
        assert evaluate_topping_substitution("anchovies", order) == "reject"

    def test_substring_match_whitelist(self):
        order = _make_pizza_order(acceptable_subs=["mushroom"])
        assert evaluate_topping_substitution("mushrooms", order) == "accept"

    def test_blacklist_checked_before_whitelist(self):
        # If somehow a topping substring-matches both lists, blacklist wins
        order = _make_pizza_order(acceptable_subs=["hot peppers"], no_go=["pepper"])
        assert evaluate_topping_substitution("pepper", order) == "reject"

    def test_strips_whitespace_from_offered(self):
        order = _make_pizza_order(acceptable_subs=["mushrooms"])
        assert evaluate_topping_substitution("  mushrooms  ", order) == "accept"

    def test_empty_lists_reject_everything(self):
        order = _make_pizza_order()
        assert evaluate_topping_substitution("anything", order) == "reject"


class TestEvaluateBudget:
    def test_proceed_when_under_budget(self):
        assert evaluate_budget(10.0, 5.0, 20.0, is_drink=False, skip_drink_if_over=False) == "proceed"

    def test_proceed_when_exactly_at_budget(self):
        assert evaluate_budget(10.0, 10.0, 20.0, is_drink=False, skip_drink_if_over=False) == "proceed"

    def test_over_budget_non_drink(self):
        assert evaluate_budget(18.0, 5.0, 20.0, is_drink=False, skip_drink_if_over=False) == "over_budget"

    def test_skip_drink_when_over_budget_with_flag(self):
        assert evaluate_budget(18.0, 5.0, 20.0, is_drink=True, skip_drink_if_over=True) == "skip_drink"

    def test_over_budget_drink_without_skip_flag(self):
        assert evaluate_budget(18.0, 5.0, 20.0, is_drink=True, skip_drink_if_over=False) == "over_budget"

    def test_proceed_drink_under_budget(self):
        assert evaluate_budget(10.0, 3.0, 20.0, is_drink=True, skip_drink_if_over=True) == "proceed"

    def test_zero_running_total(self):
        assert evaluate_budget(0.0, 15.0, 20.0, is_drink=False, skip_drink_if_over=False) == "proceed"

    def test_over_budget_from_zero(self):
        assert evaluate_budget(0.0, 25.0, 20.0, is_drink=False, skip_drink_if_over=False) == "over_budget"


class TestDetectHoldEnd:
    # Short transcripts (< 5 chars) should be ignored
    def test_empty_string_returns_false(self):
        assert detect_hold_end("") is False

    def test_short_noise_returns_false(self):
        assert detect_hold_end("hmm") is False

    def test_four_char_string_returns_false(self):
        assert detect_hold_end("yeah") is False

    # Known greeting patterns trigger detection
    def test_what_can_i_get(self):
        assert detect_hold_end("Hey, what can I get started for you?") is True

    def test_how_can_i_help(self):
        assert detect_hold_end("How can I help you today?") is True

    def test_thanks_for_calling(self):
        assert detect_hold_end("Thanks for calling Joe's Pizza") is True

    def test_thanks_for_holding(self):
        assert detect_hold_end("Thanks for holding, what can I get you?") is True

    def test_what_would_you_like(self):
        assert detect_hold_end("What would you like to order?") is True

    def test_hey_there(self):
        assert detect_hold_end("Hey there, ready to order?") is True

    def test_hi_there(self):
        assert detect_hold_end("Hi there!") is True

    def test_what_can_i_do_for_you(self):
        assert detect_hold_end("What can I do for you?") is True

    # Case insensitive matching
    def test_greeting_case_insensitive(self):
        assert detect_hold_end("HOW CAN I HELP YOU?") is True

    # 4+ word transcripts treated as human speech
    def test_four_words_returns_true(self):
        assert detect_hold_end("ready to take order") is True

    def test_long_sentence_returns_true(self):
        assert detect_hold_end("alright go ahead and tell me what you want") is True

    # 3 or fewer words (>= 5 chars) without greeting pattern
    def test_three_words_no_greeting_returns_false(self):
        assert detect_hold_end("yeah sure okay") is False

    def test_single_long_word_returns_false(self):
        assert detect_hold_end("mmhmm yeah") is False

    # Whitespace handling
    def test_strips_leading_trailing_whitespace(self):
        assert detect_hold_end("   hi   ") is False  # "hi" is < 5 chars after strip

    def test_padded_greeting_detected(self):
        assert detect_hold_end("  how can i help  ") is True


class TestExtractPrice:
    def test_dollar_sign_price(self):
        assert extract_price("That'll be $12.99") == 12.99

    def test_price_without_dollar_sign(self):
        assert extract_price("The total is 15.50 for that") == 15.50

    def test_no_price_returns_none(self):
        assert extract_price("We have pepperoni and mushrooms") is None

    def test_first_price_returned_when_multiple(self):
        assert extract_price("Pizza is $12.99 and drink is $2.50") == 12.99

    def test_price_at_end_of_string(self):
        assert extract_price("Your total comes to $25.00") == 25.00

    def test_price_with_single_digit_dollars(self):
        assert extract_price("The drink is $3.50") == 3.50

    def test_empty_string_returns_none(self):
        assert extract_price("") is None

    def test_integer_without_decimals_returns_none(self):
        assert extract_price("That's about 30 dollars") is None

    def test_price_embedded_in_sentence(self):
        assert extract_price("so the large pizza is $18.75 with tax") == 18.75
