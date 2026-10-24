import logging

import pytest

from distillrl import data
from distillrl.pii import SafeError, get_logger, has_pii, luhn_ok, scrub


@pytest.mark.parametrize(
    "card",
    ["4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111", "5555 5555 5555 4444", "3782 822463 10005"],
)
def test_luhn_valid_cards_masked(card):
    assert scrub(f"pay with {card} now") == "pay with [CARD] now"


@pytest.mark.parametrize("num", ["4111 1111 1111 1112", "1234567890123", "BW-12345", "2026102321085"])
def test_non_cards_left_alone(num):
    assert "[CARD]" not in scrub(num)


def test_luhn_function():
    assert luhn_ok("79927398713")
    assert not luhn_ok("79927398710")


@pytest.mark.parametrize("ssn", ["123-45-6789", "078-05-1121"])
def test_valid_ssn_masked(ssn):
    assert scrub(f"ssn {ssn}") == "ssn [SSN]"


@pytest.mark.parametrize("ssn", ["000-12-3456", "666-12-3456", "912-12-3456", "123-00-4567", "123-45-0000"])
def test_invalid_ssn_patterns_not_masked(ssn):
    assert "[SSN]" not in scrub(ssn)


@pytest.mark.parametrize("email", ["a.b@example.com", "cust42+x@mail.example.org"])
def test_email_masked(email):
    assert scrub(f"mail {email}.") == "mail [EMAIL]."


@pytest.mark.parametrize("phone", ["555-201-4477", "(555) 201-4477", "+1 555 201 4477", "555.201.4477"])
def test_phone_masked(phone):
    assert "[PHONE]" in scrub(f"call {phone}")


def test_order_ids_and_dates_not_phone():
    assert scrub("order BW-55512 on 2026-10-23") == "order BW-55512 on 2026-10-23"


def test_has_pii():
    assert has_pii("x@example.com") and not has_pii("hello")


def test_safe_error_scrubs_message():
    with pytest.raises(SafeError) as e:
        raise SafeError("bad row: card 4111 1111 1111 1111 email z@example.com")
    assert "4111" not in str(e.value) and "z@example.com" not in str(e.value)


def test_logger_scrubs_message_and_args(caplog):
    log = get_logger("distillrl.test")
    with caplog.at_level(logging.INFO, logger="distillrl.test"):
        log.info("refund for %s card %s", "q@example.com", "5555 5555 5555 4444")
    text = caplog.text
    assert "5555" not in text and "q@example.com" not in text and "[CARD]" in text


def test_logger_filter_added_once():
    a, b = get_logger("distillrl.once"), get_logger("distillrl.once")
    assert a is b and len(a.filters) == 1


def test_generated_raw_data_contains_pii_but_splits_do_not():
    raw = data.make_dataset(2000, 7)
    assert any(has_pii(e.text) for e in raw)
    for part in data.splits():
        assert not any(has_pii(e.text) for e in part)
