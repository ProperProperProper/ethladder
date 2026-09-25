from symbot_python.strategy.price_guard import evaluate_price_sanity


def test_no_reference_is_plausible():
    result = evaluate_price_sanity(price=100.0, reference=None)
    assert result.plausible is True
    assert result.reason == "no_reference"


def test_invalid_price_is_plausible_defers_to_other_guard():
    result = evaluate_price_sanity(price=0.0, reference=100.0)
    assert result.plausible is True
    assert result.reason == "invalid_price"


def test_within_band_is_plausible():
    result = evaluate_price_sanity(price=105.0, reference=100.0)
    assert result.plausible is True
    assert result.reason == "ok"


def test_above_band_rejected():
    # default high_ratio=2 -> anything > 2x reference is rejected
    result = evaluate_price_sanity(price=201.0, reference=100.0)
    assert result.plausible is False
    assert result.reason == "above_band"


def test_at_high_band_boundary_is_plausible():
    result = evaluate_price_sanity(price=200.0, reference=100.0)
    assert result.plausible is True


def test_below_band_rejected():
    # default low_ratio=10 -> anything < reference/10 is rejected
    result = evaluate_price_sanity(price=9.0, reference=100.0)
    assert result.plausible is False
    assert result.reason == "below_band"


def test_at_low_band_boundary_is_plausible():
    result = evaluate_price_sanity(price=10.0, reference=100.0)
    assert result.plausible is True


def test_custom_ratios_respected():
    result = evaluate_price_sanity(price=130.0, reference=100.0, max_high_ratio=1.2)
    assert result.plausible is False
    assert result.reason == "above_band"
