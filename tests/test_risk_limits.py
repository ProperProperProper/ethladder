import pytest

from symbot_python.exchange import risk_limits


class StubRiskSession:
    def __init__(self, tiers, raise_error=False):
        self.tiers = tiers
        self.raise_error = raise_error
        self.calls = 0

    def get_risk_limit(self, **kwargs):
        self.calls += 1
        if self.raise_error:
            raise RuntimeError("network error")
        return {"result": {"list": self.tiers}}


@pytest.fixture(autouse=True)
def clear_cache():
    risk_limits._cache.clear()
    yield
    risk_limits._cache.clear()


def test_picks_lowest_risk_tier():
    session = StubRiskSession([
        {"isLowestRisk": 0, "maintenanceMargin": "0.01"},
        {"isLowestRisk": 1, "maintenanceMargin": "0.0033"},
    ])
    rate = risk_limits.get_maintenance_margin_rate(session, "ETHUSDT")
    assert rate == pytest.approx(0.0033)


def test_falls_back_to_first_tier_if_no_lowest_risk_flag():
    session = StubRiskSession([{"isLowestRisk": 0, "maintenanceMargin": "0.005"}])
    rate = risk_limits.get_maintenance_margin_rate(session, "SOLUSDT")
    assert rate == pytest.approx(0.005)


def test_caches_per_symbol():
    session = StubRiskSession([{"isLowestRisk": 1, "maintenanceMargin": "0.0033"}])
    risk_limits.get_maintenance_margin_rate(session, "ETHUSDT")
    risk_limits.get_maintenance_margin_rate(session, "ETHUSDT")
    assert session.calls == 1


def test_falls_back_on_api_error():
    session = StubRiskSession([], raise_error=True)
    rate = risk_limits.get_maintenance_margin_rate(session, "XRPUSDT")
    assert rate == pytest.approx(risk_limits.FALLBACK_MAINTENANCE_MARGIN_RATE)


def test_different_symbols_cached_independently():
    session = StubRiskSession([{"isLowestRisk": 1, "maintenanceMargin": "0.0033"}])
    eth_rate = risk_limits.get_maintenance_margin_rate(session, "ETHUSDT")

    session2 = StubRiskSession([{"isLowestRisk": 1, "maintenanceMargin": "0.005"}])
    sol_rate = risk_limits.get_maintenance_margin_rate(session2, "SOLUSDT")

    assert eth_rate == pytest.approx(0.0033)
    assert sol_rate == pytest.approx(0.005)
