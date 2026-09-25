import pytest

from symbot_python.strategy.optimization_store import (
    OptimizationRecord,
    connect,
    demote_winner,
    get_best_current_winner,
    get_current_winner,
    promote_to_winner,
    record,
)


def make_record(**overrides) -> OptimizationRecord:
    defaults = dict(
        symbol="ETHUSDT", interval="15", period_days=14, params={"dca_take_profit_percent": 0.5},
        in_sample_score=10.0, out_of_sample_return_quote=10.0, out_of_sample_return_percent=1.0,
        out_of_sample_trade_count=5, out_of_sample_win_rate=0.9, max_drawdown_percent=1.0,
        max_funds_required=0.0, run_kind="search", liquidation_count=0,
    )
    defaults.update(overrides)
    return OptimizationRecord(**defaults)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test_optimization_results.db")
    yield c
    c.close()


def test_promote_then_get_current_winner(conn):
    rec_id = record(conn, make_record())
    assert get_current_winner(conn, "ETHUSDT", "15") is None
    promote_to_winner(conn, "ETHUSDT", "15", rec_id)
    row = get_current_winner(conn, "ETHUSDT", "15")
    assert row is not None
    assert row["id"] == rec_id


def test_promoting_a_new_winner_demotes_the_old_one(conn):
    first_id = record(conn, make_record())
    promote_to_winner(conn, "ETHUSDT", "15", first_id)
    second_id = record(conn, make_record(in_sample_score=20.0))
    promote_to_winner(conn, "ETHUSDT", "15", second_id)

    row = get_current_winner(conn, "ETHUSDT", "15")
    assert row["id"] == second_id


def test_demote_winner_clears_it(conn):
    rec_id = record(conn, make_record())
    promote_to_winner(conn, "ETHUSDT", "15", rec_id)
    assert get_current_winner(conn, "ETHUSDT", "15") is not None

    demote_winner(conn, "ETHUSDT", "15")
    assert get_current_winner(conn, "ETHUSDT", "15") is None


def test_get_best_current_winner_picks_highest_score_across_intervals(conn):
    id_15 = record(conn, make_record(interval="15", in_sample_score=5.0))
    promote_to_winner(conn, "ETHUSDT", "15", id_15)
    id_60 = record(conn, make_record(interval="60", in_sample_score=50.0))
    promote_to_winner(conn, "ETHUSDT", "60", id_60)

    best = get_best_current_winner(conn, "ETHUSDT")
    assert best["id"] == id_60
    assert best["interval"] == "60"


def test_get_best_current_winner_never_returns_a_liquidated_config(conn):
    # The actual safety property this exists for: even if something
    # upstream ever promoted a liquidated candidate by mistake, reads
    # must never hand it back as "the" winner to run live/paper.
    bad_id = record(conn, make_record(interval="15", in_sample_score=1000.0, liquidation_count=1))
    promote_to_winner(conn, "ETHUSDT", "15", bad_id)
    good_id = record(conn, make_record(interval="60", in_sample_score=5.0, liquidation_count=0))
    promote_to_winner(conn, "ETHUSDT", "60", good_id)

    best = get_best_current_winner(conn, "ETHUSDT")
    assert best is not None
    assert best["id"] == good_id  # NOT the higher-scoring but liquidated one


def test_get_best_current_winner_none_when_only_liquidated_winners_exist(conn):
    bad_id = record(conn, make_record(interval="15", in_sample_score=1000.0, liquidation_count=2))
    promote_to_winner(conn, "ETHUSDT", "15", bad_id)

    assert get_best_current_winner(conn, "ETHUSDT") is None


def test_get_best_current_winner_treats_pre_migration_null_as_visible(conn):
    # A row from before liquidation_count existed has it as NULL, not 0 —
    # must not be treated as "confirmed liquidated" and hidden.
    rec_id = record(conn, make_record(interval="15", in_sample_score=5.0))
    promote_to_winner(conn, "ETHUSDT", "15", rec_id)
    conn.execute("UPDATE optimization_results SET liquidation_count=NULL WHERE id=?", (rec_id,))
    conn.commit()

    best = get_best_current_winner(conn, "ETHUSDT")
    assert best is not None
    assert best["id"] == rec_id
