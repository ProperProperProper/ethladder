#!/usr/bin/env python3
"""Claude CLI bridge — direct access to ETH Ladder bot learning data.

Usage:
    python ethladder_cli_bridge.py [command] [options]

Commands:
    state              - Current bot state (positions, balance, mode)
    omlx               - OMLX 10-D learning and pattern calibration
    ml                 - ML training status and trade memory
    backtest           - Walk-forward validation results
    performance        - Complete performance snapshot
    watch              - Watch live updates (--interval 5 for 5s refresh)
    export             - Export to JSON for fine-tuning
"""

import argparse
import json
import sys
import time
from pathlib import Path

ETHLADDER_DIR = Path.home() / "Documents" / "ethladder"
SYMBOL = "ETHUSDT"


def fetch_json(path: str) -> dict:
    """Load JSON file from ETH Ladder directory."""
    try:
        file = ETHLADDER_DIR / path
        if file.exists():
            return json.loads(file.read_text())
    except Exception as e:
        print(f"Error loading {path}: {e}", file=sys.stderr)
    return {}


def get_state() -> dict:
    """Current trading state."""
    return {
        "positions": fetch_json("positions.json").get("positions", []),
        "system_metrics": fetch_json("system_metrics.json"),
        "symbol": SYMBOL,
    }


def get_omlx() -> dict:
    """OMLX learning system."""
    metrics = fetch_json("omlx_metrics_live.json")
    calibration = fetch_json("dip_calibration_state.json")
    return {
        "bounce_probability": metrics.get("bounce_probability"),
        "dimensions": metrics.get("dimension_scores", {}),
        "patterns": calibration.get("patterns", {}),
        "training_events_count": len(calibration.get("training_events", [])),
        "total_trades_calibrated": calibration.get("trades_count", 0),
    }


def get_ml() -> dict:
    """ML training status."""
    trade_memory = fetch_json("trade_memory.json")
    trainer_status = fetch_json("ml_trainer_status.json")
    return {
        "total_trades": trade_memory.get("total_trades", 0),
        "recent_trades": trade_memory.get("trades", [])[:5],
        "trainer": trainer_status,
        "xgboost_accuracy": trainer_status.get("xgboost_accuracy"),
        "rl_q_table_states": trainer_status.get("rl_q_table_states"),
    }


def get_backtest() -> dict:
    """Walk-forward backtest results."""
    summary = fetch_json("walk_forward_training_summary.json")
    optimizer = fetch_json("data/optimizer_status.json")
    return {
        "best_params": summary.get("best_parameters", {}),
        "profit_factor": summary.get("profit_factor"),
        "win_rate": summary.get("win_rate"),
        "max_drawdown": summary.get("max_drawdown"),
        "optimizer_progress": optimizer.get("search_progress"),
    }


def get_performance() -> dict:
    """Complete performance snapshot."""
    return {
        "state": get_state(),
        "omlx": get_omlx(),
        "ml": get_ml(),
        "backtest": get_backtest(),
        "symbol": SYMBOL,
    }


def watch(interval: int = 5):
    """Watch live updates."""
    print(f"Watching ETH Ladder bot (refresh every {interval}s)...")
    print("Press Ctrl+C to stop\n")
    try:
        while True:
            perf = get_performance()
            positions = perf["state"]["positions"]
            omlx = perf["omlx"]
            ml = perf["ml"]

            print(f"[{time.strftime('%H:%M:%S')}]")
            print(f"  Positions: {len(positions)} open")
            print(f"  OMLX Bounce Prob: {omlx.get('bounce_probability', 'N/A')}%")
            print(f"  Trades Analyzed: {ml.get('total_trades', 0)}/20000")
            print(f"  Win Rate: {ml.get('trainer', {}).get('win_rate', 'N/A')}%")
            print()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


def export_json(output: str = None):
    """Export complete data for fine-tuning."""
    data = get_performance()
    output_file = output or "ethladder_export.json"
    Path(output_file).write_text(json.dumps(data, indent=2))
    print(f"✓ Exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="ETH Ladder CLI bridge — access bot learning data"
    )
    parser.add_argument(
        "command",
        choices=["state", "omlx", "ml", "backtest", "performance", "watch", "export"],
        help="Command to run"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Watch refresh interval (seconds)"
    )
    parser.add_argument(
        "--output",
        help="Output file for export"
    )

    args = parser.parse_args()

    result = None
    if args.command == "state":
        result = get_state()
    elif args.command == "omlx":
        result = get_omlx()
    elif args.command == "ml":
        result = get_ml()
    elif args.command == "backtest":
        result = get_backtest()
    elif args.command == "performance":
        result = get_performance()
    elif args.command == "watch":
        watch(args.interval)
        return
    elif args.command == "export":
        export_json(args.output)
        return

    if result:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
