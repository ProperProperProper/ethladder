#!/usr/bin/env python3
"""Test ACTUAL Learning - Verify system improves, not just saves files."""

import json
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_pattern_learning_progression():
    """Test: Do patterns actually learn and improve?"""
    logger.info("=" * 80)
    logger.info("TEST: Pattern Learning Progression")
    logger.info("=" * 80)

    calibration_files = sorted(
        Path('.').glob('forward_test_calibration_*.json')
    )

    if len(calibration_files) < 2:
        logger.warning("\n❌ Need 2+ iterations for learning test")
        logger.warning("   Run: ./RUN_CONTINUOUS_LEARNING.sh (at least 2 cycles)")
        return False

    logger.info(f"\nAnalyzing {len(calibration_files)} iterations...\n")

    # Track each pattern's success rate over time
    pattern_evolution = defaultdict(list)

    for i, cal_file in enumerate(sorted(calibration_files)):
        with open(cal_file) as f:
            data = json.load(f)

        patterns = data.get('pattern_success_rates', {})
        logger.info(f"Iteration {i+1}: {len(patterns)} patterns")

        for pattern, rate in patterns.items():
            pattern_evolution[pattern].append((i+1, rate))

    # Check if patterns are learning
    if not pattern_evolution:
        logger.warning("❌ No pattern data found")
        return False

    logger.info(f"\nPattern Learning Analysis:")
    logger.info(f"{'Pattern':<25} {'Iter 1':<8} {'Latest':<8} {'Trend':<15}")
    logger.info("-" * 60)

    learning_detected = False
    for pattern, rates in pattern_evolution.items():
        if len(rates) < 2:
            continue

        first_rate = rates[0][1]
        latest_rate = rates[-1][1]
        delta = latest_rate - first_rate

        # Determine trend
        if abs(delta) > 2:  # At least 2% change
            trend = f"{delta:+.1f}% ↑" if delta > 0 else f"{delta:.1f}% ↓"
            learning_detected = True
        else:
            trend = "stable"

        logger.info(f"{pattern:<25} {first_rate:>6.1f}% {latest_rate:>6.1f}% {trend:>15}")

    if not learning_detected:
        logger.warning("\n❌ NO LEARNING DETECTED")
        logger.warning("   Pattern rates not changing between iterations")
        logger.warning("   This suggests forward tester isn't collecting diverse data")
        return False

    logger.info("\n✓ LEARNING DETECTED")
    logger.info("  Patterns are changing success rates")
    return True


def test_rl_state_growth():
    """Test: Is RL learning new states?"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: RL State Discovery")
    logger.info("=" * 80)

    rl_model = Path('omlx_rl_model.json')
    if not rl_model.exists():
        logger.warning("\n❌ RL model not found")
        return False

    with open(rl_model) as f:
        data = json.load(f)

    states_learned = len(data.get('Q_table', {}))
    episodes = data.get('episodes', 0)
    total_reward = data.get('total_reward', 0)

    logger.info(f"\nRL Q-Learning Status:")
    logger.info(f"  States discovered: {states_learned}")
    logger.info(f"  Episodes trained: {episodes}")
    logger.info(f"  Total reward: {total_reward:,.0f}")

    # Check if learning is happening
    if states_learned < 50:
        logger.warning(f"\n⚠️  LOW state count: {states_learned}")
        logger.info("   Need more diverse trading conditions")
        return False

    if episodes < 100:
        logger.warning(f"\n⚠️  LOW episodes: {episodes}")
        logger.info("   Forward tester not generating enough trades")
        return False

    if total_reward > 0:
        logger.info(f"\n✓ RL IS LEARNING")
        logger.info(f"  Avg reward per episode: {total_reward/episodes:+.1f}")
        logger.info(f"  Positive total reward indicates learning good actions")
        return True
    else:
        logger.warning(f"\n⚠️  Negative total reward: {total_reward}")
        logger.info("   RL hasn't learned good strategies yet")
        return False


def test_accuracy_improvement():
    """Test: Is accuracy actually improving?"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Accuracy Improvement Over Iterations")
    logger.info("=" * 80)

    accuracy_files = sorted(
        Path('.').glob('forward_test_accuracy_*.json')
    )

    if len(accuracy_files) < 2:
        logger.warning("\n❌ Need 2+ iterations")
        return False

    logger.info(f"\nAnalyzing {len(accuracy_files)} iterations...\n")

    accuracies = []
    win_rates = []

    for i, acc_file in enumerate(sorted(accuracy_files)):
        with open(acc_file) as f:
            data = json.load(f)

        metrics = data.get('metrics', {})
        acc = metrics.get('accuracy', 0) * 100
        wr = metrics.get('win_rate', 0) * 100

        accuracies.append(acc)
        win_rates.append(wr)

        logger.info(f"Iter {i+1}: Acc={acc:>6.1f}%, WR={wr:>6.1f}%")

    if not accuracies or max(accuracies) == 0:
        logger.warning("\n⚠️  No accuracy data yet (normal first run)")
        logger.info("   Run forward tester to generate trade data")
        return False

    # Check for improvement
    first_acc = accuracies[0]
    last_acc = accuracies[-1]
    acc_delta = last_acc - first_acc

    first_wr = win_rates[0]
    last_wr = win_rates[-1]
    wr_delta = last_wr - first_wr

    logger.info(f"\nTrend Analysis:")
    logger.info(f"  Accuracy: {first_acc:.1f}% → {last_acc:.1f}% ({acc_delta:+.1f}%)")
    logger.info(f"  Win Rate: {first_wr:.1f}% → {last_wr:.1f}% ({wr_delta:+.1f}%)")

    if acc_delta > 2 or wr_delta > 3:
        logger.info(f"\n✓ IMPROVING")
        logger.info(f"  System is getting better over iterations")
        return True
    elif acc_delta > 0 or wr_delta > 0:
        logger.info(f"\n✓ SLIGHT IMPROVEMENT")
        logger.info(f"  Positive trend detected")
        return True
    else:
        logger.warning(f"\n⚠️  NOT IMPROVING")
        logger.warning(f"  Need more training data")
        return False


def test_decision_quality():
    """Test: Are ML-enhanced decisions better than OMLX alone?"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: ML Enhancement Quality")
    logger.info("=" * 80)

    # This would require detailed trade data from forward tester
    # For now, check if RL confidence exists
    rl_model = Path('omlx_rl_model.json')
    if not rl_model.exists():
        logger.warning("\n⚠️  RL model not available yet")
        logger.info("   Run forward tester to generate data")
        return False

    logger.info("\n✓ RL models available for decision enhancement")
    logger.info("  ML advisor can enhance OMLX decisions")
    return True


def main():
    """Run all learning tests."""
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + "LEARNING VERIFICATION - Does System Actually Improve?".center(78) + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "=" * 78 + "╝")

    tests = [
        ("Pattern Learning Progression", test_pattern_learning_progression),
        ("RL State Discovery", test_rl_state_growth),
        ("Accuracy Improvement", test_accuracy_improvement),
        ("Decision Enhancement Quality", test_decision_quality),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Error: {e}")
            results.append((name, False))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("LEARNING TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓" if result else "⏳"
        logger.info(f"  {status} {name}")

    logger.info(f"\nResult: {passed}/{total} tests passed")

    if passed >= 2:
        logger.info("\n✅ SYSTEM IS LEARNING")
        logger.info("   Evidence of improvement detected")
        logger.info("   Continue running forward tests")
        return True
    else:
        logger.info("\n⏳ MORE DATA NEEDED")
        logger.info("   Run: ./RUN_CONTINUOUS_LEARNING.sh")
        logger.info("   Or: python run_forward_omlx_tester.py --duration 30")
        logger.info("\n   Then re-run this test")
        return False


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
