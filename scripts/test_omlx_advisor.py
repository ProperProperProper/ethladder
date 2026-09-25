#!/usr/bin/env python3
"""Test OMLX advisor integration with sample queries."""

import asyncio
from symbot_python.exchange.omlx_advisor import OMLXAdvisor


async def main():
    advisor = OMLXAdvisor(enabled=True)

    print("=" * 70)
    print("OMLX ADVISOR DEMO - Bybit Trading Best Practices")
    print("=" * 70)

    # Test 1: Fee optimization
    print("\n[1] Fee Optimization Question")
    print("-" * 70)
    advice = await advisor.ask_fee_optimization("ETHUSDT", 0.1, "Buy")
    if advice:
        print(advice)
    else:
        print("(OMLX unavailable)")

    # Test 2: Order type recommendation
    print("\n[2] Order Type Selection (High Volatility)")
    print("-" * 70)
    advice = await advisor.ask_order_type("ETHUSDT", 0.1, volatility="high")
    if advice:
        print(advice)
    else:
        print("(OMLX unavailable)")

    # Test 3: Position sizing
    print("\n[3] Safe Position Sizing")
    print("-" * 70)
    advice = await advisor.ask_position_sizing(leverage=11, equity=1000, market_condition="trending")
    if advice:
        print(advice)
    else:
        print("(OMLX unavailable)")

    # Test 4: Stop loss placement
    print("\n[4] Stop Loss & Liquidation Price")
    print("-" * 70)
    advice = await advisor.ask_stop_loss(entry_price=2750, leverage=11, mmr=0.05)
    if advice:
        print(advice)
    else:
        print("(OMLX unavailable)")

    # Test 5: Funding rates
    print("\n[5] Funding Rate Information")
    print("-" * 70)
    advice = await advisor.ask_funding_info("ETHUSDT")
    if advice:
        print(advice)
    else:
        print("(OMLX unavailable)")

    # Test 6: Risk validation
    print("\n[6] Trade Risk Validation")
    print("-" * 70)
    advice = await advisor.validate_risk(
        symbol="ETHUSDT",
        leverage=11,
        entry_price=2750,
        position_qty=0.1,
        mmr=0.05,
    )
    if advice:
        print(advice)
    else:
        print("(OMLX unavailable)")

    # Test 7: API gotchas
    print("\n[7] Bybit API Edge Cases")
    print("-" * 70)
    advice = await advisor.ask_api_gotcha("Placing market orders during high volatility")
    if advice:
        print(advice)
    else:
        print("(OMLX unavailable)")

    await advisor.close()
    print("\n" + "=" * 70)
    print("Demo complete")


if __name__ == "__main__":
    asyncio.run(main())
