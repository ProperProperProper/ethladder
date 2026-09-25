#!/usr/bin/env python3
"""Analyze historical data from SQLite."""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List
import json


class DataAnalyzer:
    """Analyze historical metrics."""

    def __init__(self, db_path: str = 'ethladder_analytics.db'):
        """Initialize analyzer."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def accuracy_trend(self, hours: int = 24) -> List[Dict]:
        """Get accuracy trend."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT timestamp, accuracy, win_rate
            FROM omlx_metrics
            WHERE timestamp >= datetime('now', ? || ' hours')
            ORDER BY timestamp
        ''', (f'-{hours}',))
        return [dict(row) for row in cursor.fetchall()]

    def performance_summary(self, hours: int = 24) -> Dict:
        """Get performance summary."""
        cursor = self.conn.cursor()

        # OMLX metrics
        cursor.execute('''
            SELECT
                COUNT(*) as samples,
                AVG(accuracy) as avg_accuracy,
                MAX(accuracy) as peak_accuracy,
                AVG(win_rate) as avg_win_rate,
                MAX(win_rate) as peak_win_rate
            FROM omlx_metrics
            WHERE timestamp >= datetime('now', ? || ' hours')
        ''', (f'-{hours}',))
        omlx_stats = dict(cursor.fetchone())

        # System metrics
        cursor.execute('''
            SELECT
                AVG(cpu_percent) as avg_cpu,
                MAX(cpu_percent) as peak_cpu,
                AVG(memory_percent) as avg_memory,
                AVG(temperature_celsius) as avg_temp
            FROM system_metrics
            WHERE timestamp >= datetime('now', ? || ' hours')
        ''', (f'-{hours}',))
        system_stats = dict(cursor.fetchone())

        # Trading metrics
        cursor.execute('''
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN win=1 THEN 1 ELSE 0 END) as winning_trades,
                AVG(pnl) as avg_pnl,
                SUM(pnl) as total_pnl
            FROM paper_trades
            WHERE timestamp >= datetime('now', ? || ' hours')
        ''', (f'-{hours}',))
        trading_stats = dict(cursor.fetchone())

        return {
            'period_hours': hours,
            'timestamp': datetime.now().isoformat(),
            'omlx': omlx_stats,
            'system': system_stats,
            'trading': trading_stats
        }

    def pattern_analysis(self) -> List[Dict]:
        """Analyze patterns."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT
                pattern_name,
                AVG(success_rate) as avg_success,
                SUM(occurrences) as total_occurrences,
                MAX(timestamp) as last_seen
            FROM patterns
            GROUP BY pattern_name
            ORDER BY avg_success DESC
        ''')
        return [dict(row) for row in cursor.fetchall()]

    def comparison_report(self) -> Dict:
        """Compare backtest vs paper trading."""
        cursor = self.conn.cursor()

        cursor.execute('SELECT AVG(win_rate) as avg_win_rate FROM backtest_results LIMIT 1')
        row = cursor.fetchone()
        backtest_wr = dict(row).get('avg_win_rate', 0) if row else 0

        cursor.execute('''
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN win=1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as paper_win_rate
            FROM paper_trades
        ''')
        paper_row = cursor.fetchone()
        paper_stats = dict(paper_row) if paper_row else {}

        paper_wr = paper_stats.get('paper_win_rate', 0) or 0
        diff = abs((backtest_wr or 0) - (paper_wr or 0))

        return {
            'backtest_win_rate': backtest_wr or 0,
            'paper_win_rate': paper_wr or 0,
            'paper_total_trades': paper_stats.get('total_trades', 0) or 0,
            'comparison': {
                'status': 'matching' if diff < 5 else 'diverging',
                'difference': diff
            }
        }

    def hourly_stats(self) -> List[Dict]:
        """Get hourly statistics."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT
                strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                AVG(accuracy) as avg_accuracy,
                COUNT(*) as samples
            FROM omlx_metrics
            WHERE timestamp >= datetime('now', '-24 hours')
            GROUP BY hour
            ORDER BY hour DESC
        ''')
        return [dict(row) for row in cursor.fetchall()]

    def export_report(self, filename: str = 'analytics_report.json'):
        """Export comprehensive report."""
        report = {
            'generated': datetime.now().isoformat(),
            'last_24h': self.performance_summary(24),
            'last_7d': self.performance_summary(24*7),
            'patterns': self.pattern_analysis(),
            'comparison': self.comparison_report(),
            'hourly': self.hourly_stats()
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"✓ Report exported to {filename}")
        return report

    def close(self):
        """Close connection."""
        self.conn.close()


def main():
    """Run analysis."""
    analyzer = DataAnalyzer()

    print("\n" + "="*60)
    print("ETH LADDER ANALYTICS REPORT")
    print("="*60 + "\n")

    # Performance summary
    summary_24h = analyzer.performance_summary(24)
    print("📊 LAST 24 HOURS:")
    print(f"  OMLX Accuracy: {summary_24h['omlx'].get('avg_accuracy', 0):.1f}% (peak: {summary_24h['omlx'].get('peak_accuracy', 0):.1f}%)")
    print(f"  Win Rate: {summary_24h['omlx'].get('avg_win_rate', 0):.1f}%")
    print(f"  CPU Usage: {summary_24h['system'].get('avg_cpu', 0):.1f}%")
    print(f"  Memory: {summary_24h['system'].get('avg_memory', 0):.1f}%")
    print(f"  Temperature: {summary_24h['system'].get('avg_temp', 0):.1f}°C")

    # Trading stats
    trading = summary_24h['trading']
    if trading.get('total_trades', 0) > 0:
        print(f"\n💰 PAPER TRADING (24h):")
        print(f"  Total Trades: {trading.get('total_trades', 0)}")
        print(f"  Avg PnL: {trading.get('avg_pnl', 0):.2f}%")
        print(f"  Total PnL: {trading.get('total_pnl', 0):.2f}%")

    # Patterns
    patterns = analyzer.pattern_analysis()
    if patterns:
        print(f"\n🎯 TOP PATTERNS:")
        for i, p in enumerate(patterns[:5], 1):
            print(f"  {i}. {p['pattern_name']}: {p['avg_success']*100:.1f}% (seen {p['total_occurrences']} times)")

    # Comparison
    comp = analyzer.comparison_report()
    print(f"\n📈 BACKTEST vs PAPER TRADING:")
    print(f"  Backtest WR: {comp['backtest_win_rate']:.1f}%")
    print(f"  Paper WR: {comp['paper_win_rate']:.1f}%")
    print(f"  Status: {comp['comparison']['status'].upper()}")

    # Export full report
    print("\n📄 Exporting full report...")
    analyzer.export_report()

    print("\n✅ Analysis complete!")
    analyzer.close()


if __name__ == '__main__':
    main()
