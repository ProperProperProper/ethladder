"""Bounded paper-account history with session-wide sampled drawdowns."""
from collections import deque
from datetime import datetime
from math import isfinite
from typing import NamedTuple

from symbot_python.logging_setup import MELBOURNE_TZ


class Sample(NamedTuple):
    timestamp: float
    equity: float
    realised: float
    unrealised: float
    drawdown: float
    drawdown_percent: float


class EquityHistory:
    def __init__(self, timestamp: float, balance: float):
        self.start_balance = balance
        self.peak = balance
        self.max_drawdown = 0.0
        self.max_drawdown_percent = 0.0
        self.points = deque([Sample(timestamp, balance, 0, 0, 0, 0)], maxlen=1440)

    def record(self, timestamp: float, equity: float, realised: float | None = None,
               unrealised: float = 0.0):
        if realised is None:
            realised = equity - self.start_balance - unrealised
        if not all(isfinite(v) for v in (timestamp, equity, realised, unrealised)):
            return
        self.peak = max(self.peak, equity)
        drawdown = max(0.0, self.peak - equity)
        percent = drawdown / self.peak * 100 if self.peak > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, drawdown)
        self.max_drawdown_percent = max(self.max_drawdown_percent, percent)
        self.points.append(Sample(timestamp, equity, realised, unrealised, drawdown, percent))

    @staticmethod
    def _stamp(t):
        return datetime.fromtimestamp(t, MELBOURNE_TZ).strftime('%H:%M:%S')

    def _chart(self, title, series, baseline=0.0, unit='USDT'):
        points = list(self.points)
        values = [getattr(p, field) for _, field, _ in series for p in points] + [baseline]
        low, high = min(values), max(values)
        padding = max((high - low) * 0.12, 0.01)
        low, high = low - padding, high + padding
        first, last = points[0].timestamp, points[-1].timestamp
        def x(t):
            return 80 + (t - first) / max(last - first, 1) * 790
        def y(v):
            return 210 - (v - low) / (high - low) * 180
        parts = [f'<h4>{title} ({unit})</h4><div style="display:flex;gap:18px;flex-wrap:wrap">']
        for label, _, color in series:
            parts.append(f'<span style="color:{color}">━ {label}</span>')
        parts.append(f'</div><svg viewBox="0 0 900 250" role="img" aria-label="{title}" '
                     'style="display:block;width:100%;background:#8881;border-radius:8px">')
        for i in range(4):
            value = low + (high - low) * i / 3
            yy = y(value)
            parts.append(f'<line x1="80" x2="870" y1="{yy:.2f}" y2="{yy:.2f}" stroke="#8883"/>'
                         f'<text x="70" y="{yy+4:.2f}" text-anchor="end" fill="currentColor" font-size="12">{value:.2f}</text>')
        yy = y(baseline)
        parts.append(f'<line x1="80" x2="870" y1="{yy:.2f}" y2="{yy:.2f}" stroke="#888" stroke-dasharray="5,5"><title>Baseline: {baseline:.2f} {unit}</title></line>')
        for label, field, color in series:
            coords = ' '.join(f'{x(p.timestamp):.2f},{y(getattr(p, field)):.2f}' for p in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
            for p in points:
                value = getattr(p, field)
                parts.append(f'<circle cx="{x(p.timestamp):.2f}" cy="{y(value):.2f}" r="3" fill="{color}" fill-opacity="0.25"><title>{self._stamp(p.timestamp)} — {label}: {value:.4f} {unit}</title></circle>')
        parts.append(f'<text x="80" y="238" fill="currentColor" font-size="12">{self._stamp(first)}</text>'
                     f'<text x="870" y="238" text-anchor="end" fill="currentColor" font-size="12">{self._stamp(last)} Melbourne</text></svg>')
        return ''.join(parts)

    def render(self) -> str:
        latest = self.points[-1]
        cards = [
            ('Equity', f'{latest.equity:,.2f} USDT'),
            ('Realised P/L · booked net', f'{latest.realised:+,.2f} USDT'),
            ('Unrealised P/L · net funding', f'{latest.unrealised:+,.2f} USDT'),
            ('Current drawdown', f'{latest.drawdown:,.2f} USDT ({latest.drawdown_percent:.2f}%)'),
            ('Maximum drawdown · session', f'{self.max_drawdown:,.2f} USDT / {self.max_drawdown_percent:.2f}%'),
            ('Peak equity · session', f'{self.peak:,.2f} USDT'),
        ]
        parts = ['<div class="stat-grid">']
        for label, value in cards:
            parts.append(f'<div class="stat-card"><div class="label">{label}</div><div class="value" style="font-size:1.05rem">{value}</div></div>')
        parts.append(f'</div><div style="color:#888;font-size:0.8rem">Last sample: {self._stamp(latest.timestamp)} Melbourne</div>')
        parts.append(self._chart('Paper account equity over time', [('Equity', 'equity', '#388bcc')], self.start_balance))
        parts.append(self._chart('Realised and unrealised P/L', [
            ('Realised', 'realised', '#2e9e4f'), ('Unrealised', 'unrealised', '#cc8a00')]))
        parts.append(self._chart('Drawdown from session peak', [('Drawdown', 'drawdown_percent', '#c93b3b')], unit='%'))
        return ''.join(parts)
