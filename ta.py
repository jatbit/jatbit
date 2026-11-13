"""기술적 지표: 이동평균, 볼린저밴드, RSI, MACD (주가 조회 차트와 스크리너에서 같이 씀)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    """(중심선, 상단, 하단)"""
    m = sma(s, n)
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return m, m + k * sd, m - k * sd


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """와일더 방식 RSI (0~100)."""
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(up.notna())


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """(MACD선, 시그널선, 히스토그램)"""
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


MA_COLORS = {5: "#E8A33D", 20: "#D64545", 60: "#2F855A", 120: "#6B46C1", 200: "#718096"}
PALETTE = ["#E8A33D", "#D64545", "#2F855A", "#6B46C1", "#718096", "#0987A0", "#B83280", "#975A16"]


def ma_color(n: int, i: int) -> str:
    return MA_COLORS.get(n, PALETTE[i % len(PALETTE)])


def parse_periods(text: str, default=(5, 20, 60, 120), lo: int = 2, hi: int = 400, max_n: int = 8) -> list:
    """'5, 20, 60' 같은 입력을 기간 목록으로. 잘못된 값은 빼고, 하나도 없으면 기본값."""
    import re

    out = []
    for tok in re.split(r"[^0-9]+", str(text or "")):
        if tok.isdigit():
            n = int(tok)
            if lo <= n <= hi and n not in out:
                out.append(n)
    return sorted(out)[:max_n] or list(default)
