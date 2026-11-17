"""차트 위 그리기 도구: 피보나치(채널·추세 기반 확장·되돌림)와 일목균형표.

- 피보나치 기준점은 지그재그(설정한 % 이상 되돌린 고점·저점)로 자동으로 잡거나, 날짜로 직접 지정한다.
- 레벨은 트레이딩뷰처럼 소수·음수를 자유롭게 넣을 수 있다 (예: -0.618, 0, 0.5, 1.618, 4.236).
- 일목균형표는 트레이딩뷰 기본 계산과 같게 선행스팬을 (기준-1)봉 앞으로, 후행스팬을 (기준-1)봉 뒤로 옮긴다.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

FIB_KINDS = ["피보나치 채널", "추세 기반 피보나치 확장", "피보나치 되돌림"]
FIB_DEFAULT = "0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.618, 2.618"
# 트레이딩뷰 피보나치 도구의 레벨 색 순서와 비슷하게
FIB_COLORS = ["#787B86", "#F23645", "#FF9800", "#4CAF50", "#089981", "#00BCD4", "#787B86", "#2962FF",
              "#F23645", "#9C27B0", "#E91E63", "#FFEB3B"]


def parse_levels(text: str, max_n: int = 24, lo: float = -50.0, hi: float = 50.0) -> list:
    """'0, 0.236, -0.618, 1.618' → [0, 0.236, -0.618, 1.618]. 쉼표·공백·세미콜론 구분, 중복 제거."""
    out = []
    for tok in re.split(r"[,\s;]+", str(text or "").replace("−", "-")):
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            continue
        if np.isfinite(v) and lo <= v <= hi and all(abs(v - x) > 1e-12 for x in out):
            out.append(v)
    return out[:max_n] or [0.0, 0.382, 0.5, 0.618, 1.0]


def zigzag(high: pd.Series, low: pd.Series, pct: float) -> list:
    """고가·저가에서 pct% 이상 반대로 움직인 지점을 전환점으로 본다.
    반환: [{"i": 봉 위치, "p": 가격, "k": "H"|"L", "final": 확정 여부}] (시간 순, 마지막 점은 아직 진행 중일 수 있음)."""
    h = pd.to_numeric(high, errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(low, errors="coerce").to_numpy(dtype=float)
    n = len(h)
    if n < 3:
        return []
    th = max(float(pct), 0.01) / 100.0
    piv = []
    trend = 0
    hi_i = lo_i = 0
    i = 1
    # 첫 방향 정하기
    while i < n and trend == 0:
        if np.isfinite(h[i]) and (not np.isfinite(h[hi_i]) or h[i] > h[hi_i]):
            hi_i = i
        if np.isfinite(lo[i]) and (not np.isfinite(lo[lo_i]) or lo[i] < lo[lo_i]):
            lo_i = i
        if np.isfinite(lo[lo_i]) and np.isfinite(h[i]) and h[i] >= lo[lo_i] * (1 + th) and lo_i < i:
            piv.append({"i": lo_i, "p": float(lo[lo_i]), "k": "L", "final": True})
            trend, ext_i = 1, i
        elif np.isfinite(h[hi_i]) and np.isfinite(lo[i]) and lo[i] <= h[hi_i] * (1 - th) and hi_i < i:
            piv.append({"i": hi_i, "p": float(h[hi_i]), "k": "H", "final": True})
            trend, ext_i = -1, i
        i += 1
    if trend == 0:
        return []
    for j in range(i, n):
        if trend == 1:
            if np.isfinite(h[j]) and h[j] >= h[ext_i]:
                ext_i = j
            elif np.isfinite(lo[j]) and lo[j] <= h[ext_i] * (1 - th):
                piv.append({"i": ext_i, "p": float(h[ext_i]), "k": "H", "final": True})
                trend, ext_i = -1, j
        else:
            if np.isfinite(lo[j]) and lo[j] <= lo[ext_i]:
                ext_i = j
            elif np.isfinite(h[j]) and h[j] >= lo[ext_i] * (1 + th):
                piv.append({"i": ext_i, "p": float(lo[ext_i]), "k": "L", "final": True})
                trend, ext_i = 1, j
    last = {"i": ext_i, "p": float(h[ext_i] if trend == 1 else lo[ext_i]), "k": "H" if trend == 1 else "L", "final": False}
    if last["i"] != piv[-1]["i"]:
        piv.append(last)
    return piv


def _need(kind: str) -> int:
    return 2 if kind == FIB_KINDS[2] else 3


def auto_anchors(kind: str, piv: list, nth: int = 1):
    """가장 최근 파동(nth=1) 또는 그 이전 파동의 전환점으로 기준점을 고른다."""
    k = _need(kind)
    if kind == FIB_KINDS[1]:  # 확장: 1→2 파동과 3번 되돌림이 끝난 뒤라야 의미가 있으므로 아직 진행 중인 마지막 점은 뺀다
        piv = [p for p in piv if p.get("final")]
    end = len(piv) - max(int(nth), 1)
    if end + 1 < k:
        return None
    return piv[end + 1 - k:end + 1]


def manual_anchors(kind: str, df: pd.DataFrame, dates: list):
    """날짜로 고른 기준점 → 봉 위치·가격. 오르는 파동이면 시작=저가·끝=고가, 내리는 파동이면 반대로 잡는다."""
    k = _need(kind)
    dates = [d for d in dates[:k] if d is not None]
    if len(dates) < k or df is None or df.empty:
        return None
    idx = pd.DatetimeIndex(df.index)
    pos = []
    for d in dates:
        p = int(idx.searchsorted(pd.Timestamp(d)))
        pos.append(min(max(p, 0), len(idx) - 1))
    c = df["Close"].to_numpy(dtype=float)
    hi = df["High"].to_numpy(dtype=float) if "High" in df else c
    lo = df["Low"].to_numpy(dtype=float) if "Low" in df else c
    if kind == FIB_KINDS[0]:  # 채널: 1·2번 점이 같은 쪽(추세선), 3번 점이 폭
        up = c[pos[2]] > (c[pos[0]] + c[pos[1]]) / 2
        kinds = ["L", "L", "H"] if up else ["H", "H", "L"]
        order = [pos[0], pos[2], pos[1]]  # 시간 순 아님: 함수 안에서 1·2를 추세선, 3을 폭으로 쓰도록 저장 순서를 맞춤
        k_ord = [kinds[0], kinds[2], kinds[1]]
    else:
        up = c[pos[1]] > c[pos[0]]
        k_ord = (["L", "H", "L"] if up else ["H", "L", "H"])[:k]
        order = pos
    return [{"i": i, "p": float(hi[i] if kk == "H" else lo[i]), "k": kk, "final": True} for i, kk in zip(order, k_ord)]


def fib_lines(kind: str, anc: list, levels: list, n_bars: int, extend: int = 0):
    """레벨별 선: [(레벨, 시작 봉 위치, 가격 배열)]. 봉 위치 기준 직선이라 주말·휴일을 빼고 그린 차트에서도 곧게 보인다."""
    if not anc:
        return []
    end = n_bars - 1 + max(int(extend), 0)
    out = []
    if kind == FIB_KINDS[0]:
        a, c, b = anc[0], anc[1], anc[2]  # 시간 순으로 1(추세선)·폭·2(추세선)
        if b["i"] == a["i"]:
            return []
        slope = (b["p"] - a["p"]) / (b["i"] - a["i"])
        xs = np.arange(a["i"], end + 1)
        base = a["p"] + slope * (xs - a["i"])
        d = c["p"] - (a["p"] + slope * (c["i"] - a["i"]))
        for lv in levels:
            out.append((lv, a["i"], base + lv * d))
    elif kind == FIB_KINDS[1]:
        a, b, c = anc
        xs = np.arange(c["i"], end + 1)
        for lv in levels:
            out.append((lv, c["i"], np.full(len(xs), c["p"] + lv * (b["p"] - a["p"]))))
    else:
        a, b = anc
        xs = np.arange(a["i"], end + 1)
        for lv in levels:
            out.append((lv, a["i"], np.full(len(xs), b["p"] - lv * (b["p"] - a["p"]))))
    return out


def future_index(idx: pd.DatetimeIndex, n: int, weekly: bool = False) -> pd.DatetimeIndex:
    if n <= 0 or not len(idx):
        return pd.DatetimeIndex([])
    last = pd.Timestamp(idx[-1])
    if weekly:
        return pd.date_range(last + pd.Timedelta(days=7), periods=n, freq="W-FRI")
    return pd.bdate_range(last + pd.Timedelta(days=1), periods=n)


def ichimoku(ohlc: pd.DataFrame, t: int = 9, k: int = 26, s: int = 52, d: int = 26, weekly: bool = False) -> dict:
    """전환선·기준선·선행스팬1·2·후행스팬. 선행스팬은 (d-1)봉 앞으로, 후행스팬은 (d-1)봉 뒤로 (트레이딩뷰와 같은 방식)."""
    H, L, C = ohlc["High"].astype(float), ohlc["Low"].astype(float), ohlc["Close"].astype(float)
    mid = lambda n: (H.rolling(n, min_periods=n).max() + L.rolling(n, min_periods=n).min()) / 2  # noqa: E731
    tenkan, kijun = mid(t), mid(k)
    off = max(int(d) - 1, 0)
    fut = future_index(ohlc.index, off, weekly)
    full = ohlc.index.append(fut)
    span_a = ((tenkan + kijun) / 2).reindex(full).shift(off)
    span_b = mid(s).reindex(full).shift(off)
    chikou = C.shift(-off)
    return {"tenkan": tenkan, "kijun": kijun, "span_a": span_a, "span_b": span_b, "chikou": chikou}


def cloud_segments(a: pd.Series, b: pd.Series):
    """선행스팬1·2 사이 구름을 색별로 나눈 구간들: [(index, a값, b값, 양운 여부)]. 교차점은 양쪽 구간에 함께 넣어 끊기지 않게."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if j.empty:
        return []
    up = (j["a"] >= j["b"]).to_numpy()
    segs, start = [], 0
    for i in range(1, len(j) + 1):
        if i == len(j) or up[i] != up[start]:
            stop = min(i + 1, len(j))
            seg = j.iloc[start:stop]
            segs.append((seg.index, seg["a"].to_numpy(), seg["b"].to_numpy(), bool(up[start])))
            start = i
    return segs


__all__ = [n for n in dir() if not n.startswith("__")]
