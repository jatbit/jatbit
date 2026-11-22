"""GARCH(1,1) 변동성 예측 (Student-t 오차, 분산 목표화).
σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1},  ω = V_L·(1 − α − β)  (V_L = 표본 분산 → 장기 평균 분산)
- fit(x): 일별 변화(로그수익률 또는 금리 변화)에 최대우도로 α·β·ν 추정
- forecast(f, h): 1~h일 뒤 분산 예측,  term_vol(f, days): 그 기간 평균 변동성(연환산)
- avg_vol_path(f, x, H): 과거 각 시점에서 '앞으로 H일 평균 변동성' 예측 (백테스트용)
"""
import math

import numpy as np
import pandas as pd
import streamlit as st

MIN_OBS = 250


def _filter(e2: np.ndarray, var0: float, a: float, b: float, w: float) -> np.ndarray:
    """조건부 분산 σ²_t (t = 0..n-1),  σ²_0 = var0."""
    from scipy.signal import lfilter

    n = len(e2)
    s2 = np.empty(n)
    s2[0] = var0
    if n > 1:
        x = w + a * e2[:-1]
        s2[1:] = lfilter([1.0], [1.0, -b], x, zi=[b * var0])[0]
    return np.maximum(s2, 1e-12)


def _nll(p, e, e2, var0, VL):
    from scipy.special import gammaln

    a, b, nu = p
    if a < 0 or b < 0 or a + b >= 0.9995 or nu <= 2.05:
        return 1e12
    s2 = _filter(e2, var0, a, b, VL * (1 - a - b))
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    ll = c - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(e2 / (s2 * (nu - 2)))
    v = -float(ll.sum())
    return v if np.isfinite(v) else 1e12


def fit(x, years: float | None = 3.0, periods: int = 252) -> dict | None:
    """x: 일별 변화 Series/배열. 반환 dict 또는 None(자료 부족·추정 실패)."""
    from scipy.optimize import minimize

    s = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if years:
        s = s.iloc[-int(years * periods):]
    if len(s) < MIN_OBS:
        return None
    scale = 1.0 / max(float(s.std()), 1e-12)  # 숫자 안정용 (분산 ~1로)
    r = s.values * scale
    mu = float(r.mean())
    e = r - mu
    e2 = e * e
    VL = float(e2.mean())
    best = None
    for x0 in ((0.05, 0.90, 8.0), (0.10, 0.85, 5.0), (0.03, 0.95, 12.0)):
        try:
            res = minimize(_nll, x0, args=(e, e2, VL, VL), method="L-BFGS-B",
                           bounds=[(1e-6, 0.5), (0.0, 0.9989), (2.1, 100.0)])
        except Exception:  # noqa: BLE001
            continue
        if res.success or np.isfinite(res.fun):
            if best is None or res.fun < best.fun:
                best = res
    if best is None or best.fun >= 1e11:
        return None
    a, b, nu = (float(v) for v in best.x)
    if a + b >= 0.9995:
        return None
    w = VL * (1 - a - b)
    s2 = _filter(e2, VL, a, b, w)
    nxt = w + a * e2[-1] + b * s2[-1]
    k = 1.0 / (scale * scale)  # 원래 단위로
    p = a + b
    return {"alpha": a, "beta": b, "nu": nu, "persist": p, "omega": w * k, "VL": VL * k, "mu": mu / scale,
            "s2": pd.Series(s2 * k, index=s.index), "next": float(nxt * k), "z": pd.Series(e / np.sqrt(s2), index=s.index),
            "half_life": math.log(0.5) / math.log(p) if 0 < p < 1 else float("inf"), "n": len(s), "periods": periods,
            "last_e": float(e[-1] / scale)}


def forecast(f: dict, h: int) -> np.ndarray:
    """1..h일 뒤 일별 분산 예측 E[σ²_{T+k}] = V_L + p^{k−1}(σ²_{T+1} − V_L)."""
    k = np.arange(1, max(int(h), 1) + 1)
    return f["VL"] + f["persist"] ** (k - 1) * (f["next"] - f["VL"])


def term_vol(f: dict, days: float) -> float:
    """앞으로 days 거래일 평균 변동성 (연환산)."""
    return float(math.sqrt(max(forecast(f, max(int(round(days)), 1)).mean(), 0.0) * f["periods"]))


def avg_vol_path(f: dict, x, H: int) -> pd.Series:
    """과거 각 날짜 t에서 그날까지 정보로 예측한 '다음 H일 평균 변동성(연환산)'. 모수는 f의 것(전체 표본 추정)을 쓴다."""
    s = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    a, b, VL = f["alpha"], f["beta"], f["VL"]
    e = s.values - f["mu"]
    e2 = e * e
    s2 = _filter(e2, VL, a, b, VL * (1 - a - b))
    nxt = VL * (1 - a - b) + a * e2 + b * s2  # t일 장 마감 뒤 내일 분산 예측
    p = a + b
    H = max(int(H), 1)
    fac = (1 - p ** H) / ((1 - p) * H) if p < 1 else 1.0
    avg = VL + (nxt - VL) * fac
    return pd.Series(np.sqrt(np.maximum(avg, 0) * f["periods"]), index=s.index)


def log_returns(close) -> pd.Series:
    c = pd.Series(close).dropna()
    c = c[c > 0]
    return np.log(c).diff().dropna()


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fit_close(close: pd.Series, years: float = 3.0) -> dict | None:
    """종가 → 로그수익률 GARCH (6시간 캐시)."""
    return fit(log_returns(close), years)


def vol_for(close, days: float, fallback_window: int = 60) -> tuple:
    """(연환산 변동성, 설명). GARCH가 안 되면 역사적 변동성."""
    try:
        f = fit_close(pd.Series(close).dropna())
    except Exception:  # noqa: BLE001
        f = None
    if f is not None:
        return term_vol(f, days), "GARCH(1,1)"
    import bsm

    hv = bsm.hist_vol(close, fallback_window)
    return (hv if hv == hv else 0.3), f"HV {fallback_window}일"


__all__ = [n for n in dir() if not n.startswith("__")]
