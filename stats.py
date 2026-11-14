"""수익률·무위험수익률(국고채 3년·미국채 10년)·벤치마크·CAPM·위험 지표(VaR·ES 등)."""
import math
import re
import time
from datetime import date, datetime, timedelta, timezone

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ui  # 화면 공통 요소 (ui.py)
from rates import add_rates, rate_controls
# 옵션·리서치·백테스트·채권·시장 모듈은 그 페이지를 열 때 불러온다 (첫 화면을 빠르게)

from data_load import *  # noqa: F401,F403
from period import kst_today

TRADING_DAYS = 252


def _pct(x: float, sign: bool = False) -> str:
    if x != x:  # NaN
        return "-"
    return f"{x * 100:+.2f}%" if sign else f"{x * 100:.2f}%"


def _with_date(text: str, d) -> str:
    return f'{text}<br><span style="font-size:0.6rem;opacity:0.7">{d:%Y-%m-%d}</span>'


# 무위험수익률: 조회 기간의 마지막 영업일 기준 금리를 가져온다.
#  - 한국 종목: 국고채 3년물 (한국은행 ECOS 일별 시장금리, 공개 sample 인증키)
#  - 해외 티커: 미국 국채 10년물 (FRED DGS10)
# 금리를 가져오지 못하면 아래 고정값을 쓰고, 화면에 그 사실을 표시한다.
RF_KR = 0.03613
RF_KR_ASOF = "2026-11-06"
RF_US_FALLBACK = 0.0487
RF_US_FALLBACK_ASOF = "2026-11-06"


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _load_fred_dgs10() -> pd.Series:
    import io
    import urllib.request

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    with urllib.request.urlopen(url, timeout=8) as resp:
        raw = resp.read().decode("utf-8")
    data = pd.read_csv(io.StringIO(raw), na_values=["."]).dropna()
    return pd.Series(data.iloc[:, 1].astype(float).values / 100, index=pd.to_datetime(data.iloc[:, 0]))


def us10y_asof(asof: date):
    """asof 이전(당일 포함) 가장 최근 미국 10년물 금리 → (금리, 기준일) 또는 None."""
    try:
        s = _load_fred_dgs10()
        s = s[s.index <= pd.Timestamp(asof)]
        if s.empty:
            return None
        return float(s.iloc[-1]), s.index[-1].strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _load_kr3y(start: str, end: str) -> list:
    import json
    import urllib.request

    url = f"https://ecos.bok.or.kr/api/StatisticSearch/sample/json/kr/1/10/817Y002/D/{start}/{end}/010200000"
    with urllib.request.urlopen(url, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    rows = data["StatisticSearch"]["row"]
    return [(r["TIME"], float(r["DATA_VALUE"])) for r in rows if r.get("DATA_VALUE")]


def kr3y_asof(asof: date):
    """asof 이전(당일 포함) 가장 최근 국고채 3년물 금리 → (금리, 기준일) 또는 None.
    sample 인증키는 한 번에 10건까지라서 최근 12일 구간만 요청한다."""
    try:
        start = (asof - timedelta(days=12)).strftime("%Y%m%d")
        end = asof.strftime("%Y%m%d")
        rows = [x for x in _load_kr3y(start, end) if x[0] <= end]
        if not rows:
            return None
        t, v = max(rows)
        return v / 100, f"{t[:4]}-{t[4:6]}-{t[6:]}"
    except Exception:  # noqa: BLE001
        return None


def risk_free_for(domestic: bool, asof: date) -> tuple:
    """종목이 속한 나라의 (조회 기간 마지막 영업일 기준 무위험수익률, 설명 문구)."""
    if domestic:
        got = kr3y_asof(asof)
        if got:
            return got[0], f"한국 국고채 3년물 {got[0] * 100:.2f}% ({got[1]} 기준 · 조회 기간 마지막 영업일 · 한국은행 ECOS)"
        return RF_KR, (
            f"한국 국고채 3년물 {RF_KR * 100:.2f}% (금리 조회 실패로 {RF_KR_ASOF} 고정값 사용, "
            "조회 기간 마지막 영업일 값이 아님)"
        )
    got = us10y_asof(asof)
    if got:
        return got[0], (
            f"미국 국채 10년물 {got[0] * 100:.2f}% ({got[1]} 기준 · 조회 기간 마지막 영업일 · FRED) "
            "· 해외 티커는 모두 미국 금리 적용"
        )
    return RF_US_FALLBACK, (
        f"미국 국채 10년물 {RF_US_FALLBACK * 100:.2f}% (금리 조회 실패로 {RF_US_FALLBACK_ASOF} 고정값 사용, "
        "조회 기간 마지막 영업일 값이 아님)"
    )


def benchmark_for(market: str, domestic: bool) -> tuple:
    """종목이 속한 시장의 벤치마크 지수 (코드, 이름)."""
    if not domestic:
        return "US500", "S&P 500"
    if str(market).upper().startswith("KOSDAQ"):
        return "KQ11", "코스닥 지수"
    return "KS11", "코스피 지수"


def capm_stats(r: pd.Series, m: pd.Series, rf: float):
    """같은 날짜로 맞춘 일별 수익률 r(종목/포트폴리오), m(시장)의 CAPM 지표 딕셔너리."""
    var_m = float(m.var(ddof=1))
    if var_m <= 0:
        return None
    beta = float(r.cov(m)) / var_m
    corr = float(r.corr(m))
    mean_a = float(r.mean()) * TRADING_DAYS
    mean_m = float(m.mean()) * TRADING_DAYS
    std_a = float(r.std(ddof=1)) * TRADING_DAYS ** 0.5
    std_m = float(m.std(ddof=1)) * TRADING_DAYS ** 0.5
    capm = rf + beta * (mean_m - rf)
    sys_risk = abs(beta) * std_m
    return {
        "beta": beta, "corr": corr, "r2": corr ** 2,
        "mean_a": mean_a, "mean_m": mean_m, "std_a": std_a, "std_m": std_m,
        "capm": capm, "alpha": mean_a - capm,
        "treynor": (mean_a - rf) / beta if beta != 0 else float("nan"),
        "sys": sys_risk, "idio": max(std_a ** 2 - sys_risk ** 2, 0.0) ** 0.5,
        "n": len(r),
    }


Z95, Z99 = 1.6449, 2.3263


def risk_metrics(r: pd.Series) -> dict:
    """1일 기준 VaR / ES (손실률, 양수). 과거 시뮬레이션(실제 수익률 분포)과 정규분포 가정 두 가지."""
    out = {}
    mu_d, sd_d = float(r.mean()), float(r.std(ddof=1))
    for cl, z, tag in ((0.95, Z95, "95"), (0.99, Z99, "99")):
        q = float(r.quantile(1 - cl))
        tail = r[r <= q]
        out[f"var{tag}"] = -q
        out[f"es{tag}"] = -float(tail.mean()) if len(tail) else -q
        out[f"nvar{tag}"] = -(mu_d - z * sd_d)
    out["worst"] = -float(r.min())
    out["n"] = len(r)
    return out


def risk_grid(k: dict) -> None:
    metric_grid([
        ("VaR 95% (1일, 과거)", _pct(k["var95"]), None, True),
        ("ES 95% (1일, 과거)", _pct(k["es95"]), None, True),
        ("VaR 99% (1일, 과거)", _pct(k["var99"]), None, True),
        ("ES 99% (1일, 과거)", _pct(k["es99"]), None, True),
        ("VaR 95% (정규분포)", _pct(k["nvar95"]), None, True),
        ("VaR 99% (정규분포)", _pct(k["nvar99"]), None, True),
        ("최악의 하루 손실", _pct(k["worst"]), None, True),
        ("표본 수", f"{k['n']:,}일", None, True),
    ], cols=4)


RISK_NOTE = "손실은 양수로 표시합니다. 조회 기간이 짧으면 99% 값은 최악의 하루 한두 개에 좌우되므로 참고용으로만 보세요."


def capm_grid(c: dict, name: str) -> None:
    metric_grid([
        ("벤치마크", name, None, True),
        ("베타 (β)", f"{c['beta']:.2f}", None, True),
        ("상관계수 (ρ)", f"{c['corr']:.2f}", None, True),
        ("결정계수 (R²)", f"{c['r2']:.2f}", None, True),
        ("시장 연환산 수익률", _pct(c["mean_m"], True), None, True),
        ("시장 연환산 표준편차", _pct(c["std_m"]), None, True),
        ("CAPM 기대수익률", _pct(c["capm"], True), None, True),
        ("젠센 알파 (연)", _pct(c["alpha"], True), None, True),
        ("트레이너 비율", "-" if c["treynor"] != c["treynor"] else f"{c['treynor']:.3f}", None, True),
        ("체계적 위험 (|β|σm)", _pct(c["sys"]), None, True),
        ("비체계적 위험", _pct(c["idio"]), None, True),
        ("공통 표본 수", f"{c['n']:,}일", None, True),
    ], cols=4)


def capm_note(name: str) -> str:
    return f"벤치마크: {name} · 조회 기간이 짧으면 값이 크게 흔들립니다."


def show_capm(df: pd.DataFrame, rf: float, bench: tuple) -> None:
    """CAPM 기본 지표: 벤치마크 지수 대비 베타, 알파 등 (일별 수익률 회귀)."""
    code, name = bench
    try:
        mk = load_prices(code, df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"))
        m = mk["Close"].dropna().pct_change().dropna()
    except Exception as e:  # noqa: BLE001
        st.caption(f"CAPM 지표: 벤치마크({name}) 데이터를 불러오지 못했습니다 ({e})")
        return
    r = df["Close"].dropna().pct_change().dropna()
    j = pd.concat([r.rename("r"), m.rename("m")], axis=1, join="inner").dropna()
    if len(j) < 20:
        st.caption(f"CAPM 지표: 벤치마크({name})와 겹치는 거래일이 {len(j)}일뿐이라 계산하지 않았습니다. 기간을 늘려 보세요.")
        return
    c = capm_stats(j["r"], j["m"], rf)
    if c is None:
        return
    st.markdown("**CAPM 기본 지표**")
    capm_grid(c, name)
    st.caption(capm_note(name))


def show_stats(df: pd.DataFrame, rf: float, rf_label: str, bench: tuple = None) -> None:
    """조회 기간의 기본 통계 (일별 종가 수익률 기준)."""
    close = df["Close"].dropna()
    r = close.pct_change().dropna()
    if len(r) < 2:
        st.info("통계를 계산하기에 데이터가 부족합니다. 기간을 늘려 보세요.")
        return

    n = len(r)
    mean_d = float(r.mean())
    std_d = float(r.std(ddof=1))
    mean_a = mean_d * TRADING_DAYS
    std_a = std_d * TRADING_DAYS ** 0.5
    total = float(close.iloc[-1] / close.iloc[0] - 1)
    years = n / TRADING_DAYS
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 and years > 0 else float("nan")
    dd = close / close.cummax() - 1
    mdd, mdd_date = float(dd.min()), dd.idxmin()
    best_date, worst_date = r.idxmax(), r.idxmin()
    up_ratio = float((r > 0).mean())
    sharpe = (mean_a - rf) / std_a if std_a > 0 else float("nan")

    metric_grid([
        ("기간 수익률", _pct(total, True), None, True),
        ("연환산 수익률(복리)", _pct(cagr, True), None, True),
        ("일평균 수익률", _pct(mean_d, True), None, True),
        ("연환산 기대수익률", _pct(mean_a, True), None, True),
        ("일간 표준편차", _pct(std_d), None, True),
        ("연환산 표준편차", _pct(std_a), None, True),
        ("최대 낙폭(MDD)", _with_date(_pct(mdd), mdd_date), None, True),
        ("최대 일간 상승", _with_date(_pct(float(r.max()), True), best_date), None, True),
        ("최대 일간 하락", _with_date(_pct(float(r.min()), True), worst_date), None, True),
        ("상승일 비율", _pct(up_ratio), None, True),
        (f"샤프비율 (무위험 {rf * 100:.2f}%)", "-" if sharpe != sharpe else f"{sharpe:.2f}", None, True),
        ("수익률 표본 수", f"{n:,}일", None, True),
    ], cols=4)
    st.caption(f"일별 종가 기준 과거 통계 · 무위험수익률: {rf_label}")
    if bench:
        show_capm(df, rf, bench)


__all__ = [n for n in dir() if not n.startswith("__")]  # 밑줄로 시작하는 도우미 함수도 다른 모듈에서 쓸 수 있게
