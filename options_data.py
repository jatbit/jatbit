"""옵션 데이터: 만기·체인·기초자산·금리·환율 불러오기, 시장 변수, 체인 계산(그릭스). (지표 격자 mgrid는 ui.py로 옮김)"""
import html
import io
import math
import re
import time
import urllib.request
from datetime import datetime, timedelta

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import streamlit as st

import ui
import streamlit.components.v1 as components

import bsm

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None

try:
    from zoneinfo import ZoneInfo

    NY = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001
    NY = None

MULT = 100  # 미국 주식옵션 1계약 = 100주
OPT_PRESETS = ["AAPL", "NVDA", "TSLA", "MSFT", "SPY", "QQQ"]
HV_WINDOWS = {"20일": 20, "60일": 60, "120일": 120, "1년(252일)": 252}
GARCH_PICK = "GARCH(1,1) 예측"
TYPE_KO = {True: "콜", False: "풋"}


# ======================================================================
# 데이터 불러오기 (느려도 되도록 넉넉히 재시도)
# ======================================================================


# ======================================================================
def _retry(fn, tries=5, base=2.0):
    err = None
    for i in range(tries):
        try:
            out = fn()
            if out is not None:
                return out
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(base * (i + 1))
    if err:
        raise err
    return None


def is_us_ticker(code: str) -> bool:
    code = str(code).strip().upper()
    return bool(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", code)) and "=" not in code and not code.startswith("^") \
        and code not in ("US500", "KS11", "KQ11", "DX-Y.NYB")


@st.cache_data(ttl=600, show_spinner="옵션 만기 목록을 불러오는 중... (Yahoo 응답이 느리면 몇 번 다시 시도합니다)")
def load_expiries(ticker: str) -> list:
    """빈 목록이 오면(일시적 차단일 수 있음) 다시 시도하고, 끝까지 비면 오류로 처리해 캐시에 남기지 않는다."""
    for i in range(5):
        try:
            exps = list(yf.Ticker(ticker).options or [])
            if exps:
                return exps
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0 * (i + 1))
    raise RuntimeError("만기 목록이 비어 있습니다 (옵션이 없는 종목이거나 Yahoo 응답 지연)")


@st.cache_data(ttl=300, show_spinner="옵션 가격을 불러오는 중... (시간이 걸릴 수 있습니다)")
def load_chain_raw(ticker: str, expiry: str) -> pd.DataFrame:
    ch = _retry(lambda: yf.Ticker(ticker).option_chain(expiry))
    c, p = ch.calls.copy(), ch.puts.copy()
    c["is_call"], p["is_call"] = True, False
    return pd.concat([c, p], ignore_index=True)


@st.cache_data(ttl=600, show_spinner="기초자산 가격을 불러오는 중...")
def load_underlying(ticker: str) -> dict:
    """현재가(최근 종가), 1년 이상 일별 종가, 최근 12개월 배당 합계."""
    close = None
    start = (datetime.now() - timedelta(days=800)).strftime("%Y-%m-%d")
    try:
        d = _retry(lambda: fdr.DataReader(ticker, start), tries=3)
        if d is not None and "Close" in d:
            close = d["Close"].dropna()
            close = close[close > 0]
    except Exception:  # noqa: BLE001
        close = None
    t = yf.Ticker(ticker)
    if close is None or len(close) < 30:
        h = _retry(lambda: t.history(period="2y", auto_adjust=False), tries=3)
        close = h["Close"].dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
    spot = float(close.iloc[-1])
    asof = str(pd.Timestamp(close.index[-1]).date())
    try:  # 장중이면 Yahoo의 최근 체결가가 더 최신
        h1 = t.history(period="1d", interval="1m")
        if h1 is not None and len(h1):
            spot = float(h1["Close"].dropna().iloc[-1])
            asof = pd.Timestamp(h1.index[-1]).strftime("%Y-%m-%d %H:%M (뉴욕)")
    except Exception:  # noqa: BLE001
        pass
    div12 = 0.0
    try:
        dv = t.dividends
        if dv is not None and len(dv):
            idx = pd.to_datetime(dv.index).tz_localize(None)
            dv = pd.Series(dv.values, index=idx)
            div12 = float(dv[dv.index >= pd.Timestamp.now() - pd.Timedelta(days=365)].sum())
    except Exception:  # noqa: BLE001
        pass
    return {"spot": spot, "asof": asof, "close": close, "div12": div12}


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_us_rate_3m() -> tuple:
    """FRED 3개월 미 국채(DGS3MO) 최근값. 실패하면 None."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
            data = pd.read_csv(io.StringIO(raw), na_values=["."]).dropna()
            return float(data.iloc[-1, 1]) / 100, str(data.iloc[-1, 0])
        except Exception:  # noqa: BLE001
            time.sleep(1.0)
    return None


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_usdkrw() -> tuple:
    try:
        d = fdr.DataReader("USD/KRW", (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d"))
        s = d["Close"].dropna()
        return float(s.iloc[-1]), str(s.index[-1].date())
    except Exception:  # noqa: BLE001
        return None


def years_to_expiry(expiry: str) -> float:
    """만기일 뉴욕 16:00까지 남은 기간(연, 달력일/365)."""
    exp = pd.Timestamp(expiry)
    if NY is not None:
        exp_dt = datetime(exp.year, exp.month, exp.day, 16, 0, tzinfo=NY)
        now = datetime.now(NY)
    else:
        exp_dt = datetime(exp.year, exp.month, exp.day, 21, 0)
        now = datetime.utcnow()
    return max((exp_dt - now).total_seconds() / (365.0 * 86400), 1.0 / (365 * 24))


# ======================================================================
# 시장 환경(기초자산·금리·배당·변동성)과 체인 계산
# ======================================================================
def market_params(ticker: str, key: str, show: bool = True) -> dict | None:
    """기초자산 정보 + 사용자가 고친 r, q, 이론가용 변동성. show=True면 화면에 입력칸과 지표를 그리고,
    False면 그리지 않고 (이전에 고친 값이 있으면 그 값, 없으면 기본값으로) 계산만 한다."""
    try:
        u = load_underlying(ticker)
    except Exception as e:  # noqa: BLE001
        st.error(f"{ticker} 기초자산 가격을 불러오지 못했습니다: {e}")
        return None
    hv = {k: bsm.hist_vol(u["close"], w) for k, w in HV_WINDOWS.items()}
    try:
        import garch

        gf = garch.fit_close(u["close"])
    except Exception:  # noqa: BLE001
        gf = None
    g63 = garch.term_vol(gf, 63) if gf else float("nan")
    rr = load_us_rate_3m()
    r_default = rr[0] if rr else 0.04
    q_default = u["div12"] / u["spot"] if u["spot"] > 0 else 0.0
    if not show:
        ss = st.session_state
        r_pct = float(ss.get(f"{key}_r_{ticker}", round(r_default * 100, 3)))
        q_pct = float(ss.get(f"{key}_q_{ticker}", round(q_default * 100, 3)))
        hv_pick = ss.get(f"{key}_hvpick_{ticker}", GARCH_PICK if gf else list(HV_WINDOWS)[1])
        sig_theo = (float(ss.get(f"{key}_hvman_{ticker}", 30.0)) / 100 if hv_pick == "직접 입력" else
                    g63 if hv_pick == GARCH_PICK else hv.get(hv_pick, float("nan")))
        if not (sig_theo == sig_theo) or sig_theo <= 0:
            sig_theo = hv.get("60일", 0.3) if hv.get("60일", 0) == hv.get("60일", 0) else 0.3
        return {"S": u["spot"], "asof": u["asof"], "r": math.log1p(r_pct / 100), "r_simple": r_pct / 100,
                "q": q_pct / 100, "sig_theo": sig_theo, "hv": hv, "hv_pick": hv_pick, "garch": gf}

    mgrid([("기초자산 가격", f"{u['spot']:,.2f}", f"기준: {u['asof']}")] +
          [(f"HV {k}", f"{v * 100:.1f}%" if v == v else "-", f"역사적 변동성 {k}") for k, v in hv.items()]
          + ([("GARCH 3개월", f"{g63 * 100:.1f}%", "GARCH(1,1)로 예측한 앞으로 63거래일 평균 변동성")] if gf else []), cols=6 if gf else 5)

    box = st.popover("설정 · 금리·배당·변동성") if hasattr(st, "popover") else st.expander("설정 · 금리·배당·변동성")
    with box:
        c1, c2, c3 = st.columns(3)
    r_pct = c1.number_input("무위험금리 r (%)", value=round(r_default * 100, 3), step=0.05, format="%.3f",
                            key=f"{key}_r_{ticker}",
                            help=(f"FRED 미 국채 3개월 {rr[0] * 100:.2f}% ({rr[1]} 기준)" if rr else "FRED 조회 실패로 4% 기본값") + ", 연속복리로 환산해 사용")
    q_pct = c2.number_input("배당수익률 q (%)", value=round(q_default * 100, 3), step=0.05, format="%.3f",
                            key=f"{key}_q_{ticker}", help=f"최근 12개월 배당 {u['div12']:.2f} ÷ 현재가")
    hv_opts = ([GARCH_PICK] if gf else []) + list(HV_WINDOWS) + ["직접 입력"]
    if st.session_state.get(f"{key}_hvpick_{ticker}") not in hv_opts:
        st.session_state.pop(f"{key}_hvpick_{ticker}", None)
    hv_pick = c3.selectbox("이론가에 쓸 변동성", hv_opts, index=0 if gf else 1, key=f"{key}_hvpick_{ticker}",
                           help="GARCH(1,1): 최근 3년 수익률로 추정해 '만기까지 평균 변동성'을 예측 (체인에서는 고른 만기에 맞춤)")
    if hv_pick == "직접 입력":
        sig_theo = box.number_input("이론가 변동성 (%)", value=round((hv["60일"] if hv["60일"] == hv["60일"] else 0.3) * 100, 2),
                                   step=0.5, key=f"{key}_hvman_{ticker}") / 100
    elif hv_pick == GARCH_PICK:
        sig_theo = g63
    else:
        sig_theo = hv[hv_pick]
    if key == "chain":
        box.radio("그릭스 기준 변동성", ["IV", "HV"], horizontal=True, key="opt_greek_basis",
                  format_func=lambda x: "내재(IV)" if x == "IV" else "이론가 변동성", help="IV: 각 옵션의 내재변동성으로 그릭스 계산 · 이론가 변동성: 위에서 고른 변동성(GARCH 또는 HV)으로 계산")
    if not (sig_theo == sig_theo) or sig_theo <= 0:
        sig_theo = 0.3
    return {"S": u["spot"], "asof": u["asof"], "r": math.log1p(r_pct / 100), "r_simple": r_pct / 100,
            "q": q_pct / 100, "sig_theo": sig_theo, "hv": hv, "hv_pick": hv_pick, "garch": gf}


def build_chain(raw: pd.DataFrame, P: dict, T: float, greek_basis: str = "IV") -> pd.DataFrame:
    d = raw.copy()
    for c in ("bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility", "strike"):
        if c not in d:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    bid, ask, last = d["bid"].fillna(0), d["ask"].fillna(0), d["lastPrice"]
    has_q = (bid > 0) & (ask > 0) & (ask >= bid)
    d["mid"] = np.where(has_q, (bid + ask) / 2, np.nan)
    d["mkt"] = np.where(has_q, d["mid"], last)
    d["mkt_src"] = np.where(has_q, "중간가", "최근체결가")
    S, r, q = P["S"], P["r"], P["q"]
    K, c = d["strike"].values, d["is_call"].values.astype(bool)
    d["iv"] = bsm.implied_vol(d["mkt"].values, S, K, T, r, q, c)
    d["theo"] = bsm.price(S, K, T, r, q, P["sig_theo"], c)
    d["diff"] = d["mkt"] - d["theo"]
    sig_g = np.where(np.isfinite(d["iv"]), d["iv"], P["sig_theo"]) if greek_basis == "IV" else np.full(len(d), P["sig_theo"])
    g = bsm.greeks(S, K, T, r, q, sig_g, c)
    for k, v in g.items():
        d[k] = v
    d["sig_g"] = sig_g
    d["itm"] = np.where(c, K < S, K > S)
    return d.sort_values(["is_call", "strike"], ascending=[False, True]).reset_index(drop=True)


# ======================================================================
# 체인 표 (HTML, 시장가에 마우스를 올리면 그릭스 팝업)
# ======================================================================


def _is_dark() -> bool:
    try:
        t = st.context.theme.type
        if t:
            return t == "dark"
    except Exception:  # noqa: BLE001
        pass
    return (st.get_option("theme.base") or "dark") != "light"


from ui import mgrid  # noqa: E402,F401  지표 격자는 ui.py로 옮김 (예전 이름 그대로 쓸 수 있게)


# 보유 포지션 표의 빈 양식 (옵션 체인의 "포지션에 추가"와 헷지 화면이 같이 씀)
def settings_box():
    return st.popover("설정 · 금리·배당·변동성") if hasattr(st, "popover") else st.expander("설정 · 금리·배당·변동성")


def _empty_opts() -> pd.DataFrame:
    return pd.DataFrame({"기초자산": pd.Series(dtype=str), "만기": pd.Series(dtype=str), "종류": pd.Series(dtype=str),
                         "행사가": pd.Series(dtype=float), "수량(계약)": pd.Series(dtype=int)})


def _empty_stocks() -> pd.DataFrame:
    return pd.DataFrame({"종목코드": pd.Series(dtype=str), "종목명": pd.Series(dtype=str),
                         "보유 주식 수": pd.Series(dtype=float), "비고": pd.Series(dtype=str)})


__all__ = [n for n in dir() if not n.startswith("__")]  # 밑줄로 시작하는 도우미도 다른 모듈에서 쓸 수 있게
