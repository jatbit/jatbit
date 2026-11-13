"""채권 탭
1) 미국 국채 금리 곡선 (FRED): 만기별 금리, 1개월·1년 전과 비교, 장단기 금리차
2) 국채 조회: 만기를 고르면 현재 금리 기준 가격·듀레이션·볼록성·DV01 등
3) 채권 계산기: 쿠폰·만기·금리(또는 가격)를 넣으면 가격, YTM, 맥컬리/수정 듀레이션, 볼록성, DV01, 경과이자, 현금흐름표
4) 채권 ETF: 가격·분배수익률과 금리 변화에 대한 실증 듀레이션(회귀)
"""
import io
import math
import time
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TENORS = {  # FRED 시리즈: (표시 이름, 만기 연수)
    "DGS1MO": ("1개월", 1 / 12), "DGS3MO": ("3개월", 0.25), "DGS6MO": ("6개월", 0.5), "DGS1": ("1년", 1),
    "DGS2": ("2년", 2), "DGS3": ("3년", 3), "DGS5": ("5년", 5), "DGS7": ("7년", 7), "DGS10": ("10년", 10),
    "DGS20": ("20년", 20), "DGS30": ("30년", 30),
}
BOND_ETFS = {  # 티커: (이름, 비교할 국채 금리)
    "SHY": ("미 국채 1-3년", "DGS2"), "IEF": ("미 국채 7-10년", "DGS10"), "TLT": ("미 국채 20년+", "DGS20"),
    "AGG": ("미 종합채권", "DGS5"), "BND": ("미 종합채권(뱅가드)", "DGS5"), "LQD": ("미 투자등급 회사채", "DGS10"),
    "HYG": ("미 하이일드 회사채", "DGS5"), "TIP": ("미 물가연동국채", "DGS7"), "EMB": ("신흥국 달러채", "DGS10"),
}


# ======================================================================
# 채권 수학 (가격·수익률·듀레이션·볼록성)
# ======================================================================
def _schedule(T: float, freq: int):
    """만기까지 남은 T년, 연 freq회 이자 → (현금흐름 시점(년) 배열, 직전 이자일 이후 경과 비율)."""
    n = max(int(math.ceil(T * freq - 1e-9)), 1)
    first = T - (n - 1) / freq           # 첫 이자까지 남은 기간(년)
    accrued_frac = 1.0 - first * freq    # 이번 이자 기간 중 이미 지난 비율
    times = first + np.arange(n) / freq
    return times, min(max(accrued_frac, 0.0), 1.0)


def bond_metrics(coupon: float, T: float, ytm: float, freq: int = 2, face: float = 100.0) -> dict:
    """coupon·ytm은 연율(소수), 복리는 freq 기준. 가격은 액면 100 기준."""
    times, acc = _schedule(T, freq)
    cf = np.full(len(times), face * coupon / freq)
    cf[-1] += face
    k = times * freq
    disc = (1 + ytm / freq) ** (-k)
    pv = cf * disc
    dirty = float(pv.sum())
    accrued = face * coupon / freq * acc
    clean = dirty - accrued
    mac = float((times * pv).sum() / dirty)
    mod = mac / (1 + ytm / freq)
    conv = float((pv * times * (times + 1 / freq)).sum() / (dirty * (1 + ytm / freq) ** 2))
    dv01 = mod * dirty * 1e-4
    # 유효 듀레이션·볼록성 (±1bp 재평가) — 검산용
    p_up = float((cf * (1 + (ytm + 1e-4) / freq) ** (-k)).sum())
    p_dn = float((cf * (1 + (ytm - 1e-4) / freq) ** (-k)).sum())
    eff_d = (p_dn - p_up) / (2 * dirty * 1e-4)
    eff_c = (p_dn + p_up - 2 * dirty) / (dirty * 1e-8)
    return {"dirty": dirty, "clean": clean, "accrued": accrued, "mac": mac, "mod": mod, "conv": conv, "dv01": dv01,
            "eff_dur": eff_d, "eff_conv": eff_c, "times": times, "cf": cf, "pv": pv,
            "current_yield": face * coupon / clean if clean > 0 else float("nan")}


def ytm_from_price(clean: float, coupon: float, T: float, freq: int = 2, face: float = 100.0) -> float:
    """깨끗한 가격(액면 100 기준) → 만기수익률(이분법)."""
    lo, hi = -0.05, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if bond_metrics(coupon, T, mid, freq, face)["clean"] > clean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ======================================================================
# 데이터
# ======================================================================
def _fred_one(sid: str) -> pd.Series:
    """FRED 시리즈 하나 (앱의 다른 곳과 같은 방식: id만 넣은 CSV). 실패하면 두 번 더 시도."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    err = None
    for i in range(3):
        try:
            with urllib.request.urlopen(url, timeout=12) as resp:
                raw = resp.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(raw), na_values=["."])
            s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(df.iloc[:, 0]), name=sid)
            return s.dropna()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(0.5 * (i + 1))
    raise RuntimeError(f"{sid}: {err}")


def _fred_csv(ids: list, start: str) -> pd.DataFrame:
    """여러 시리즈를 동시에(병렬) 받아 start 이후만 남긴다. 받지 못한 시리즈는 빠진다."""
    import concurrent.futures as cf

    out = {}
    with cf.ThreadPoolExecutor(max_workers=min(8, len(ids))) as ex:
        futs = {ex.submit(_fred_one, i): i for i in ids}
        for f in cf.as_completed(futs):
            try:
                out[futs[f]] = f.result()
            except Exception:  # noqa: BLE001
                pass
    if not out:
        raise RuntimeError("FRED에서 데이터를 받지 못했습니다")
    df = pd.DataFrame(out).sort_index()
    return df[df.index >= pd.Timestamp(start)]


@st.cache_data(ttl=6 * 60 * 60, show_spinner="미국 국채 금리를 불러오는 중 (FRED)...")
def load_treasury(years: int = 10) -> pd.DataFrame:
    start = (datetime.now() - timedelta(days=365 * years + 30)).strftime("%Y-%m-%d")
    df = _fred_csv(list(TENORS), start)
    return df[[c for c in TENORS if c in df.columns]] / 100


@st.cache_data(ttl=6 * 60 * 60, show_spinner="채권 ETF 가격을 불러오는 중...")
def load_etf(ticker: str) -> dict:
    import yfinance as yf

    t = yf.Ticker(ticker)
    h = t.history(period="3y", auto_adjust=True)["Close"].dropna()
    h.index = pd.to_datetime(h.index).tz_localize(None)
    info = {}
    try:
        info = t.info or {}
    except Exception:  # noqa: BLE001
        pass
    return {"close": h, "info": info}


# ======================================================================
# 화면
# ======================================================================
def _fmt_pct(x, nd=2):
    return "-" if x is None or not np.isfinite(x) else f"{x * 100:.{nd}f}%"


def _grid(items, cols=4):
    from options import mgrid

    mgrid(items, cols)


def render_bonds() -> None:
    st.caption("미국 국채 금리는 FRED(미 연준 경제 데이터) 일별 수익률 곡선 기준입니다. 개별 회사채·국내 채권 시세는 무료 데이터가 없어, "
               "직접 입력하는 채권 계산기와 채권 ETF로 대신합니다.")
    t1, t2, t3, t4 = st.tabs(["국채 금리 곡선", "국채 조회 (듀레이션)", "채권 계산기", "채권 ETF"])
    try:
        ts = load_treasury()
    except Exception as e:  # noqa: BLE001
        ts = None
        st.error(f"국채 금리를 불러오지 못했습니다: {e}")
    with t1:
        if ts is not None:
            _curve(ts)
    with t2:
        if ts is not None:
            _treasury_lookup(ts)
    with t3:
        _calculator(ts)
    with t4:
        _etf(ts)


def _curve(ts: pd.DataFrame) -> None:
    last = ts.dropna(how="all").ffill().iloc[-1]
    asof = ts.dropna(how="all").index[-1]

    def at(days):
        sub = ts[ts.index <= asof - pd.Timedelta(days=days)].ffill()
        return sub.iloc[-1] if len(sub) else None

    xs = [TENORS[c][1] for c in ts.columns]
    labels = [TENORS[c][0] for c in ts.columns]
    fig = go.Figure()
    for nm, row, color, dash in (("오늘", last, "#E5534B", "solid"), ("1개월 전", at(30), "#F2A93B", "dash"),
                                 ("1년 전", at(365), "#4C9BE8", "dot")):
        if row is None:
            continue
        fig.add_trace(go.Scatter(x=xs, y=row.values * 100, name=nm, mode="lines+markers", line=dict(color=color, dash=dash),
                                 text=labels, hovertemplate="%{text} %{y:.2f}%<extra>" + nm + "</extra>"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                      xaxis=dict(title="만기(년)", type="log", tickvals=xs, ticktext=labels),
                      yaxis=dict(title="금리(%)", ticksuffix="%"), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)
    s10, s2, s3m = last.get("DGS10"), last.get("DGS2"), last.get("DGS3MO")
    _grid([("기준일", f"{asof:%Y-%m-%d}"), ("10년물", _fmt_pct(s10)), ("2년물", _fmt_pct(s2)), ("3개월물", _fmt_pct(s3m)),
           ("10년−2년", f"{(s10 - s2) * 1e4:+.0f}bp", "음수면 장단기 금리 역전"), ("10년−3개월", f"{(s10 - s3m) * 1e4:+.0f}bp")])
    spr = (ts["DGS10"] - ts["DGS2"]).dropna() * 100
    spr2 = (ts["DGS10"] - ts["DGS3MO"]).dropna() * 100
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=spr.index, y=spr.values, name="10년−2년", mode="lines", line=dict(color="#4C9BE8", width=1.4)))
    f2.add_trace(go.Scatter(x=spr2.index, y=spr2.values, name="10년−3개월", mode="lines", line=dict(color="#F2A93B", width=1.2)))
    f2.add_hline(y=0, line_color="#888", line_dash="dot")
    f2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                     yaxis=dict(title="금리차(%p)"), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(f2, use_container_width=True)


def _treasury_lookup(ts: pd.DataFrame) -> None:
    last = ts.ffill().iloc[-1]
    rows = []
    for c, (nm, T) in TENORS.items():
        y = last.get(c)
        if y is None or not np.isfinite(y):
            continue
        cpn = 0.0 if T < 1 else y            # 1년 미만은 할인채, 그 외는 액면가 채권(쿠폰 = 금리)
        m = bond_metrics(cpn, T, y, freq=1 if T < 1 else 2)
        rows.append({"만기": nm, "금리(YTM)": y * 100, "가정 쿠폰": cpn * 100, "가격": m["clean"],
                     "맥컬리 듀레이션(년)": m["mac"], "수정 듀레이션": m["mod"], "볼록성": m["conv"],
                     "DV01(액면 100)": m["dv01"], "1%p 상승 시 가격변화(%)": (-m["mod"] * 0.01 + 0.5 * m["conv"] * 1e-4) * 100})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"금리(YTM)": "{:.2f}%", "가정 쿠폰": "{:.2f}%", "가격": "{:.3f}", "맥컬리 듀레이션(년)": "{:.2f}",
                                  "수정 듀레이션": "{:.2f}", "볼록성": "{:.2f}", "DV01(액면 100)": "{:.4f}",
                                  "1%p 상승 시 가격변화(%)": "{:+.2f}%"}), use_container_width=True, hide_index=True,
                 column_config=__import__("glossary").col_help(df.columns))
    st.caption("현재 FRED 금리로 '새로 발행된 국채'(쿠폰 = 금리, 액면가 거래)를 가정해 계산했습니다. 1년 미만은 무이표 할인채입니다. "
               "실제 개별 국채는 쿠폰·잔존만기가 달라 값이 조금 다를 수 있습니다. 정확한 종목은 '채권 계산기'에 넣어 계산하세요.")
    pick = st.selectbox("자세히 볼 만기", [TENORS[c][0] for c in TENORS if c in ts.columns], index=8, key="bd_pick")
    code = next(c for c in TENORS if TENORS[c][0] == pick)
    y = float(last[code])
    T = TENORS[code][1]
    _detail(0.0 if T < 1 else y, T, y, 1 if T < 1 else 2, f"미 국채 {pick}")
    hist = ts[code].dropna() * 100
    f = go.Figure(go.Scatter(x=hist.index, y=hist.values, mode="lines", line=dict(color="#4C9BE8", width=1.4),
                             hovertemplate="%{x|%Y-%m-%d} %{y:.2f}%<extra></extra>"))
    f.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), yaxis=dict(title=f"{pick} 금리(%)"))
    st.plotly_chart(f, use_container_width=True)


def _detail(coupon: float, T: float, ytm: float, freq: int, label: str) -> None:
    m = bond_metrics(coupon, T, ytm, freq)
    _grid([("깨끗한 가격(액면 100)", f"{m['clean']:.4f}"), ("경과이자", f"{m['accrued']:.4f}"), ("더러운 가격", f"{m['dirty']:.4f}"),
           ("만기수익률", _fmt_pct(ytm, 3)), ("경상수익률", _fmt_pct(m["current_yield"], 3)),
           ("맥컬리 듀레이션", f"{m['mac']:.3f}년"), ("수정 듀레이션", f"{m['mod']:.3f}"), ("볼록성", f"{m['conv']:.3f}"),
           ("DV01 (1bp당 가격변화)", f"{m['dv01']:.4f}", "액면 100 기준. 액면 1억 원이면 × 1,000,000"),
           ("유효 듀레이션(검산)", f"{m['eff_dur']:.3f}")], cols=5)
    # 가격-금리 곡선과 근사
    ys = np.linspace(max(ytm - 0.03, -0.005), ytm + 0.03, 61)
    prices = [bond_metrics(coupon, T, y, freq)["dirty"] for y in ys]
    dy = ys - ytm
    approx1 = m["dirty"] * (1 - m["mod"] * dy)
    approx2 = m["dirty"] * (1 - m["mod"] * dy + 0.5 * m["conv"] * dy ** 2)
    f = go.Figure()
    f.add_trace(go.Scatter(x=ys * 100, y=prices, name="실제 가격", mode="lines", line=dict(color="#4C9BE8", width=2.2)))
    f.add_trace(go.Scatter(x=ys * 100, y=approx1, name="듀레이션 근사(직선)", mode="lines", line=dict(color="#F2A93B", dash="dash")))
    f.add_trace(go.Scatter(x=ys * 100, y=approx2, name="듀레이션+볼록성 근사", mode="lines", line=dict(color="#E5534B", dash="dot")))
    f.add_vline(x=ytm * 100, line_dash="dot", line_color="#888")
    f.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                    xaxis=dict(title="만기수익률(%)", ticksuffix="%"), yaxis=dict(title="가격(액면 100, 경과이자 포함)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.markdown(f"**{label}** · 가격-금리 곡선")
    st.plotly_chart(f, use_container_width=True)
    with st.expander("현금흐름표"):
        cft = pd.DataFrame({"시점(년)": m["times"], "현금흐름": m["cf"], "현재가치": m["pv"],
                            "가중치(PV/가격)": m["pv"] / m["dirty"], "시점×가중치": m["times"] * m["pv"] / m["dirty"]})
        st.dataframe(cft.round(4), use_container_width=True, hide_index=True)


def _calculator(ts) -> None:
    c1, c2, c3 = st.columns(3)
    coupon = c1.number_input("표면이자율(쿠폰, %)", value=4.0, step=0.125, format="%.3f",
                             key="bd_cpn") / 100
    T = c2.number_input("잔존만기(년)", value=10.0, min_value=0.01, step=0.25, key="bd_T")
    freq = c3.selectbox("이자 지급 횟수(연)", [1, 2, 4, 12], index=1, key="bd_f", format_func=lambda x: f"연 {x}회")
    mode = st.radio("입력 기준", ["만기수익률로 가격 계산", "가격으로 만기수익률 계산"], horizontal=True, key="bd_mode")
    if mode.startswith("만기"):
        y = st.number_input("만기수익률(%)", value=float(coupon * 100), step=0.05, format="%.3f", key="bd_y") / 100
    else:
        px = st.number_input("깨끗한 가격(액면 100 기준)", value=100.0, step=0.25, format="%.4f", key="bd_px")
        y = ytm_from_price(px, coupon, T, freq)
        st.info(f"가격 {px:.4f} → 만기수익률 {y * 100:.4f}%")
    _detail(coupon, T, y, freq, f"쿠폰 {coupon * 100:.3f}% · 잔존 {T:g}년")
    st.caption("국내 채권도 쿠폰·만기·수익률(민평 금리 등)을 넣으면 같은 방식으로 계산됩니다. 국고채는 보통 연 2회(6개월) 이자 지급입니다. "
               "날짜 관행(Act/Act 등)은 단순화해 기간 비율로 계산했습니다.")


def _etf(ts) -> None:
    pick = st.selectbox("채권 ETF", list(BOND_ETFS), key="bd_etf", format_func=lambda k: f"{k} · {BOND_ETFS[k][0]}")
    try:
        E = load_etf(pick)
    except Exception as e:  # noqa: BLE001
        st.error(f"{pick} 데이터를 불러오지 못했습니다: {e}")
        return
    h, info = E["close"], E["info"]
    ref = BOND_ETFS[pick][1]
    dur = r2 = None
    if ts is not None and ref in ts:
        dy = ts[ref].diff()
        r = np.log(h).diff()
        j = pd.concat([r, dy], axis=1, join="inner").dropna().iloc[-252:]
        if len(j) > 60:
            c = np.cov(j.iloc[:, 0], j.iloc[:, 1])
            dur = -c[0, 1] / c[1, 1]
            r2 = (c[0, 1] ** 2) / (c[0, 0] * c[1, 1])
    yld = info.get("yield") or info.get("trailingAnnualDividendYield")
    if yld is not None and yld > 1:   # 일부 응답은 % 단위
        yld = yld / 100
    _grid([("현재가", f"{h.iloc[-1]:,.2f}"), ("1년 수익률", _fmt_pct(h.iloc[-1] / h.iloc[-253] - 1) if len(h) > 253 else "-"),
           ("분배수익률", _fmt_pct(yld) if yld else "-"), ("총보수", _fmt_pct((info.get("netExpenseRatio") or 0) / 100) if info.get("netExpenseRatio") else "-"),
           (f"실증 듀레이션 ({TENORS[ref][0]} 금리 기준)", f"{dur:.2f}" if dur is not None else "-",
            "최근 1년 일별 수익률을 국채 금리 변화에 회귀한 기울기(−). 금리 1%p 상승 시 약 이만큼 % 하락"),
           ("설명력 R²", f"{r2:.2f}" if r2 is not None else "-")])
    f = go.Figure(go.Scatter(x=h.index, y=h.values, mode="lines", name=pick, line=dict(color="#4C9BE8", width=1.4)))
    if ts is not None and ref in ts:
        yy = ts[ref].dropna()
        yy = yy[yy.index >= h.index[0]] * 100
        f.add_trace(go.Scatter(x=yy.index, y=yy.values, name=f"{TENORS[ref][0]} 국채 금리(%)", mode="lines", yaxis="y2",
                               line=dict(color="#E5534B", width=1.2, dash="dot")))
    f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                    yaxis=dict(title="가격(분배금 재투자)"), yaxis2=dict(title="금리(%)", overlaying="y", side="right", showgrid=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(f, use_container_width=True)
    st.caption("ETF는 만기가 계속 바뀌는 채권 묶음이라, 운용사 공시 듀레이션 대신 시장 데이터로 추정한 '실증 듀레이션'을 보여줍니다. "
               "회사채·하이일드 ETF는 금리보다 신용 위험의 영향이 커서 R²가 낮게 나옵니다.")
