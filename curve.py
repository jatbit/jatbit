"""채권 → 금리 곡선 탭: 미국(FRED)·한국(국고채) 곡선, 제로·선도금리(NSS), 주성분(PCA), 만기별 듀레이션 표."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bondmath as bm
import ui

KR_TENORS = {"1Y": ("1년", 1.0), "3Y": ("3년", 3.0), "5Y": ("5년", 5.0), "10Y": ("10년", 10.0), "20Y": ("20년", 20.0),
             "30Y": ("30년", 30.0)}


@st.cache_data(ttl=6 * 60 * 60, show_spinner="국고채 금리를 불러오는 중...")
def load_kr_curve(years: int = 5) -> pd.DataFrame | None:
    """국고채 만기별 금리 (FinanceDataReader, 키 없이). 못 받으면 None."""
    import concurrent.futures as cf

    import FinanceDataReader as fdr

    start = (datetime.now() - timedelta(days=365 * years + 30)).strftime("%Y-%m-%d")

    def one(t):
        s = fdr.DataReader(f"KR{t}T=RR", start)["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return t, s / 100

    out = {}
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for f in cf.as_completed([ex.submit(one, t) for t in KR_TENORS]):
            try:
                t, s = f.result()
                if len(s) > 20:
                    out[t] = s
            except Exception:  # noqa: BLE001
                pass
    if len(out) < 3:
        return None
    df = pd.DataFrame(out).sort_index()
    return df[[c for c in KR_TENORS if c in df.columns]]


def us_curve(ts: pd.DataFrame, asof=None) -> bm.Curve | None:
    from bonds import TENORS

    d = ts.ffill()
    if asof is not None:
        d = d[d.index <= asof]
    if not len(d):
        return None
    row = d.iloc[-1].dropna()
    if len(row) < 4:
        return None
    return bm.Curve([TENORS[c][1] for c in row.index], row.values, f"미 국채 ({d.index[-1]:%Y-%m-%d})")


def kr_curve(kts: pd.DataFrame | None, asof=None) -> bm.Curve | None:
    if kts is None:
        return None
    d = kts.ffill()
    if asof is not None:
        d = d[d.index <= asof]
    if not len(d):
        return None
    row = d.iloc[-1].dropna()
    if len(row) < 3:
        return None
    return bm.Curve([KR_TENORS[c][1] for c in row.index], row.values, f"국고채 ({d.index[-1]:%Y-%m-%d})")


def curves(ts) -> dict:
    """{'USD': Curve, 'KRW': Curve} 중 받을 수 있는 것만."""
    out = {}
    if ts is not None:
        c = us_curve(ts)
        if c is not None:
            out["USD"] = c
    try:
        c = kr_curve(load_kr_curve())
        if c is not None:
            out["KRW"] = c
    except Exception:  # noqa: BLE001
        pass
    return out


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
def _yield_chart(ts, names: dict, title_unit="금리(%)"):
    asof = ts.dropna(how="all").index[-1]
    last = ts.dropna(how="all").ffill().iloc[-1]

    def at(days):
        sub = ts[ts.index <= asof - pd.Timedelta(days=days)].ffill()
        return sub.iloc[-1] if len(sub) else None

    xs = [names[c][1] for c in ts.columns]
    labels = [names[c][0] for c in ts.columns]
    fig = go.Figure()
    for nm, row, color, dash in (("오늘", last, "#E5534B", "solid"), ("1개월 전", at(30), "#F2A93B", "dash"),
                                 ("1년 전", at(365), "#4C9BE8", "dot")):
        if row is None:
            continue
        fig.add_trace(go.Scatter(x=xs, y=row.values * 100, name=nm, mode="lines+markers", line=dict(color=color, dash=dash),
                                 text=labels, hovertemplate="%{text} %{y:.2f}%<extra>" + nm + "</extra>"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                      xaxis=dict(title="만기(년)", type="log", tickvals=xs, ticktext=labels),
                      yaxis=dict(title=title_unit, ticksuffix="%"), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)
    return asof, last


def _zero_fwd(C: bm.Curve) -> None:
    tt = np.linspace(0.1, C.t.max(), 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=C.tenors, y=C.par * 100, name="만기수익률 (시장)", mode="markers", marker=dict(color="#888", size=8)))
    fig.add_trace(go.Scatter(x=tt, y=C.smooth_zero(tt) * 100, name="제로금리 (NSS)", line=dict(color="#4C9BE8", width=2.2)))
    fig.add_trace(go.Scatter(x=tt, y=C.fwd(tt) * 100, name="순간 선도금리", line=dict(color="#E5534B", dash="dash")))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                      xaxis=dict(title="만기(년)"), yaxis=dict(title="금리(%)", ticksuffix="%"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)
    b0, b1, b2, b3, l1, l2 = C.p
    f = lambda a, b: float(np.exp(-(C.zero(a) * a))) / float(np.exp(-(C.zero(b) * b)))  # noqa: E731
    fwd = lambda a, b: (np.log(f(a, b)) / (b - a))  # noqa: E731
    ui.cards([("장기 수준 β0", f"{b0 * 100:.2f}%"), ("단기 − 장기 β1", f"{b1 * 100:+.2f}%p"),
              ("1년 뒤 1년 금리", f"{fwd(1, 2) * 100:.2f}%", "지금 곡선이 말하는 값"),
              ("2년 뒤 1년 금리", f"{fwd(2, 3) * 100:.2f}%"),
              ("5년 뒤 5년 금리", f"{fwd(5, 10) * 100:.2f}%")], cols=5)
    st.caption(f"만기수익률(액면가 채권 금리)에서 할인계수를 차례로 풀어(부트스트랩) 제로금리를 구하고, 넬슨-시겔-스벤슨 식으로 "
               f"매끈하게 이었습니다 (적합 오차 {C.rmse * 1e4:.1f}bp). 선도금리는 '지금 곡선대로라면 그때 시작할 금리'로, "
               "기대 금리에 기간 프리미엄이 더해진 값입니다.")


def _pca(ts, names) -> None:
    yrs = st.radio("기간", [3, 5, 10], index=1, horizontal=True, key="cv_pca_y", format_func=lambda x: f"최근 {x}년")
    R = bm.curve_pca(ts, yrs)
    if R is None:
        st.info("데이터가 부족해 계산하지 못했습니다.")
        return
    L, ev, sd = R
    xs = [names[c][1] for c in L.index]
    fig = go.Figure()
    for col, color in zip(L.columns, ["#4C9BE8", "#E5534B", "#D4AF37"]):
        fig.add_trace(go.Scatter(x=xs, y=L[col], name=f"{col} ({ev[list(L.columns).index(col)] * 100:.0f}%)", mode="lines+markers",
                                 line=dict(color=color)))
    fig.add_hline(y=0, line_dash="dot", line_color="#888")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                      xaxis=dict(title="만기(년)", type="log", tickvals=xs, ticktext=[names[c][0] for c in L.index]),
                      yaxis=dict(title="움직임 비중"), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"일별 금리 변화를 주성분으로 나누면 세 가지 움직임이 대부분을 설명합니다: 수준(모든 만기가 같이 움직임, {ev[0] * 100:.0f}%), "
               f"기울기(단기와 장기가 반대로, {ev[1] * 100:.0f}%), 곡률(중기만 따로, {ev[2] * 100:.0f}%). "
               f"하루 표준편차는 각각 {sd[0]:.1f}bp · {sd[1]:.1f}bp · {sd[2]:.1f}bp입니다. '내 채권' 탭의 시나리오가 이 세 움직임을 본뜬 것입니다.")


def render_curve_tab(ts) -> None:
    from bonds import TENORS, _treasury_lookup

    mkt = st.radio("시장", ["미국 국채", "한국 국고채"], horizontal=True, key="cv_mkt", label_visibility="collapsed")
    if mkt == "미국 국채":
        if ts is None:
            st.error("미국 국채 금리를 불러오지 못했습니다.")
            return
        data, names, C = ts, TENORS, us_curve(ts)
    else:
        try:
            data = load_kr_curve()
        except Exception:  # noqa: BLE001
            data = None
        if data is None:
            st.warning("국고채 금리를 받지 못했습니다 (FinanceDataReader). 잠시 뒤 다시 시도해 주세요.")
            return
        names, C = KR_TENORS, kr_curve(data)
    asof, last = _yield_chart(data, names)
    long_ = data.columns[-1] if mkt == "한국 국고채" else "DGS10"
    two = "DGS2" if mkt == "미국 국채" else "3Y"
    if long_ in last and two in last:
        spr = (data[long_] - data[two]).dropna() * 100
        ui.cards([("기준일", f"{asof:%Y-%m-%d}"), (f"{names[two][0]}", f"{last[two] * 100:.2f}%"),
                  (f"{names[long_][0]}", f"{last[long_] * 100:.2f}%"),
                  (f"{names[long_][0]}−{names[two][0]}", f"{(last[long_] - last[two]) * 1e4:+.0f}bp", "음수면 장단기 금리 역전"),
                  ("1년 범위", f"{spr.iloc[-252:].min() * 100:+.0f} ~ {spr.iloc[-252:].max() * 100:+.0f}bp")], cols=5)
        f2 = go.Figure(go.Scatter(x=spr.index, y=spr.values, mode="lines", line=dict(color="#4C9BE8", width=1.4),
                                  name=f"{names[long_][0]}−{names[two][0]}"))
        f2.add_hline(y=0, line_color="#888", line_dash="dot")
        f2.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), yaxis=dict(title="금리차(%p)"))
        st.plotly_chart(f2, use_container_width=True)
    if C is not None:
        exp, op = ui.more("제로금리 · 선도금리 (부트스트랩 + NSS)", key=f"cv_more_zf_{mkt}")
        if op:
            with exp:
                _zero_fwd(C)
    exp, op = ui.more("금리 곡선이 움직이는 방식 (주성분: 수준·기울기·곡률)", key=f"cv_more_pca_{mkt}")
    if op:
        with exp:
            _pca(data, names)
    if mkt == "미국 국채":
        exp, op = ui.more("만기별 국채 가격·듀레이션 표", key="cv_more_tbl")
        if op:
            with exp:
                _treasury_lookup(ts)
    else:
        st.caption("국고채 금리는 FinanceDataReader(Investing.com 경유) 일별 종가입니다.")


__all__ = [n for n in dir() if not n.startswith("__")]
