"""채권 → 금리 곡선 탭: 미국(FRED)·한국(국고채) 곡선, 제로·선도금리(NSS), 주성분(PCA), 만기별 듀레이션 표."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bondmath as bm
import ui

KR_TENORS = {"1Y": ("1년", 1.0), "2Y": ("2년", 2.0), "3Y": ("3년", 3.0), "5Y": ("5년", 5.0), "10Y": ("10년", 10.0),
             "20Y": ("20년", 20.0), "30Y": ("30년", 30.0)}
ECOS_KR = {"1Y": "010190000", "2Y": "010195000", "3Y": "010200000", "5Y": "010200001", "10Y": "010210000",
           "20Y": "010220000", "30Y": "010230000"}  # 한국은행 ECOS 817Y002 시장금리(일별)


def _ecos(code: str, s: str, e: str) -> list:
    """ECOS 공개 sample 키 (한 번에 10건) — 키 없이 쓸 수 있는 범위만."""
    import json
    import time
    import urllib.request

    url = f"https://ecos.bok.or.kr/api/StatisticSearch/sample/json/kr/1/10/817Y002/D/{s}/{e}/{code}"
    for i in range(3):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                rows = json.loads(resp.read().decode("utf-8")).get("StatisticSearch", {}).get("row", [])
            return [(r["TIME"], float(r["DATA_VALUE"])) for r in rows if r.get("DATA_VALUE")]
        except Exception:  # noqa: BLE001
            time.sleep(0.4 * (i + 1))
    return []


def _windows(start, end, days=13):
    out, cur = [], start
    while cur <= end:
        w = min(cur + timedelta(days=days), end)
        out.append((cur.strftime("%Y%m%d"), w.strftime("%Y%m%d")))
        cur = w + timedelta(days=1)
    return out


def _fdr_kr(start: str) -> pd.DataFrame | None:
    import concurrent.futures as cf

    import FinanceDataReader as fdr

    def one(t):
        s = fdr.DataReader(f"KR{t}T=RR", start)["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return t, s / 100

    try:  # 하나 먼저 시험해 보고 안 되면 바로 포기
        t0, s0 = one("10Y")
        if len(s0) < 20:
            return None
    except Exception:  # noqa: BLE001
        return None
    out = {t0: s0}
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for f in cf.as_completed([ex.submit(one, t) for t in KR_TENORS if t != "10Y"]):
            try:
                t, s = f.result()
                if len(s) > 20:
                    out[t] = s
            except Exception:  # noqa: BLE001
                pass
    return pd.DataFrame(out).sort_index() if len(out) >= 3 else None


@st.cache_data(ttl=6 * 60 * 60, show_spinner="국고채 금리를 불러오는 중 (한국은행)...")
def load_kr_curve(months: int = 6) -> pd.DataFrame | None:
    """국고채 만기별 일별 금리 (소수). FinanceDataReader가 되면 5년치, 아니면 한국은행 ECOS 최근 months개월. 못 받으면 None."""
    import concurrent.futures as cf

    df = _fdr_kr((datetime.now() - timedelta(days=365 * 5 + 30)).strftime("%Y-%m-%d"))
    if df is None:
        end = datetime.now().date()
        wins = _windows(end - timedelta(days=int(months * 30.5)), end)
        jobs = [(t, w) for t in KR_TENORS for w in wins]
        data = {t: {} for t in KR_TENORS}
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            for (t, _), rows in zip(jobs, ex.map(lambda j: _ecos(ECOS_KR[j[0]], *j[1]), jobs)):
                for d, v in rows:
                    data[t][pd.Timestamp(d)] = v / 100
        cols = {t: pd.Series(v).sort_index() for t, v in data.items() if len(v) > 20}
        if len(cols) < 3:
            return None
        df = pd.DataFrame(cols).sort_index()
    return df[[c for c in KR_TENORS if c in df.columns]]


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_kr_year_ago() -> pd.Series | None:
    """1년 전 국고채 곡선 (ECOS, 1년 전 2주 구간의 마지막 값)."""
    end = datetime.now().date() - timedelta(days=365)
    s, e = (end - timedelta(days=13)).strftime("%Y%m%d"), end.strftime("%Y%m%d")
    row = {}
    for t, code in ECOS_KR.items():
        r = _ecos(code, s, e)
        if r:
            row[t] = r[-1][1] / 100
    return pd.Series(row) if len(row) >= 3 else None


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
def _yield_chart(ts, names: dict, title_unit="금리(%)", year_ago=None):
    asof = ts.dropna(how="all").index[-1]
    last = ts.dropna(how="all").ffill().iloc[-1]

    def at(days):
        if days >= 365 and year_ago is not None and (ts.index[0] > asof - pd.Timedelta(days=days)):
            return year_ago.reindex(ts.columns)
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
    f = lambda a, b: float(np.exp(-(C.zero(a) * a))) / float(np.exp(-(C.zero(b) * b)))  # noqa: E731
    fwd = lambda a, b: (np.log(f(a, b)) / (b - a))  # noqa: E731
    ui.cards([("1년 뒤 1년 금리", f"{fwd(1, 2) * 100:.2f}%", "지금 곡선이 말하는 값"),
              ("2년 뒤 1년 금리", f"{fwd(2, 3) * 100:.2f}%"),
              ("5년 뒤 5년 금리", f"{fwd(5, 10) * 100:.2f}%")], cols=3)
    st.caption(f"만기수익률(액면가 채권 금리)에서 할인계수를 차례로 풀어(부트스트랩) 제로금리를 구하고, 넬슨-시겔-스벤슨 식으로 "
               f"매끈하게 이었습니다 (적합 오차 {C.rmse * 1e4:.1f}bp). 선도금리는 '지금 곡선대로라면 그때 시작할 금리'로, "
               "기대 금리에 기간 프리미엄이 더해진 값입니다.")


def _pca(ts, names) -> None:
    span = (ts.index[-1] - ts.index[0]).days / 365
    if span > 2.5:
        yrs = st.radio("기간", [y for y in (3, 5, 10) if y <= span + 0.5] or [3], horizontal=True, key="cv_pca_y",
                       format_func=lambda x: f"최근 {x}년")
    else:
        yrs = span
        st.caption(f"최근 {span * 12:.0f}개월 일별 금리 변화 기준")
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
    ya = None
    if mkt == "한국 국고채":
        try:
            ya = load_kr_year_ago()
        except Exception:  # noqa: BLE001
            ya = None
    asof, last = _yield_chart(data, names, year_ago=ya)
    long_ = "10Y" if mkt == "한국 국고채" else "DGS10"
    two = "DGS2" if mkt == "미국 국채" else "3Y"
    if long_ in last and two in last:
        spr = (data[long_] - data[two]).dropna() * 100
        months = (spr.index[-1] - spr.index[0]).days / 30.4
        rng_lab = "최근 1년 범위" if months >= 12 else f"최근 {months:.0f}개월 범위"
        ui.cards([("기준일", f"{asof:%Y-%m-%d}"), (f"{names[two][0]}", f"{last[two] * 100:.2f}%"),
                  (f"{names[long_][0]}", f"{last[long_] * 100:.2f}%"),
                  (f"{names[long_][0]}−{names[two][0]}", f"{(last[long_] - last[two]) * 1e4:+.0f}bp", "음수면 장단기 금리 역전"),
                  (rng_lab, f"{spr.iloc[-252:].min() * 100:+.0f} ~ {spr.iloc[-252:].max() * 100:+.0f}bp")], cols=5)
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
        span = (data.index[-1] - data.index[0]).days
        st.caption("국고채 금리는 FinanceDataReader 일별 종가입니다." if span > 400 else
                   "국고채 금리는 한국은행 ECOS 시장금리(일별)입니다. 키 없이 받을 수 있는 범위라 최근 6개월치이며, 1년 전 곡선만 따로 받았습니다.")


__all__ = [n for n in dir() if not n.startswith("__")]
