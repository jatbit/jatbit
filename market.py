"""시장 개요 (첫 화면): 주요 지수·환율·금리·원자재·가상자산 시세, 국내·미국 시장 상승·하락 종목과 시장 지도, 시가총액·거래대금 상위 종목.

증권사 MTS/WTS 첫 화면처럼 자주 보는 숫자를 한눈에 보여주고, 기간별 수익률 같은 세부는 '더보기'로 접어 둔다.
"""
from __future__ import annotations

import html
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st

import ui

# (Yahoo 심볼, 표시 이름, 단위·형식)
GROUPS = [
    ("국내 · 미국 지수", [("^KS11", "코스피", "idx"), ("^KQ11", "코스닥", "idx"), ("^GSPC", "S&P 500", "idx"),
                       ("^IXIC", "나스닥 종합", "idx"), ("^DJI", "다우존스", "idx"), ("^SOX", "필라델피아 반도체", "idx")]),
    ("환율 · 금리 · 변동성", [("KRW=X", "원/달러 환율", "fx"), ("DX-Y.NYB", "달러인덱스", "idx"), ("^IRX", "미국 국채 3개월", "rate"),
                         ("^TNX", "미국 국채 10년", "rate"), ("^VIX", "VIX (공포지수)", "idx"), ("JPYKRW=X", "원/엔 (100엔)", "jpy")]),
    ("원자재 · 가상자산", [("CL=F", "WTI 원유", "usd"), ("GC=F", "금", "usd"), ("HG=F", "구리", "usd2"),
                       ("NG=F", "천연가스", "usd2"), ("BTC-USD", "비트코인", "usd0"), ("ETH-USD", "이더리움", "usd0")]),
]
RATE_KINDS = {"^IRX", "^TNX"}


# 세계 지도에 올릴 나라: ISO-3 코드 → (이름, 대표지수 심볼, 지수 이름). 나라를 늘리려면 여기에 추가.
COUNTRIES = {"KOR": ("한국", "^KS11", "코스피"), "USA": ("미국", "^GSPC", "S&P 500")}
PERIODS = {"1일": None, "1주": 7, "1개월": 30, "3개월": 91, "연초 이후": "ytd", "1년": 365}


def load_quotes() -> pd.DataFrame:
    """Yahoo 일별 종가 (최근 2년). 열 = 심볼. 5분이 지나면 직전 값을 먼저 쓰고 뒤에서 새로 받는다."""
    from data_load import swr

    return swr("market_quotes", _quotes_raw).copy()


def _quotes_raw() -> pd.DataFrame:
    import yfinance as yf

    syms = [s for _, items in GROUPS for s, _, _ in items]
    syms += [v[1] for v in COUNTRIES.values() if v[1] not in syms]
    d = yf.download(syms, period="2y", interval="1d", progress=False, auto_adjust=False, threads=True)
    close = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None) if getattr(close.index, "tz", None) else pd.to_datetime(close.index)
    close = close.sort_index()
    return _krx_index_fix(close)


def _krx_index_fix(close: pd.DataFrame) -> pd.DataFrame:
    """Yahoo의 코스피·코스닥 일봉은 하루 이틀 늦게 올라오는 때가 있어, KRX 일봉(FinanceDataReader)으로 최근 구간을 덮어쓴다."""
    try:
        import FinanceDataReader as fdr

        start = (pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        for sym, code in (("^KS11", "KS11"), ("^KQ11", "KQ11")):
            if sym not in close.columns:
                continue
            k = fdr.DataReader(code, start)["Close"].dropna()
            k.index = pd.to_datetime(k.index).tz_localize(None) if getattr(k.index, "tz", None) else pd.to_datetime(k.index)
            if k.empty:
                continue
            close = close.reindex(close.index.union(k.index)).sort_index()
            close.loc[k.index, sym] = k.values
    except Exception:  # noqa: BLE001  KRX 쪽이 안 되면 야후 값을 그대로 쓴다
        pass
    return close


def load_krx_snapshot() -> pd.DataFrame:  # 캐시는 krx_listing_raw (뒤에서 새로 고침)가 맡음
    """KRX 전 종목 당일 시세 (현재가·등락률·거래대금·시가총액)."""
    from data_load import krx_listing_raw

    df = krx_listing_raw().copy()
    if "Symbol" in df.columns and "Code" not in df.columns:
        df = df.rename(columns={"Symbol": "Code"})
    if "ChagesRatio" in df.columns and "ChangesRatio" not in df.columns:
        df = df.rename(columns={"ChagesRatio": "ChangesRatio"})
    keep = [c for c in ["Code", "Name", "Market", "Close", "Changes", "ChangesRatio", "Volume", "Amount", "Marcap"] if c in df.columns]
    df = df[keep].copy()
    for c in ["Close", "Changes", "ChangesRatio", "Volume", "Amount", "Marcap"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _fmt(v: float, kind: str) -> str:
    if v is None or v != v:
        return "-"
    if kind == "rate":
        return f"{v:.3f}%"
    if kind in ("fx", "jpy"):
        return f"{v:,.2f}"
    if kind == "usd0":
        return f"${v:,.0f}"
    if kind == "usd2":
        return f"${v:,.3f}"
    if kind == "usd":
        return f"${v:,.2f}"
    return f"{v:,.2f}"


def _norm(s: pd.Series, sym: str, kind: str) -> pd.Series:
    s = s.dropna()
    if sym in ("^TNX", "^IRX") and len(s) and float(s.iloc[-1]) > 20:  # 예전 표기(×10) 대비
        s = s / 10
    if kind == "jpy":
        s = s * 100
    return s


def _series(close: pd.DataFrame, kr3y, sym: str, kind: str) -> pd.Series:
    if close is None or sym not in close.columns:
        return pd.Series(dtype=float)
    return _norm(close[sym], sym, kind)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_long(sym: str) -> pd.Series:
    """한 지표의 일봉 종가 최근 5년 (세부 차트의 '5년'용)."""
    import yfinance as yf

    d = yf.download(sym, period="5y", interval="1d", progress=False, auto_adjust=False)
    if d is None or d.empty:
        return pd.Series(dtype=float)
    c = d["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c.index = pd.to_datetime(c.index).tz_localize(None) if getattr(c.index, "tz", None) else pd.to_datetime(c.index)
    return c.sort_index()


DETAIL_PERIODS = {"1개월": 30, "3개월": 91, "6개월": 182, "1년": 365, "2년": 730, "5년": 1826}


def _detail(sym: str, name: str, kind: str, close: pd.DataFrame, key: str) -> None:
    """지수·환율·금리·원자재 하나의 세부 차트: 기간 고르기 · 현재·기간 등락·최고·최저 카드 · 선 차트."""
    import plotly.graph_objects as go

    per = ui.seg("기간", list(DETAIL_PERIODS), default="1년", key=f"{key}_per") or "1년"
    if per == "5년":
        with st.spinner(f"{name} 5년 시세를 불러오는 중..."):
            try:
                s = _norm(load_long(sym), sym, kind)
            except Exception:  # noqa: BLE001
                s = pd.Series(dtype=float)
    else:
        s = _series(close, None, sym, kind)
    if len(s) < 2:
        st.caption(f"{name} 시세를 불러오지 못했습니다.")
        return
    s = s[s.index >= s.index[-1] - pd.Timedelta(days=DETAIL_PERIODS[per])]
    last, first = float(s.iloc[-1]), float(s.iloc[0])
    hi_d, lo_d = s.idxmax(), s.idxmin()
    if kind == "rate":
        chg = (last - first) * 100
        chg_txt, sign = f"{chg:+.0f}bp", chg
    else:
        chg = (last / first - 1) * 100 if first else np.nan
        chg_txt, sign = (f"{chg:+.2f}%" if chg == chg else "-"), chg
    items = [(f"{name} 현재", _fmt(last, kind), f"{s.index[-1]:%Y-%m-%d} 기준"),
             (f"{per} 등락", chg_txt, f"{s.index[0]:%Y-%m-%d} {_fmt(first, kind)} 대비", sign),
             (f"{per} 최고", _fmt(float(s.max()), kind), f"{hi_d:%Y-%m-%d}"),
             (f"{per} 최저", _fmt(float(s.min()), kind), f"{lo_d:%Y-%m-%d}")]
    if kind != "rate" and len(s) > 20:
        vol = float(np.log(s).diff().std() * np.sqrt(252) * 100)
        items.append((f"{per} 변동성 (연)", f"{vol:.1f}%", "일간 로그수익률 기준"))
    ui.cards(items, cols=len(items))
    color = ui.UP if sign == sign and sign > 0 else (ui.DOWN if sign == sign and sign < 0 else ui.FLAT)
    unit = "%" if kind == "rate" else ""
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=color, width=1.8), name=name,
                               hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.3f}" + unit + "<extra></extra>"))
    fig.add_hline(y=first, line=dict(color="rgba(140,135,124,.6)", width=1, dash="dot"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      yaxis=dict(ticksuffix=unit, tickformat=",.2f" if kind != "usd0" else ",.0f"))
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_fig")
    st.caption(f"점선 = {per} 시작 값 · 출처: Yahoo Finance 일봉 종가 ({sym})")


def _tiles(rows: list) -> None:
    cells = ""
    for name, s, kind in rows:
        if len(s) < 2:
            cells += f'<div class="mt-c"><div class="mt-n">{html.escape(name)}</div><div class="mt-v">-</div></div>'
            continue
        last, prev = float(s.iloc[-1]), float(s.iloc[-2])
        if kind == "rate":
            diff = (last - prev) * 100
            cls = "up" if diff > 0 else ("down" if diff < 0 else "flat")
            d = f'{"▲" if diff > 0 else "▼" if diff < 0 else ""} {abs(diff):.1f}bp'
        else:
            pct = (last / prev - 1) * 100 if prev else 0.0
            cls = "up" if pct > 0 else ("down" if pct < 0 else "flat")
            d = f'{"▲" if pct > 0 else "▼" if pct < 0 else ""} {abs(last - prev):,.2f} ({pct:+.2f}%)'
        color = {"up": ui.UP, "down": ui.DOWN}.get(cls, ui.FLAT)
        spark = ui.sparkline(s.iloc[-66:].values, color)
        cells += (f'<div class="mt-c" title="기준 {s.index[-1]:%Y-%m-%d}"><div class="mt-n">{html.escape(name)}</div>'
                  f'<div class="mt-v">{_fmt(last, kind)}</div><div class="mt-d {cls}">{d}</div>{spark}</div>')
    st.markdown(f'<div class="mt" style="grid-template-columns:repeat(6,minmax(0,1fr))">{cells}</div>'.replace("$", "&#36;"),
                unsafe_allow_html=True)


def _ret(s: pd.Series, days: int | None = None, ytd: bool = False):
    if len(s) < 2:
        return np.nan
    last = float(s.iloc[-1])
    if ytd:
        base = s[s.index < pd.Timestamp(s.index[-1].year, 1, 1)]
        if base.empty:
            return np.nan
        b = float(base.iloc[-1])
    else:
        t = s.index[-1] - pd.Timedelta(days=days)
        base = s[s.index <= t]
        if base.empty:
            return np.nan
        b = float(base.iloc[-1])
    return (last / b - 1) * 100 if b else np.nan


def _movers_table(df: pd.DataFrame, title: str) -> str:
    rows = ""
    for r in df.itertuples():
        pct = float(r.ChangesRatio) if r.ChangesRatio == r.ChangesRatio else 0.0
        cls = "up" if pct > 0 else ("down" if pct < 0 else "flat")
        amt = f"{r.Amount / 1e8:,.0f}억" if getattr(r, "Amount", None) == getattr(r, "Amount", None) else "-"
        rows += (f'<tr><td style="padding:4px 6px;max-width:9.5rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{html.escape(str(r.Name))}</td>'
                 f'<td style="padding:4px 6px;text-align:right">{r.Close:,.0f}</td>'
                 f'<td class="{cls}" style="padding:4px 6px;text-align:right;font-weight:600">{pct:+.2f}%</td>'
                 f'<td style="padding:4px 6px;text-align:right;opacity:.7">{amt}</td></tr>')
    return (f'<div class="sec-t" style="margin-top:4px">{html.escape(title)}</div>'
            '<table style="border-collapse:collapse;width:100%;font-size:0.8rem;font-variant-numeric:tabular-nums">'
            '<thead><tr style="opacity:.6;border-bottom:1px solid rgba(128,128,128,.3)"><th style="text-align:left;padding:3px 6px;font-weight:500">종목</th>'
            '<th style="text-align:right;padding:3px 6px;font-weight:500">현재가</th><th style="text-align:right;padding:3px 6px;font-weight:500">등락률</th>'
            f'<th style="text-align:right;padding:3px 6px;font-weight:500">거래대금</th></tr></thead><tbody>{rows}</tbody></table>')


def _sector_map() -> pd.DataFrame:
    """종목코드 → 섹터·업종 (WICS, 실패하면 KRX 업종을 WICS 이름으로 변환). 스크리너도 함께 씀."""
    import sectors

    return sectors.kr_sector_map()[0]


# 시장 지도 색: 0% 회색 → ±30%까지 (작은 등락도 보이게 곡선으로) 진해지고, ±30% 이상은 가장 진한 빨강·파랑으로 고정 (일일 등락 제한폭 기준)
HEAT_LIM = 30.0
HEAT_SCALE = [[0, "#0033CC"], [0.5, "#41454F"], [1, "#D4000F"]]


def _heat_pos(v: float) -> float:
    """등락률 → 색 위치(0~1). 제곱근 곡선: 작은 등락도 상승·하락 색이 구분되고, 30%에서 가장 진해진다."""
    return (min(abs(v), HEAT_LIM) / HEAT_LIM) ** 0.5


def _treemap(m: pd.DataFrame, title: str, sub: str, money_fmt, height: int = 560) -> None:
    """Finviz식 시장 지도: 섹터 → 업종 → 종목. 칸 크기 = 시가총액, 색 = 오늘 등락률 (상승 빨강, 하락 파랑).
    m: Key(고유), Name, Sector, Industry, Marcap, ChangesRatio, Close."""
    import plotly.graph_objects as go

    m = m.dropna(subset=["Marcap", "ChangesRatio"])
    m = m[m["Marcap"] > 0].copy()
    if m.empty:
        st.info("시장 지도를 그릴 시세가 없습니다.")
        return
    m["Sector"] = m["Sector"].fillna("분류 없음").astype(str)
    m["Industry"] = m["Industry"].fillna(m["Sector"]).astype(str)
    m["_w"] = m["ChangesRatio"] * m["Marcap"]
    gs = m.groupby("Sector").agg(Marcap=("Marcap", "sum"), W=("_w", "sum"))
    gi = m.groupby(["Sector", "Industry"]).agg(Marcap=("Marcap", "sum"), W=("_w", "sum"))
    gs["Chg"], gi["Chg"] = gs["W"] / gs["Marcap"], gi["W"] / gi["Marcap"]
    tot = float(m["Marcap"].sum())
    ids = ["root"] + [f"s:{k}" for k in gs.index] + [f"i:{a}|{b}" for a, b in gi.index] + [f"c:{c}" for c in m["Key"]]
    labels = ["전체"] + list(gs.index) + [b for _a, b in gi.index] + list(m["Name"].astype(str))
    parents = ([""] + ["root"] * len(gs) + [f"s:{a}" for a, _b in gi.index]
               + [f"i:{a}|{b}" for a, b in zip(m["Sector"], m["Industry"])])
    values = [tot] + list(gs["Marcap"]) + list(gi["Marcap"]) + list(m["Marcap"])
    chg = [float(np.average(m["ChangesRatio"], weights=m["Marcap"]))] + list(gs["Chg"]) + list(gi["Chg"]) + list(m["ChangesRatio"])
    full = m["FullName"].astype(str) + " · " if "FullName" in m.columns else ""
    px_txt = [""] * (1 + len(gs) + len(gi)) + list(full + pd.Series([f"현재가 {money_fmt(x, 'px')}" for x in m["Close"]], index=m.index))
    cap_txt = [money_fmt(v, "cap") for v in values]
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
        customdata=[[float(c), t, p] for c, t, p in zip(chg, cap_txt, px_txt)],
        texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%",
        hovertemplate="<b>%{label}</b><br>등락률 %{customdata[0]:+.2f}%<br>시가총액 %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        marker=dict(colors=[float(np.sign(x)) * _heat_pos(float(x)) for x in chg], cmin=-1, cmax=1, showscale=False,
                    colorscale=HEAT_SCALE,
                    line=dict(width=1, color="rgba(0,0,0,0.35)")),
        tiling=dict(pad=1), pathbar=dict(visible=True), maxdepth=4, textfont=dict(color="white"),
    ))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=24, b=0))
    st.plotly_chart(fig, use_container_width=True)
    if sub:
        st.caption(sub)


def _krw(v, kind):
    if v is None or v != v:
        return "-"
    return f"{v:,.0f}원" if kind == "px" else (f"{v / 1e12:,.1f}조원" if v >= 1e12 else f"{v / 1e8:,.0f}억원")


def _usd(v, kind):
    if v is None or v != v:
        return "-"
    if kind == "px":
        return f"${v:,.2f}"
    return f"${v / 1e12:,.2f}T" if v >= 1e12 else f"${v / 1e9:,.1f}B"


def _map_controls(prefix: str, markets: list) -> str:
    """시장 선택 (종목은 항상 전체)."""
    return ui.seg("시장", markets, default=markets[0], key=f"{prefix}_mk") or markets[0]


@ui.fragment
def _heatmap(snap: pd.DataFrame) -> None:
    need = {"Code", "Name", "Marcap", "ChangesRatio"}
    if not need.issubset(snap.columns):
        return
    ui.section("국내 시장 지도", "섹터 → 업종 → 종목 · 칸 크기 = 시가총액 · 색 = 오늘 등락률 · 칸을 누르면 확대")
    mk = _map_controls("mk_map", ["전체", "KOSPI", "KOSDAQ"])
    m = snap.dropna(subset=["Marcap", "ChangesRatio"]).copy()
    if mk != "전체" and "Market" in m.columns:
        m = m[m["Market"].astype(str).str.upper().str.startswith(mk)]
    m["Code"] = m["Code"].astype(str).str.zfill(6)
    m = m[m["Code"].str.endswith("0")]  # 우선주 제외
    if m.empty:
        return
    st.caption(f"{mk if mk != '전체' else '코스피·코스닥'} 전체 {len(m):,}개 종목 (우선주 제외)")
    import sectors

    try:
        smap, src = sectors.kr_sector_map()
        m = m.merge(smap, on="Code", how="left")
    except Exception as e:  # noqa: BLE001
        m["Sector"], m["Industry"], src = "분류 없음", "분류 없음", f"업종 분류를 받지 못함 ({e})"
    m["Key"] = m["Code"]
    _treemap(m, "국내 시장 지도", f"분류: {src}", _krw, height=760)


def _us_movers_table(df: pd.DataFrame, title: str) -> str:
    rows = ""
    for r in df.itertuples():
        pct = float(r.ChangesRatio) if r.ChangesRatio == r.ChangesRatio else 0.0
        cls = "up" if pct > 0 else ("down" if pct < 0 else "flat")
        amt = f"${r.Amount / 1e9:,.1f}B" if r.Amount == r.Amount else "-"
        rows += (f'<tr><td style="padding:4px 6px;max-width:11rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
                 f'<b>{html.escape(str(r.Symbol))}</b> <span style="opacity:.65">{html.escape(str(r.Name))}</span></td>'
                 f'<td style="padding:4px 6px;text-align:right">{r.Close:,.2f}</td>'
                 f'<td class="{cls}" style="padding:4px 6px;text-align:right;font-weight:600">{pct:+.2f}%</td>'
                 f'<td style="padding:4px 6px;text-align:right;opacity:.7">{amt}</td></tr>')
    return (f'<div class="sec-t" style="margin-top:4px">{html.escape(title)}</div>'
            '<table style="border-collapse:collapse;width:100%;font-size:0.8rem;font-variant-numeric:tabular-nums">'
            '<thead><tr style="opacity:.6;border-bottom:1px solid rgba(128,128,128,.3)"><th style="text-align:left;padding:3px 6px;font-weight:500">종목</th>'
            '<th style="text-align:right;padding:3px 6px;font-weight:500">현재가($)</th><th style="text-align:right;padding:3px 6px;font-weight:500">등락률</th>'
            f'<th style="text-align:right;padding:3px 6px;font-weight:500">거래대금</th></tr></thead><tbody>{rows}</tbody></table>').replace("$", "&#36;")


@ui.fragment
def _us_market() -> None:
    """미국 주식시장: 국내와 같은 구성 (상승·하락 종목 수, 거래대금, 시장 지도, 시가총액·거래대금 상위, 상승·하락률 상위)."""
    import sectors

    try:
        with st.spinner("미국 시세를 불러오는 중..."):
            sp, allq, src = sectors.us_snapshot()
    except Exception as e:  # noqa: BLE001
        st.caption(f"미국 종목 시세를 불러오지 못했습니다: {e}")
        return
    sp = sp.dropna(subset=["ChangesRatio"])
    if sp.empty:
        st.caption("미국 종목 시세가 비어 있습니다.")
        return
    ui.section("미국 주식시장", "S&P 500 구성 종목 (장중엔 15분 지연 · 정규장 기준)")
    items = []
    up, dn = int((sp["ChangesRatio"] > 0).sum()), int((sp["ChangesRatio"] < 0).sum())
    items += [("S&P 500 상승 / 하락", ui.updown(up, dn), f"보합 {int((sp['ChangesRatio'] == 0).sum()):,}"),
              ("S&P 500 거래대금", f"${sp['Amount'].sum() / 1e9:,.0f}B" if sp["Amount"].notna().any() else "-")]
    if allq is not None and len(allq):
        a = allq.dropna(subset=["ChangesRatio"])
        a = a[(a["Marcap"] > 0) | a["Marcap"].isna()]
        u2, d2 = int((a["ChangesRatio"] > 0).sum()), int((a["ChangesRatio"] < 0).sum())
        items += [("미국 상장 전체 상승 / 하락", ui.updown(u2, d2), "나스닥·NYSE·AMEX"),
                  ("미국 상장 전체 거래대금", f"${a['Amount'].sum() / 1e9:,.0f}B")]
    ui.cards(items, cols=4)

    ui.section("미국 시장 지도 (S&P 500)", "섹터 → 업종(GICS) → 종목 · 칸 크기 = 시가총액 · 색 = 오늘 등락률 · 칸을 누르면 확대")
    m = sp.dropna(subset=["Marcap"]).copy()
    st.caption(f"S&P 500 전체 {len(m):,}개 종목")
    m["Key"], m["Sector"] = m["Symbol"], m["SectorKo"]
    m["FullName"], m["Name"] = m["Name"], m["Symbol"]
    _treemap(m, "미국 시장 지도", f"분류: GICS 섹터·세부 업종 · 시세: {src}", _usd, height=760)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_us_movers_table(sp.nlargest(10, "Marcap"), "시가총액 상위"), unsafe_allow_html=True)
    with c2:
        st.markdown(_us_movers_table(sp.nlargest(10, "Amount"), "거래대금 상위"), unsafe_allow_html=True)
    exp, op = ui.more("상승률 · 하락률 상위 더보기 (S&P 500)", key="us_more_movers")
    if op:
        with exp:
            c3, c4 = st.columns(2)
            c3.markdown(_us_movers_table(sp.nlargest(10, "ChangesRatio"), "상승률 상위"), unsafe_allow_html=True)
            c4.markdown(_us_movers_table(sp.nsmallest(10, "ChangesRatio"), "하락률 상위"), unsafe_allow_html=True)
            g = sp.assign(_w=sp["ChangesRatio"] * sp["Marcap"]).groupby("SectorKo").agg(M=("Marcap", "sum"), W=("_w", "sum"), N=("Symbol", "count"))
            g["등락률(시총가중)"] = (g["W"] / g["M"]).map(lambda x: f"{x:+.2f}%")
            g["시가총액"] = g["M"].map(lambda v: _usd(v, "cap"))
            st.dataframe(g.sort_values("M", ascending=False)[["등락률(시총가중)", "시가총액", "N"]].rename(columns={"N": "종목 수"})
                         .rename_axis("섹터"), use_container_width=True)


def _kr_market(snap) -> None:
    """국내 주식시장: 상승·하락 종목 수, 거래대금, 시장 지도, 시가총액·거래대금 상위, 상승·하락률 상위."""
    if snap is None or snap.empty or "ChangesRatio" not in snap.columns:
        st.caption("국내 종목 시세를 불러오지 못했습니다.")
        return
    ui.section("국내 주식시장", "KRX 전 종목 (장중엔 지연 시세)")
    items = []
    for mk in ("KOSPI", "KOSDAQ"):
        m = snap[snap["Market"].astype(str).str.upper().str.startswith(mk)] if "Market" in snap.columns else snap
        up, dn = int((m["ChangesRatio"] > 0).sum()), int((m["ChangesRatio"] < 0).sum())
        flat = int((m["ChangesRatio"] == 0).sum())
        amt = float(m["Amount"].sum()) if "Amount" in m.columns else np.nan
        items += [(f"{mk} 상승 / 하락", ui.updown(up, dn), f"보합 {flat:,}"),
                  (f"{mk} 거래대금", f"{amt / 1e12:,.1f}조원" if amt == amt else "-")]
    ui.cards(items, cols=4)

    _heatmap(snap)
    liquid = snap[snap["Amount"] >= 1e10] if "Amount" in snap.columns else snap  # 거래대금 100억 이상만 (급등락 잡주 제외)
    c1, c2 = st.columns(2)
    with c1:
        if "Marcap" in snap.columns:
            st.markdown(_movers_table(snap.nlargest(10, "Marcap"), "시가총액 상위"), unsafe_allow_html=True)
    with c2:
        if "Amount" in snap.columns:
            st.markdown(_movers_table(snap.nlargest(10, "Amount"), "거래대금 상위"), unsafe_allow_html=True)
    exp, op = ui.more("상승률 · 하락률 상위 더보기 (거래대금 100억 이상)", key="mk_more_movers")
    if op:
        with exp:
            c3, c4 = st.columns(2)
            c3.markdown(_movers_table(liquid.nlargest(10, "ChangesRatio"), "상승률 상위"), unsafe_allow_html=True)
            c4.markdown(_movers_table(liquid.nsmallest(10, "ChangesRatio"), "하락률 상위"), unsafe_allow_html=True)


def _period_ret(s: pd.Series, per: str) -> float:
    if len(s) < 2:
        return np.nan
    v = PERIODS[per]
    if v is None:
        return (float(s.iloc[-1]) / float(s.iloc[-2]) - 1) * 100
    return _ret(s, ytd=True) if v == "ytd" else _ret(s, v)


def _world_map(close) -> str:
    """세계 지도 (나라별 대표지수 수익률로 색칠). 고른 나라 이름을 돌려준다."""
    import plotly.graph_objects as go

    ss = st.session_state
    names = [v[0] for v in COUNTRIES.values()]
    ss.setdefault("mk_country", names[0])
    ui.section("세계 증시", "나라별 대표지수 수익률 · 나라를 누르면 아래에 그 나라 지도와 시장 지도")
    per = ui.seg("기간", list(PERIODS), default="1일", key="mk_world_per") or "1일"
    iso, rets, txt, hov = [], [], [], []
    for code, (nm, sym, inm) in COUNTRIES.items():
        s = close[sym].dropna() if close is not None and sym in close.columns else pd.Series(dtype=float)
        r = _period_ret(s, per)
        iso.append(code)
        rets.append(r)
        lv = f"{float(s.iloc[-1]):,.2f}" if len(s) else "-"
        txt.append(f"<b>{inm}</b><br>{r:+.2f}%" if r == r else f"<b>{inm}</b><br>-")
        hov.append(f"{nm} · {inm} {lv}<br>{per} {r:+.2f}%" if r == r else f"{nm} · {inm}")
    vals = np.array([x if x == x else 0.0 for x in rets])
    m = max(float(np.nanmax(np.abs(vals))) if len(vals) else 1.0, 0.5)
    from geomap import UPDOWN as scale  # 나라 지도와 같은 진한 색

    fig = go.Figure()
    fig.add_trace(go.Choropleth(locations=iso, z=vals, locationmode="ISO-3", colorscale=scale, zmin=-m, zmax=m,
                                marker_line_color="rgba(255,255,255,.6)", marker_line_width=0.6, text=hov,
                                hovertemplate="%{text}<extra></extra>", showscale=False))
    fig.add_trace(go.Scattergeo(locations=iso, locationmode="ISO-3", text=txt, mode="markers+text", textposition="top center",
                                marker=dict(size=16, color=[ui.UP if v > 0 else (ui.DOWN if v < 0 else ui.FLAT) for v in vals],
                                            opacity=.85, line=dict(width=1.5, color="white")),
                                textfont=dict(size=12), hovertext=hov, hoverinfo="text"))
    fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="rgba(128,128,128,.35)",
                    showcoastlines=False, showland=True, landcolor="rgba(128,128,128,.12)", showocean=False,
                    showframe=False, lataxis_range=[-50, 78], bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      dragmode="pan", clickmode="event+select")
    hit = ui.new_click(ui.chart_points(fig, "mk_world", config={"displayModeBar": False, "scrollZoom": True, "doubleClick": "reset"}),
                       "mk_world")
    if hit and 0 <= hit[1] < len(names):
        ss["mk_country"] = names[hit[1]]
    ss["mk_country_seg"] = ss["mk_country"]
    pick = ui.seg("나라", names, key="mk_country_seg",
                  on_change=lambda: ss.__setitem__("mk_country", ss.get("mk_country_seg") or ss["mk_country"]))
    if pick:
        ss["mk_country"] = pick
    st.caption(f"색 = 대표지수 {per} 수익률 (빨강 상승 · 파랑 하락). 지금은 한국·미국만 · 지도에서 나라를 누르거나 위 버튼으로 고르세요.")
    return ss["mk_country"]


def _preload(country: str) -> dict:
    """처음 열 때 오래 걸리는 자료를 한꺼번에 동시에 받는다 (전에는 지수 → 지역 정보 → 지도 경계 → 업종 분류 순서로 차례차례 받았음).
    여기서 실패한 것은 그 자료를 쓰는 곳에서 다시 시도한다 (실패는 캐시되지 않음)."""
    import geodata as gd
    import sectors

    jobs = {"quotes": load_quotes, "snap": load_krx_snapshot}
    if country == "한국":
        jobs.update(regions=gd.kr_regions, geo=gd.kr_geojson, wics=sectors.wics_map)
    elif country == "미국":
        jobs.update(usmap=gd.us_frame)
    out = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = {k: ex.submit(ui.with_ctx(fn)) for k, fn in jobs.items()}
        for k, f in futs.items():
            try:
                out[k] = f.result()
            except Exception as e:  # noqa: BLE001
                out[k] = e
    return out


def render_market() -> None:
    country0 = st.session_state.get("mk_country", COUNTRIES["KOR"][0])
    with st.spinner("시장 데이터를 불러오는 중..."):
        got = _preload(country0)
    close = kr3y = snap = None
    errs = []
    for nm, k in (("해외·지수", "quotes"), ("국내 종목", "snap")):
        v = got.get(k)
        if isinstance(v, Exception):
            errs.append(f"{nm}: {v}")
        elif k == "quotes":
            close = v
        else:
            snap = v

    asof = close.index[-1].strftime("%Y-%m-%d") if close is not None and len(close) else ""
    st.markdown(f'<div class="qh-n">시장 개요 <span class="qh-code">{asof} 기준 · 일봉 종가(장중엔 지연 시세)</span></div>',
                unsafe_allow_html=True)
    country = _world_map(close)
    note = st.empty()  # 나라를 바꾼 직후, 아래에 이전 나라 지도가 남아 있는 동안 안내 (제목과 내용이 어긋나 보이는 것 방지)
    shown = st.session_state.get("_mk_shown")
    if shown is not None and shown != country:
        if note is not None:
            note.info(f"{country} 지도를 불러오는 중입니다. 아래 화면은 이전 나라 것이라 곧 바뀝니다.")
    _country_map(country, snap, close)
    if country == "한국":
        _kr_market(snap)
    elif country == "미국":
        _us_market()
    if note is not None:
        note.empty()
    st.session_state["_mk_shown"] = country
    _tiles_part(close, kr3y)
    _period_part(close, kr3y)
    if errs:
        st.caption("일부 데이터를 불러오지 못했습니다: " + " · ".join(errs))
    st.caption("출처: Yahoo Finance · KRX(FinanceDataReader) · WICS(와이즈인덱스) · 나스닥 · S&P 500(GICS). 투자 판단의 참고 자료로만 사용하세요.")
    # 화면을 다 그린 뒤에 그 나라 전 종목 2년 일봉을 뒤에서 받기 시작 (먼저 시작하면 첫 화면 다운로드와 겹쳐 느려짐)
    try:
        import geodata as gd

        iso = {v[0]: k for k, v in COUNTRIES.items()}.get(country)
        if iso:
            gd.prewarm(iso)
    except Exception:  # noqa: BLE001
        pass


# 아래 부분들은 각각 따로 다시 그린다 (ui.fragment): 한 곳의 버튼을 눌러도 다른 지도·표를 다시 계산하지 않아 빠르다.
@ui.fragment
def _country_map(country, snap, close) -> None:
    try:
        import geomap

        geomap.render(country, snap, close)
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ in ("RerunException", "StopException"):  # 화면 다시 그리기·페이지 이동은 그대로 보낸다
            raise
        st.caption(f"나라 지도를 그리지 못했습니다: {e}")


@ui.fragment
def _tiles_part(close, kr3y) -> None:
    exp, op = ui.more("지수 · 환율 · 금리 · 원자재 · 가상자산 시세", key="mk_more_tiles")
    if op:
        with exp:
            for gi, (title, items) in enumerate(GROUPS):
                ui.section(title)
                _tiles([(name, _series(close, kr3y, sym, kind), kind) for sym, name, kind in items])
                names = [name for _s, name, _k in items]
                pick = ui.seg("아래에서 고르면 차트가 열립니다", names, key=f"mk_tile_pick_{gi}", collapsed=False)
                if pick in names:
                    sym, name, kind = items[names.index(pick)]
                    _detail(sym, name, kind, close, key=f"mk_tile_det_{gi}")



@ui.fragment
def _period_part(close, kr3y) -> None:
    all_rows = [(name, _series(close, kr3y, sym, kind), kind) for _t, items in GROUPS for sym, name, kind in items]
    exp, op = ui.more("기간별 등락률 더보기 (1주 · 1개월 · 3개월 · 연초 이후 · 1년)", key="mk_more_ret")
    if op:
        with exp:
            rows, row_items = [], []
            for (name, s, kind), (sym, _n, _k) in zip(all_rows, [it for _t, items in GROUPS for it in items]):
                if len(s) < 2:
                    continue
                row_items.append((sym, name, kind))
                if kind == "rate":
                    def bp(days=None, ytd=False, s=s):
                        if ytd:
                            b = s[s.index < pd.Timestamp(s.index[-1].year, 1, 1)]
                        else:
                            b = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
                        return f"{(float(s.iloc[-1]) - float(b.iloc[-1])) * 100:+.0f}bp" if len(b) else "-"
                    rows.append({"지표": name, "현재": _fmt(float(s.iloc[-1]), kind), "1주": bp(7), "1개월": bp(30), "3개월": bp(91),
                                 "연초 이후": bp(ytd=True), "1년": bp(365)})
                else:
                    f = lambda x: "-" if x != x else f"{x:+.2f}%"  # noqa: E731
                    rows.append({"지표": name, "현재": _fmt(float(s.iloc[-1]), kind), "1주": f(_ret(s, 7)), "1개월": f(_ret(s, 30)),
                                 "3개월": f(_ret(s, 91)), "연초 이후": f(_ret(s, ytd=True)), "1년": f(_ret(s, 365))})
            t = pd.DataFrame(rows)
            pcols = ["1주", "1개월", "3개월", "연초 이후", "1년"]

            def _color(v):  # 오르면 빨강, 내리면 파랑 (0이거나 값이 없으면 그대로)
                v = str(v)
                if not any(ch in "123456789" for ch in v):
                    return ""
                if v.startswith("+"):
                    return f"color:{ui.UP};font-weight:600"
                if v.startswith("-"):
                    return f"color:{ui.DOWN};font-weight:600"
                return ""

            sty = t.style
            sty = (sty.map if hasattr(sty, "map") else sty.applymap)(_color, subset=pcols)
            i = ui.table_pick(sty, "mk_ret_tbl", use_container_width=True, hide_index=True)
            st.caption("금리는 bp(0.01%p) 변화, 나머지는 % 등락률입니다. 행을 누르면 아래에 그 지표의 차트가 나옵니다.")
            if i is not None and i < len(row_items):
                sym, name, kind = row_items[i]
                ui.section(f"{name} 차트")
                _detail(sym, name, kind, close, key="mk_ret_det")
