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


@st.cache_data(ttl=300, show_spinner=False)
def load_quotes() -> pd.DataFrame:
    """Yahoo 일별 종가 (최근 1년). 열 = 심볼."""
    import yfinance as yf

    syms = [s for _, items in GROUPS for s, _, _ in items]
    d = yf.download(syms, period="1y", interval="1d", progress=False, auto_adjust=False, threads=True)
    close = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None) if getattr(close.index, "tz", None) else pd.to_datetime(close.index)
    return close.sort_index()


@st.cache_data(ttl=300, show_spinner=False)
def load_krx_snapshot() -> pd.DataFrame:
    """KRX 전 종목 당일 시세 (현재가·등락률·거래대금·시가총액)."""
    import FinanceDataReader as fdr

    df = fdr.StockListing("KRX")
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


def _series(close: pd.DataFrame, kr3y, sym: str, kind: str) -> pd.Series:
    if close is None or sym not in close.columns:
        return pd.Series(dtype=float)
    s = close[sym].dropna()
    if sym in ("^TNX", "^IRX") and len(s) and float(s.iloc[-1]) > 20:  # 예전 표기(×10) 대비
        s = s / 10
    if kind == "jpy":
        s = s * 100
    return s


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
        marker=dict(colors=[float(x) for x in np.clip(chg, -3, 3)], cmin=-3, cmax=3, showscale=False,
                    colorscale=[[0, "#1E5BC6"], [0.35, "#4F7CC9"], [0.5, "#4A4F5A"], [0.65, "#CF5A5D"], [1, "#C8232C"]],
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


def _map_controls(prefix: str, markets: list | None, sizes: list, default_n: int):
    """시장 선택 + 종목 수(상위 N) 선택. 선택된 '상위 N'을 한 번 더 누르면 선택이 풀리고 n=None(그 시장의 모든 종목)."""
    try:
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    except TypeError:
        c1, c2 = st.columns([2, 1])
    seg = getattr(st, "segmented_control", None) or getattr(st, "pills", None)
    mk, n = (markets[0] if markets else None), default_n
    if seg:
        if markets:
            with c1:
                mk = seg("시장", markets, default=markets[0], key=f"{prefix}_mk", label_visibility="collapsed") or markets[0]
        with c2:
            n = seg("종목 수", sizes, default=default_n, key=f"{prefix}_n", label_visibility="collapsed",
                    format_func=lambda x: f"상위 {x}")
    return mk, (int(n) if n else None)


def _size_note(n, total: int, what: str) -> str:
    return (f"시가총액 상위 {n}개 · 선택한 '상위 {n}'을 다시 누르면 {what} 전체 {total:,}개를 봅니다" if n
            else f"{what} 전체 {total:,}개 종목 · 위 버튼을 누르면 상위 종목만 봅니다")


def _heatmap(snap: pd.DataFrame) -> None:
    need = {"Code", "Name", "Marcap", "ChangesRatio"}
    if not need.issubset(snap.columns):
        return
    ui.section("국내 시장 지도", "섹터 → 업종 → 종목 · 칸 크기 = 시가총액 · 색 = 오늘 등락률 · 칸을 누르면 확대")
    mk, n = _map_controls("mk_map", ["전체", "KOSPI", "KOSDAQ"], [100, 200, 400], 200)
    m = snap.dropna(subset=["Marcap", "ChangesRatio"]).copy()
    if mk != "전체" and "Market" in m.columns:
        m = m[m["Market"].astype(str).str.upper().str.startswith(mk)]
    m["Code"] = m["Code"].astype(str).str.zfill(6)
    m = m[m["Code"].str.endswith("0")]  # 우선주 제외
    total = len(m)
    if n:
        m = m.nlargest(int(n), "Marcap")
    if m.empty:
        return
    st.caption(_size_note(n, total, mk if mk != "전체" else "코스피·코스닥"))
    import sectors

    try:
        smap, src = sectors.kr_sector_map()
        m = m.merge(smap, on="Code", how="left")
    except Exception as e:  # noqa: BLE001
        m["Sector"], m["Industry"], src = "분류 없음", "분류 없음", f"업종 분류를 받지 못함 ({e})"
    m["Key"] = m["Code"]
    _treemap(m, "국내 시장 지도", f"분류: {src}", _krw, height=560 if n else 760)


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
    items += [("S&P 500 상승 / 하락", f"{up:,} / {dn:,}", f"보합 {int((sp['ChangesRatio'] == 0).sum()):,}", up - dn),
              ("S&P 500 거래대금", f"${sp['Amount'].sum() / 1e9:,.0f}B" if sp["Amount"].notna().any() else "-")]
    if allq is not None and len(allq):
        a = allq.dropna(subset=["ChangesRatio"])
        a = a[(a["Marcap"] > 0) | a["Marcap"].isna()]
        u2, d2 = int((a["ChangesRatio"] > 0).sum()), int((a["ChangesRatio"] < 0).sum())
        items += [("미국 상장 전체 상승 / 하락", f"{u2:,} / {d2:,}", "나스닥·NYSE·AMEX", u2 - d2),
                  ("미국 상장 전체 거래대금", f"${a['Amount'].sum() / 1e9:,.0f}B")]
    ui.cards(items, cols=4)

    ui.section("미국 시장 지도 (S&P 500)", "섹터 → 업종(GICS) → 종목 · 칸 크기 = 시가총액 · 색 = 오늘 등락률 · 칸을 누르면 확대")
    _mk, n = _map_controls("us_map", None, [100, 250], 250)
    m = sp.dropna(subset=["Marcap"]).copy()
    total = len(m)
    if n:
        m = m.nlargest(n, "Marcap").copy()
    st.caption(_size_note(n, total, "S&P 500"))
    m["Key"], m["Sector"] = m["Symbol"], m["SectorKo"]
    m["FullName"], m["Name"] = m["Name"], m["Symbol"]
    _treemap(m, "미국 시장 지도", f"분류: GICS 섹터·세부 업종 · 시세: {src}", _usd, height=560 if n else 760)

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


def _with_ctx(fn):
    """작업 스레드에서도 Streamlit 캐시가 경고 없이 동작하도록 현재 실행 정보를 붙인다."""
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

        ctx = get_script_run_ctx()
    except Exception:  # noqa: BLE001
        return fn

    def run():
        import threading

        try:
            add_script_run_ctx(threading.current_thread(), ctx)
        except Exception:  # noqa: BLE001
            pass
        return fn()

    return run


def render_market() -> None:
    with st.spinner("시장 데이터를 불러오는 중..."):
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_q, f_x = (ex.submit(_with_ctx(fn)) for fn in (load_quotes, load_krx_snapshot))
            close = kr3y = snap = None
            errs = []
            for nm, f in (("해외·지수", f_q), ("국내 종목", f_x)):
                try:
                    v = f.result()
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{nm}: {e}")
                    continue
                if nm == "해외·지수":
                    close = v
                else:
                    snap = v

    asof = close.index[-1].strftime("%Y-%m-%d") if close is not None and len(close) else ""
    st.markdown(f'<div class="qh-n">시장 개요 <span class="qh-code">{asof} 기준 · 일봉 종가(장중엔 지연 시세)</span></div>',
                unsafe_allow_html=True)
    all_rows = []
    for title, items in GROUPS:
        ui.section(title)
        rows = [(name, _series(close, kr3y, sym, kind), kind) for sym, name, kind in items]
        all_rows += rows
        _tiles(rows)

    # ---- 국내 시장 상황
    if snap is not None and not snap.empty and "ChangesRatio" in snap.columns:
        ui.section("국내 주식시장", "KRX 전 종목 (장중엔 지연 시세)")
        items = []
        for mk in ("KOSPI", "KOSDAQ"):
            m = snap[snap["Market"].astype(str).str.upper().str.startswith(mk)] if "Market" in snap.columns else snap
            up, dn = int((m["ChangesRatio"] > 0).sum()), int((m["ChangesRatio"] < 0).sum())
            flat = int((m["ChangesRatio"] == 0).sum())
            amt = float(m["Amount"].sum()) if "Amount" in m.columns else np.nan
            items += [(f"{mk} 상승 / 하락", f"{up:,} / {dn:,}", f"보합 {flat:,}", up - dn),
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

    # ---- 미국 시장 상황
    _us_market()

    # ---- 기간별 수익률 (더보기)
    exp, op = ui.more("기간별 등락률 더보기 (1주 · 1개월 · 3개월 · 연초 이후 · 1년)", key="mk_more_ret")
    if op:
        with exp:
            rows = []
            for name, s, kind in all_rows:
                if len(s) < 2:
                    continue
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
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("금리는 bp(0.01%p) 변화, 나머지는 % 등락률입니다.")
    if errs:
        st.caption("일부 데이터를 불러오지 못했습니다: " + " · ".join(errs))
    st.caption("출처: Yahoo Finance · KRX(FinanceDataReader) · WICS(와이즈인덱스) · 나스닥 · S&P 500(GICS). 투자 판단의 참고 자료로만 사용하세요.")
