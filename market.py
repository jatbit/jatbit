"""시장 개요 (첫 화면): 주요 지수·환율·금리·원자재·가상자산 시세와 국내 시장 상승·하락 종목, 시가총액·거래대금 상위 종목.

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
    ("환율 · 금리 · 변동성", [("KRW=X", "원/달러 환율", "fx"), ("DX-Y.NYB", "달러인덱스", "idx"), ("KR3Y", "국고채 3년", "rate"),
                         ("^TNX", "미국 국채 10년", "rate"), ("^VIX", "VIX (공포지수)", "idx"), ("JPYKRW=X", "원/엔 (100엔)", "jpy")]),
    ("원자재 · 가상자산", [("CL=F", "WTI 원유", "usd"), ("GC=F", "금", "usd"), ("HG=F", "구리", "usd2"),
                       ("NG=F", "천연가스", "usd2"), ("BTC-USD", "비트코인", "usd0"), ("ETH-USD", "이더리움", "usd0")]),
]
RATE_KINDS = {"KR3Y", "^TNX"}


@st.cache_data(ttl=300, show_spinner=False)
def load_quotes() -> pd.DataFrame:
    """Yahoo 일별 종가 (최근 1년). 열 = 심볼."""
    import yfinance as yf

    syms = [s for _, items in GROUPS for s, _, _ in items if s != "KR3Y"]
    d = yf.download(syms, period="1y", interval="1d", progress=False, auto_adjust=False, threads=True)
    close = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None) if getattr(close.index, "tz", None) else pd.to_datetime(close.index)
    return close.sort_index()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_kr3y() -> pd.Series:
    import FinanceDataReader as fdr

    start = (pd.Timestamp.today() - pd.Timedelta(days=380)).strftime("%Y-%m-%d")
    d = fdr.DataReader("KR3YT=RR", start)
    return d["Close"].dropna()


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
    if sym == "KR3Y":
        return kr3y if kr3y is not None else pd.Series(dtype=float)
    if close is None or sym not in close.columns:
        return pd.Series(dtype=float)
    s = close[sym].dropna()
    if sym == "^TNX" and len(s) and float(s.iloc[-1]) > 20:  # 예전 표기(×10) 대비
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
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_q, f_k, f_x = (ex.submit(_with_ctx(fn)) for fn in (load_quotes, load_kr3y, load_krx_snapshot))
            close = kr3y = snap = None
            errs = []
            for nm, f in (("해외·지수", f_q), ("국고채", f_k), ("국내 종목", f_x)):
                try:
                    v = f.result()
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{nm}: {e}")
                    continue
                if nm == "해외·지수":
                    close = v
                elif nm == "국고채":
                    kr3y = v
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
    st.caption("출처: Yahoo Finance · KRX(FinanceDataReader) · 국고채 금리(FinanceDataReader). 투자 판단의 참고 자료로만 사용하세요.")
