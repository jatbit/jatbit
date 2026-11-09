"""기업 리서치 탭 (해외 종목): Yahoo Finance(yfinance)에서 기업 개요·밸류에이션·수익성·재무제표·애널리스트 의견·일정·뉴스를 모아 보여준다."""
import html
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from options import OPT_PRESETS, is_us_ticker, mgrid

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None

REC_KO = {"strong_buy": "적극 매수", "buy": "매수", "hold": "보유", "underperform": "비중 축소", "sell": "매도",
          "strongBuy": "적극 매수", "strongSell": "적극 매도", "none": "-"}


# ----------------------------------------------------------------------
# 데이터
# ----------------------------------------------------------------------
def _try(fn, tries=4, base=1.5):
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


def _safe(fn, default=None):
    try:
        v = fn()
        return default if v is None else v
    except Exception:  # noqa: BLE001
        return default


@st.cache_data(ttl=60 * 60, show_spinner="기업 정보를 불러오는 중... (Yahoo 응답이 느리면 다시 시도합니다)")
def load_research(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = _try(lambda: t.info) or {}
    if not info or (not info.get("longName") and not info.get("shortName") and not info.get("quoteType")):
        raise RuntimeError("기업 정보를 찾지 못했습니다 (티커를 확인해 주세요)")
    out = {"info": info}
    out["income"] = _safe(lambda: t.income_stmt, pd.DataFrame())
    out["balance"] = _safe(lambda: t.balance_sheet, pd.DataFrame())
    out["cashflow"] = _safe(lambda: t.cashflow, pd.DataFrame())
    out["recs"] = _safe(lambda: t.recommendations, pd.DataFrame())
    out["targets"] = _safe(lambda: dict(t.analyst_price_targets), {})
    out["calendar"] = _safe(lambda: dict(t.calendar), {})
    out["news"] = _safe(lambda: list(t.news), [])
    out["hist"] = _safe(lambda: t.history(period="1y")["Close"].dropna(), pd.Series(dtype=float))
    return out


# ----------------------------------------------------------------------
# 표시 도우미
# ----------------------------------------------------------------------
def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:  # noqa: BLE001
        return None


def big(x, cur="$"):
    v = _num(x)
    if v is None:
        return "-"
    a = abs(v)
    for unit, d in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if a >= d:
            return f"{'-' if v < 0 else ''}{cur}{a / d:,.2f}{unit}"
    return f"{'-' if v < 0 else ''}{cur}{a:,.2f}"


def pct(x, nd=1):
    v = _num(x)
    return "-" if v is None else f"{v * 100:.{nd}f}%"


def ratio(x, nd=2):
    v = _num(x)
    return "-" if v is None else f"{v:,.{nd}f}"


def _row(df: pd.DataFrame, *names):
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def _div_yield(info: dict):
    px = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    rate = _num(info.get("dividendRate"))
    if px and rate:
        return rate / px
    y = _num(info.get("trailingAnnualDividendYield"))
    return y


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
def _set_research(tk):
    st.session_state["rs_ticker"] = tk


def _send_to_options(tk):
    st.session_state["opt_ticker"] = tk


def render_research() -> None:
    if yf is None:
        st.error("yfinance 패키지가 설치되지 않았습니다.")
        return
    st.caption("미국 상장 종목·ETF의 기업 정보를 Yahoo Finance에서 모아 보여줍니다 (일부 항목은 종목에 따라 비어 있을 수 있음). 국내 종목은 다음 단계에서 지원 예정입니다.")

    res = st.session_state.get("pf_opt_result") or {}
    pf = [it["code"] for it in res.get("items", []) if not it.get("domestic") and is_us_ticker(it["code"])]
    presets = list(dict.fromkeys(pf + OPT_PRESETS))[:10]
    if hasattr(st, "pills"):
        st.pills("빠른 선택", presets, key="rs_pill",
                 on_change=lambda: _set_research(st.session_state.get("rs_pill")) if st.session_state.get("rs_pill") else None)
    raw = st.text_input("미국 티커 (예: AAPL, MSFT, NVDA)", key="rs_ticker", placeholder="AAPL")
    ticker = (raw or "").strip().upper()
    if not ticker:
        st.info("티커를 입력하거나 위에서 고르세요.")
        return
    if not is_us_ticker(ticker):
        st.warning("지금은 미국 종목만 지원합니다. 미국 티커를 입력해 주세요.")
        return
    try:
        R = load_research(ticker)
    except Exception as e:  # noqa: BLE001
        st.error(f"{ticker} 정보를 불러오지 못했습니다: {e}. 잠시 후 다시 시도해 주세요.")
        return
    info = R["info"]
    cur = "$" if (info.get("currency") or "USD") == "USD" else ""

    # ---- 1. 개요
    name = info.get("longName") or info.get("shortName") or ticker
    st.subheader(f"{name} ({ticker})")
    meta = [x for x in [info.get("sector"), info.get("industry"), info.get("country"),
                        f"직원 {int(info['fullTimeEmployees']):,}명" if _num(info.get("fullTimeEmployees")) else None,
                        info.get("exchange")] if x]
    st.caption(" · ".join(map(str, meta)) + (f" · [홈페이지]({info['website']})" if info.get("website") else ""))
    if info.get("longBusinessSummary"):
        with st.expander("사업 설명 (영문 원문)"):
            st.write(info["longBusinessSummary"].replace("$", "\\$"))
    st.button("이 종목을 '옵션 조회' 탭에 설정", key="rs_to_opt", on_click=_send_to_options, args=(ticker,))

    # ---- 2. 주가·밸류에이션
    px = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    lo52, hi52 = _num(info.get("fiftyTwoWeekLow")), _num(info.get("fiftyTwoWeekHigh"))
    pos52 = (px - lo52) / (hi52 - lo52) if px and lo52 and hi52 and hi52 > lo52 else None
    st.markdown("##### 주가 · 밸류에이션")
    mgrid([("현재가", f"{cur}{px:,.2f}" if px else "-"),
           ("시가총액", big(info.get("marketCap"), cur)),
           ("기업가치(EV)", big(info.get("enterpriseValue"), cur)),
           ("52주 범위", f"{lo52:,.2f} ~ {hi52:,.2f}" if lo52 and hi52 else "-",
            f"현재가는 52주 범위의 {pos52 * 100:.0f}% 위치" if pos52 is not None else ""),
           ("PER (최근 12개월)", ratio(info.get("trailingPE"), 1)),
           ("선행 PER", ratio(info.get("forwardPE"), 1)),
           ("PBR", ratio(info.get("priceToBook"))),
           ("EV/EBITDA", ratio(info.get("enterpriseToEbitda"), 1)),
           ("PEG", ratio(info.get("trailingPegRatio") or info.get("pegRatio"))),
           ("PSR", ratio(info.get("priceToSalesTrailing12Months"))),
           ("배당수익률", pct(_div_yield(info), 2)),
           ("베타", ratio(info.get("beta")))])

    # ---- 3. 수익성·성장·재무건전성
    st.markdown("##### 수익성 · 성장 · 재무건전성")
    de = _num(info.get("debtToEquity"))
    mgrid([("매출 성장률(YoY)", pct(info.get("revenueGrowth"))),
           ("이익 성장률(YoY)", pct(info.get("earningsGrowth"))),
           ("매출총이익률", pct(info.get("grossMargins"))),
           ("영업이익률", pct(info.get("operatingMargins"))),
           ("순이익률", pct(info.get("profitMargins"))),
           ("ROE", pct(info.get("returnOnEquity"))),
           ("ROA", pct(info.get("returnOnAssets"))),
           ("부채비율(D/E)", f"{de:.0f}%" if de is not None else "-", "총부채 ÷ 자기자본 (Yahoo는 % 단위로 제공)"),
           ("현금성 자산", big(info.get("totalCash"), cur)),
           ("총차입금", big(info.get("totalDebt"), cur)),
           ("잉여현금흐름(FCF)", big(info.get("freeCashflow"), cur)),
           ("유동비율", ratio(info.get("currentRatio")))])

    # ---- 4. 애널리스트
    st.markdown("##### 애널리스트 의견")
    tg = R["targets"] or {}
    mean_t = _num(tg.get("mean")) or _num(info.get("targetMeanPrice"))
    up = (mean_t / px - 1) if mean_t and px else None
    rec = info.get("recommendationKey") or "none"
    mgrid([("목표주가 평균", f"{cur}{mean_t:,.2f}" if mean_t else "-"),
           ("상승 여력", pct(up), "목표주가 평균 ÷ 현재가 − 1"),
           ("목표주가 범위", f"{_num(tg.get('low')) or _num(info.get('targetLowPrice')) or 0:,.0f} ~ "
                         f"{_num(tg.get('high')) or _num(info.get('targetHighPrice')) or 0:,.0f}" if mean_t else "-"),
           ("투자의견", f"{REC_KO.get(rec, rec)} ({ratio(info.get('recommendationMean'), 2)})",
            "1=적극 매수 … 5=매도"),
           ("의견 수", f"{int(info['numberOfAnalystOpinions'])}명" if _num(info.get("numberOfAnalystOpinions")) else "-")])
    recs = R["recs"]
    if isinstance(recs, pd.DataFrame) and not recs.empty and {"strongBuy", "buy", "hold", "sell", "strongSell"} <= set(recs.columns):
        r = recs.copy()
        labels = {"0m": "이번 달", "-1m": "1개월 전", "-2m": "2개월 전", "-3m": "3개월 전"}
        r["기간"] = r["period"].map(lambda p: labels.get(p, p)) if "period" in r else [str(i) for i in range(len(r))]
        r = r.iloc[::-1]
        fig = go.Figure()
        for col, nm, color in [("strongBuy", "적극 매수", "#1B7F3B"), ("buy", "매수", "#5BB56E"), ("hold", "보유", "#B0B4BA"),
                               ("sell", "매도", "#E8875B"), ("strongSell", "적극 매도", "#C93C3C")]:
            fig.add_trace(go.Bar(y=r["기간"], x=r[col], name=nm, orientation="h", marker_color=color,
                                 hovertemplate=nm + " %{x}명<extra></extra>"))
        fig.update_layout(barmode="stack", height=200, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), xaxis_title="애널리스트 수")
        st.plotly_chart(fig, use_container_width=True)

    # ---- 5. 연간 재무제표
    st.markdown("##### 연간 재무제표 (최근 4개 회계연도)")
    inc, bal, cf = R["income"], R["balance"], R["cashflow"]
    items = [
        ("매출", _row(inc, "Total Revenue", "Operating Revenue")),
        ("매출총이익", _row(inc, "Gross Profit")),
        ("영업이익", _row(inc, "Operating Income")),
        ("EBITDA", _row(inc, "EBITDA", "Normalized EBITDA")),
        ("순이익", _row(inc, "Net Income", "Net Income Common Stockholders")),
        ("희석 EPS", _row(inc, "Diluted EPS", "Basic EPS")),
        ("영업현금흐름", _row(cf, "Operating Cash Flow")),
        ("잉여현금흐름", _row(cf, "Free Cash Flow")),
        ("총자산", _row(bal, "Total Assets")),
        ("총부채", _row(bal, "Total Liabilities Net Minority Interest")),
        ("자기자본", _row(bal, "Stockholders Equity", "Common Stock Equity")),
    ]
    items = [(k, v) for k, v in items if v is not None]
    if items:
        fs = pd.DataFrame({k: pd.to_numeric(v, errors="coerce") for k, v in items}).T
        fs = fs.dropna(axis=1, how="all")
        fs = fs[sorted(fs.columns)[-4:]]
        fs.columns = [pd.Timestamp(c).strftime("%Y-%m") for c in fs.columns]
        show = fs.copy().astype(object)
        for k in show.index:
            show.loc[k] = [ratio(v) if k == "희석 EPS" else big(v, cur) for v in fs.loc[k]]
        if "매출" in fs.index:
            rev = fs.loc["매출"]
            show.loc["매출 성장률"] = [pct(v) for v in rev.pct_change()]
            for k, lab in (("영업이익", "영업이익률"), ("순이익", "순이익률")):
                if k in fs.index:
                    show.loc[lab] = [pct(a / b) if b else "-" for a, b in zip(fs.loc[k], rev)]
        st.dataframe(show, use_container_width=True)
        if "매출" in fs.index:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=fs.columns, y=fs.loc["매출"], name="매출", marker_color="#4C9BE8",
                                 hovertemplate="%{x} 매출 %{y:$,.0f}<extra></extra>"))
            if "순이익" in fs.index:
                fig.add_trace(go.Bar(x=fs.columns, y=fs.loc["순이익"], name="순이익", marker_color="#F2A93B",
                                     hovertemplate="%{x} 순이익 %{y:$,.0f}<extra></extra>"))
            if "영업이익" in fs.index:
                fig.add_trace(go.Scatter(x=fs.columns, y=(fs.loc["영업이익"] / fs.loc["매출"] * 100), name="영업이익률(%)",
                                         yaxis="y2", mode="lines+markers", line=dict(color="#E5534B"),
                                         hovertemplate="%{x} 영업이익률 %{y:.1f}%<extra></extra>"))
            fig.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                              yaxis=dict(title="금액"), yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("재무제표를 불러오지 못했습니다 (ETF이거나 Yahoo 응답 지연).")

    # ---- 6. 일정
    cal = R["calendar"] or {}
    ev = []
    ed = cal.get("Earnings Date")
    if ed:
        ed = ed if isinstance(ed, (list, tuple)) else [ed]
        ev.append(("다음 실적 발표", " ~ ".join(str(pd.Timestamp(d).date()) for d in ed[:2])))
    if cal.get("Ex-Dividend Date"):
        ev.append(("배당락일", str(pd.Timestamp(cal["Ex-Dividend Date"]).date())))
    if cal.get("Dividend Date"):
        ev.append(("배당 지급일", str(pd.Timestamp(cal["Dividend Date"]).date())))
    if _num(cal.get("Earnings Average")):
        ev.append(("다음 분기 EPS 예상", ratio(cal.get("Earnings Average"))))
    if ev:
        st.markdown("##### 주요 일정")
        mgrid(ev)

    # ---- 7. 뉴스
    news = R["news"] or []
    rows = []
    for n in news[:12]:
        c = n.get("content") if isinstance(n, dict) and isinstance(n.get("content"), dict) else n
        if not isinstance(c, dict):
            continue
        title = c.get("title")
        url = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else None
        url = url or (c.get("clickThroughUrl") or {}).get("url") if isinstance(c.get("clickThroughUrl"), dict) else url
        url = url or c.get("link")
        prov = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else c.get("publisher")
        when = c.get("pubDate") or c.get("displayTime")
        if not when and c.get("providerPublishTime"):
            when = datetime.utcfromtimestamp(int(c["providerPublishTime"])).isoformat()
        if title:
            rows.append((title, url, prov, str(when)[:10] if when else ""))
    if rows:
        st.markdown("##### 최근 뉴스")
        lines = []
        for title, url, prov, when in rows:
            t = html.escape(title)
            link = f'<a href="{html.escape(url, quote=True)}" target="_blank">{t}</a>' if url else t
            lines.append(f'<div style="margin:0 0 8px">{link}<br><span style="font-size:0.8rem;opacity:.65">'
                         f'{html.escape(prov or "")} · {when}</span></div>')
        st.markdown("".join(lines).replace("$", "&#36;"), unsafe_allow_html=True)
    st.caption("출처: Yahoo Finance. 지연·누락이 있을 수 있으며 투자 판단의 참고 자료로만 사용하세요.")
