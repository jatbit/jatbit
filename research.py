"""기업 리서치 탭 (해외 종목): Yahoo Finance(yfinance)에서 기업 개요·밸류에이션·수익성·재무제표·애널리스트 의견·일정·뉴스를 모아 보여준다."""
# 파일 구성: research_data.py(데이터·계산) · research_chart.py(겹쳐 그리기 그래프) · 이 파일(화면)
import html
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

from options import OPT_PRESETS, is_us_ticker, mgrid

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None

from research_data import *  # noqa: F401,F403
from research_chart import *  # noqa: F401,F403


def render_research() -> None:
    if yf is None:
        st.error("yfinance 패키지가 설치되지 않았습니다.")
        return
    import kr
    import money

    res = st.session_state.get("pf_opt_result") or {}
    pf = [it["code"] if not it.get("domestic") else it["name"] for it in res.get("items", [])
          if (not it.get("domestic") and is_us_ticker(it["code"])) or (it.get("domestic") and re.fullmatch(r"\d{6}", str(it["code"])))]
    presets = list(dict.fromkeys(pf + OPT_PRESETS[:6] + ["삼성전자", "SK하이닉스"]))[:12]
    try:
        c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    except TypeError:
        c1, c2 = st.columns([5, 1])
    raw = c1.text_input("종목 (미국 티커 또는 국내 종목명·코드)", key="rs_ticker", placeholder="AAPL, 삼성전자, 005930",
                        label_visibility="collapsed")
    with c2:
        with ui.popover("설정", icon=":material/tune:"):
            ko = st.toggle("한국어 번역 (Google 자동 번역)", value=True, key="rs_ko")
            st.text_input("OpenDART API 키 (선택)", key="dart_key_input", type="password",
                          help="opendart.fss.or.kr에서 무료로 발급. 국내 종목 10년 재무제표·지분 공시에 씁니다. "
                               "계속 쓰려면 Streamlit 앱 설정의 Secrets에 DART_API_KEY로 넣으세요.")
            st.caption("Secrets에 등록된 키가 있으면 비워 둬도 됩니다." if kr.dart_key() else "키가 없으면 Yahoo 재무제표(약 4년)를 씁니다.")
    if hasattr(st, "pills"):
        st.pills("빠른 선택", presets, key="rs_pill", label_visibility="collapsed",
                 on_change=lambda: _set_research(st.session_state.get("rs_pill")) if st.session_state.get("rs_pill") else None)
    q = (raw or "").strip()
    if not q:
        st.info("티커나 종목명을 입력하거나 위에서 고르세요.")
        return
    domestic, code, kname = False, None, None
    if kr.is_kr_query(q):
        try:
            cands = kr.resolve(q)
        except Exception as e:  # noqa: BLE001
            st.error(f"국내 종목 목록을 불러오지 못했습니다: {e}")
            return
        if not cands:
            st.warning(f"'{q}'에 해당하는 국내 상장 종목을 찾지 못했습니다.")
            return
        if len(cands) > 1 and cands[0]["name"] != q and not re.fullmatch(r"\d{6}", q):
            labels = [f"{c['name']} ({c['code']}, {c['market']})" for c in cands]
            i = st.selectbox("종목 선택", range(len(cands)), format_func=lambda j: labels[j], key=f"rs_pick_{q}")
            c = cands[i]
        else:
            c = cands[0]
        domestic, code, kname, ticker = True, c["code"], c["name"], c["yahoo"]
    else:
        ticker = q.upper()
        if not is_us_ticker(ticker):
            st.warning("미국 티커 또는 국내 종목명·6자리 코드를 입력해 주세요.")
            return
    try:
        R = load_research(ticker)
    except Exception as e:  # noqa: BLE001
        if domestic and ticker.endswith(".KS"):  # 코스피·코스닥 구분이 틀린 경우 다른 쪽으로 다시 시도
            try:
                ticker = ticker.replace(".KS", ".KQ")
                R = load_research(ticker)
            except Exception:  # noqa: BLE001
                st.error(f"{kname} 정보를 불러오지 못했습니다: {e}. 잠시 후 다시 시도해 주세요.")
                return
        else:
            st.error(f"{ticker} 정보를 불러오지 못했습니다: {e}. 잠시 후 다시 시도해 주세요.")
            return
    info = R["info"]
    ccy = info.get("currency") or ("KRW" if domestic else "USD")
    money.set_cur(ccy)
    cur = "$" if ccy == "USD" else ("₩" if ccy == "KRW" else "")

    # ---- 시세 헤더
    name = kname or info.get("longName") or info.get("shortName") or ticker
    if domestic:
        meta = [c.get("market"), c.get("sector") or info.get("sector"), c.get("industry") or info.get("industry")]
    else:
        meta = [info.get("exchange"), info.get("sector"), info.get("industry"), info.get("country")]
    meta = [html.escape(str(x)) for x in meta if x]
    if info.get("website"):
        meta.append(f'<a href="{html.escape(info["website"], quote=True)}" target="_blank">홈페이지</a>')
    px = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    prev = _num(info.get("regularMarketPreviousClose")) or _num(info.get("previousClose"))
    chg = (px - prev) if px and prev else None
    ui.quote_header(name, code or ticker, " · ".join(meta), money.price(px) if px else "-",
                    money.price(abs(chg)) if chg is not None else None, (chg / prev * 100) if chg is not None else None,
                    "Yahoo 지연 시세")
    if True:
        try:  # PER(직전 사업연도 EPS 기준)·PBR 계산용 연간 재무제표 (첫 탭에서도 쓰므로 캐시됨)
            R["_F"] = _financials(ticker, R, domestic, code)[0]
        except Exception:  # noqa: BLE001
            pass
    _key_stats(R, info, cur, px)

    tabs = ui.lazy_tabs(["재무·밸류에이션", "뉴스·기업 개요", "동종업계 비교", "실적·추정치", "DCF 가치평가", "신용위험·회사채", "지분 구조·내부자"],
                        key="rs_tabs")
    if tabs[0][1]:
        with tabs[0][0]:
            F, src, ferr = _financials(ticker, R, domestic, code)
            if src == "Yahoo" and ferr and not domestic:
                st.caption("SEC 10개년 재무제표가 없어 Yahoo 재무제표(약 4년)를 씁니다.")
            _render_main(ticker, R, info, cur, ko, F, src, domestic)
    if tabs[1][1]:
        with tabs[1][0]:
            summary = info.get("longBusinessSummary")
            if summary:
                tr = translate_ko(summary) if ko else None
                ui.section("사업 설명")
                if tr:
                    st.write(tr.replace("$", "\\$"))
                    with st.expander("영문 원문"):
                        st.write(summary.replace("$", "\\$"))
                else:
                    if ko:
                        st.caption("번역에 실패해 영문 원문을 보여줍니다.")
                    st.write(summary.replace("$", "\\$"))
            facts = []
            if _num(info.get("fullTimeEmployees")):
                facts.append(f"직원 {int(info['fullTimeEmployees']):,}명")
            if info.get("city") or info.get("country"):
                facts.append(" ".join(x for x in [info.get("city"), info.get("country")] if x))
            if facts:
                st.caption(" · ".join(facts))
            if not domestic:
                st.button("이 종목을 '옵션' 페이지에 설정", key="rs_to_opt", on_click=_send_to_options, args=(ticker,))
            from news import render_latest_news
            render_latest_news(code or ticker, kname or info.get("shortName") or name, R.get("news") or [], ko, translate_ko,
                               lang="ko" if domestic else "en")
            _render_yahoo_news(R, ko)
    if tabs[2][1]:
        with tabs[2][0]:
            from peers import render_peers
            render_peers(ticker, R, domestic, code)
    if tabs[3][1]:
        with tabs[3][0]:
            from earnings import render_earnings
            render_earnings(ticker, R, domestic)
    if tabs[4][1]:
        with tabs[4][0]:
            F, src, _ = _financials(ticker, R, domestic, code)
            from dcf import render_dcf
            render_dcf(ticker, R, F, src, domestic)
    if tabs[5][1]:
        with tabs[5][0]:
            F, src, _ = _financials(ticker, R, domestic, code)
            from credit import render_credit
            render_credit(ticker, R, F, domestic)
    if tabs[6][1]:
        with tabs[6][0]:
            if domestic:
                kr.render_kr_ownership(code, R)
            else:
                from insider import render_insider
                render_insider(ticker, R)


def _key_stats(R: dict, info: dict, cur: str, px) -> None:
    """MTS·Yahoo 요약 화면처럼 가장 자주 보는 지표만 먼저."""
    import money

    lo52, hi52 = _num(info.get("fiftyTwoWeekLow")), _num(info.get("fiftyTwoWeekHigh"))
    pos52 = (px - lo52) / (hi52 - lo52) if px and lo52 and hi52 and hi52 > lo52 else None
    tg = R.get("targets") or {}
    mean_t = _num(tg.get("mean")) or _num(info.get("targetMeanPrice"))
    up = (mean_t / px - 1) if mean_t and px else None
    rec = info.get("recommendationKey") or "none"
    n_op = int(info["numberOfAnalystOpinions"]) if _num(info.get("numberOfAnalystOpinions")) else None
    rng = (f"{lo52:,.0f} ~ {hi52:,.0f}" if cur == "₩" else f"{lo52:,.2f} ~ {hi52:,.2f}") if lo52 and hi52 else "-"
    pe, pb = _pe_pb(info, px, R.get("_F"), R)
    pe_a = annual_pe_now(px, R.get("_F"), R.get("splits"))
    sub = " · ".join(x for x in ([f"12M {ratio(pe, 1)}"] if pe_a and pe else []) +
                     ([f"선행 {ratio(info.get('forwardPE'), 1)}"] if _num(info.get("forwardPE")) else []))
    ui.cards([
        ("시가총액", big(info.get("marketCap"), cur)),
        ("PER", ratio(pe_a or pe, 1), sub),
        ("PBR", ratio(pb)),
        ("배당수익률", pct(_div_yield(info), 2)),
        ("ROE", pct(info.get("returnOnEquity")), f"영업이익률 {pct(info.get('operatingMargins'))}" if _num(info.get("operatingMargins")) is not None else ""),
        ("52주 범위", rng, f"현재가는 범위의 {pos52 * 100:.0f}% 위치" if pos52 is not None else ""),
        ("목표주가 평균", money.price(mean_t) if mean_t else "-", f"상승 여력 {pct(up)}" if up is not None else "", up),
        ("투자의견", REC_KO.get(rec, "-") if rec != "none" else "-", f"애널리스트 {n_op}명" if n_op else ""),
    ], cols=4)


def _render_main(ticker: str, R: dict, info: dict, cur: str, ko: bool, F=None, src: str = "SEC 10-K", domestic: bool = False) -> None:
    import money

    # ---- 2. 주가·밸류에이션
    px = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    lo52, hi52 = _num(info.get("fiftyTwoWeekLow")), _num(info.get("fiftyTwoWeekHigh"))
    pos52 = (px - lo52) / (hi52 - lo52) if px and lo52 and hi52 and hi52 > lo52 else None
    st.markdown("##### 주가 · 밸류에이션")
    pe_now, pb_now = _pe_pb(info, px, F, R)
    n_eps, n_info = ntm_eps(info, R.get("eps_est"))
    fwd_eps = _num(info.get("forwardEps"))
    fwd_note = "Yahoo 'Forward P/E' = 현재가 ÷ forwardEps"
    if n_info:
        e0, e1, w, fy_end = n_info
        which = min(((abs((fwd_eps or 0) - e0), "이번 회계연도"), (abs((fwd_eps or 0) - e1), "다음 회계연도")))[1] if fwd_eps else "?"
        fwd_note += f" ({fwd_eps:.2f}, {which} 추정치 기준)" if fwd_eps else ""
    mgrid([("현재가", money.price(px) if px else "-"),
           ("시가총액", big(info.get("marketCap"), cur)),
           ("기업가치(EV)", big(info.get("enterpriseValue"), cur)),
           ("52주 범위", (f"{lo52:,.0f} ~ {hi52:,.0f}" if cur == "₩" else f"{lo52:,.2f} ~ {hi52:,.2f}") if lo52 and hi52 else "-",
            f"현재가는 52주 범위의 {pos52 * 100:.0f}% 위치" if pos52 is not None else ""),
           ("PER", ratio(annual_pe_now(px, F, R.get("splits")) or pe_now, 1), "직전 사업연도 EPS 기준"),
           ("12M PER", ratio(pe_now, 1), "최근 4개 분기 EPS 합 기준"),
           ("선행 PER (Yahoo)", ratio(info.get("forwardPE"), 1), f"{which} 추정치 기준" if n_info and fwd_eps else ""),
           ("12M 선행 PER (NTM)", ratio(px / n_eps, 1) if px and n_eps and n_eps > 0 else "-"),
           ("PBR", ratio(pb_now)),
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
           ("부채비율(D/E)", f"{de:.0f}%" if de is not None else "-"),
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
    mgrid([("목표주가 평균", money.price(mean_t) if mean_t else "-"),
           ("상승 여력", pct(up)),
           ("목표주가 범위", f"{_num(tg.get('low')) or _num(info.get('targetLowPrice')) or 0:,.0f} ~ "
                         f"{_num(tg.get('high')) or _num(info.get('targetHighPrice')) or 0:,.0f}" if mean_t else "-"),
           ("투자의견", f"{REC_KO.get(rec, rec)} ({ratio(info.get('recommendationMean'), 2)})"),
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
    sec = F if F is not None and F.shape[1] >= 2 else None
    if sec is not None:
        st.markdown(f"##### 연간 재무제표 (최근 {sec.shape[1]}개 회계연도 · {src})")
        fs = sec.loc[[r for r in SEC_SHOW if r in sec.index]].copy()
        fs.columns = [pd.Timestamp(c).strftime("%Y-%m") for c in fs.columns]
        show = fs.copy().astype(object)
        for k in show.index:
            show.loc[k] = [ratio(v) if k == "희석 EPS" else money.big(v) for v in fs.loc[k]]
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
                                 hovertemplate="%{x} 매출 " + money.hv() + "<extra></extra>"))
            if "순이익" in fs.index:
                fig.add_trace(go.Bar(x=fs.columns, y=fs.loc["순이익"], name="순이익", marker_color="#F2A93B",
                                     hovertemplate="%{x} 순이익 " + money.hv() + "<extra></extra>"))
            if "영업이익" in fs.index:
                fig.add_trace(go.Scatter(x=fs.columns, y=(fs.loc["영업이익"] / fs.loc["매출"] * 100), name="영업이익률(%)",
                                         yaxis="y2", mode="lines+markers", line=dict(color="#E5534B"),
                                         hovertemplate="%{x} 영업이익률 %{y:.1f}%<extra></extra>"))
            fig.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                              yaxis=dict(title="금액"), yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("재무제표를 불러오지 못했습니다 (ETF이거나 데이터 제공처 응답 지연).")

    # ---- 5-1. 재무비율·주가·PER 그래프 (직접 선택)
    exp, op = ui.more("재무비율 · 주가 · 밸류에이션 그래프 더보기 (직접 골라서 겹쳐 보기)", key=f"rs_more_combo_{ticker}")
    if op:
        with exp:
            _combo_chart(sec, R, ticker, px, n_eps, src)


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

    st.caption("출처: Yahoo Finance. 지연·누락이 있을 수 있으며 투자 판단의 참고 자료로만 사용하세요.")


def _render_yahoo_news(R: dict, ko: bool) -> None:
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
        kos = None
        if ko:
            joined = translate_ko("\n".join(r[0].replace("\n", " ") for r in rows))
            kos = joined.split("\n") if joined else None
            if kos is not None and len(kos) != len(rows):
                kos = None
        lines = []
        for i, (title, url, prov, when) in enumerate(rows):
            main = kos[i].strip() if kos and kos[i].strip() else title
            t = html.escape(main)
            link = f'<a href="{html.escape(url, quote=True)}" target="_blank">{t}</a>' if url else t
            orig = f'<br><span style="font-size:0.8rem;opacity:.55">{html.escape(title)}</span>' if main != title else ""
            lines.append(f'<div style="margin:0 0 10px">{link}{orig}<br><span style="font-size:0.8rem;opacity:.65">'
                         f'{html.escape(prov or "")} · {when}</span></div>')
        st.markdown("".join(lines).replace("$", "&#36;"), unsafe_allow_html=True)
