"""주가 차트 · 시장 분석 앱.

상단 메뉴: 시장(첫 화면) · 주가 조회 · 기업 리서치 · 스크리너 · 수익률 비교 · 포트폴리오 · 백테스트 · 옵션 · 채권.
열린 페이지와 열린 탭만 계산해서 빠르게 보여주고, 자주 보는 지표를 먼저, 세부 지표는 '더보기'에 접어 둔다.
수익률 비교와 포트폴리오의 종목·기간 입력은 서로 동기화된다.

데이터: FinanceDataReader (국내 KRX 전 종목 + 해외 티커 + 해외 선물), Yahoo Finance. 일봉 기준이며 실시간 틱 데이터는 아닙니다.
"""
import streamlit as st

import autoreload  # GitHub에 올린 새 코드를 새로고침만으로 반영 (autoreload.py)

autoreload.refresh()

st.set_page_config(page_title="주가 차트", page_icon="📈", layout="wide")

import ui  # 화면 공통 요소 (ui.py)

ui.apply_theme()

# 파일 구성 (하나 고칠 때 그 파일만 올리면 되도록 나눔)
#   common.py          종목 목록·주가 불러오기·기간 입력·CAPM/위험 지표
#   page_single.py     주가 조회        page_compare.py   수익률 비교      page_portfolio.py  포트폴리오
#   market.py 시장 · research.py 기업 리서치 · screener.py · backtest.py · options.py · bonds.py
from common import find_candidates, load_listing, load_prices, risk_free_for

# ======================================================================
try:
    LISTING = load_listing()
    st.session_state["_listing"] = LISTING
except Exception as e:  # noqa: BLE001
    st.error(f"종목 목록을 불러오지 못했습니다: {e}")
    st.stop()



def page_market():
    from market import render_market

    render_market()


def page_single():
    from page_single import render_single

    render_single(LISTING)


def page_screener():
    from screener import render_screener

    render_screener()


def page_compare():
    from page_compare import render_compare

    render_compare(LISTING)


def page_portfolio():
    from page_portfolio import render_portfolio

    render_portfolio(LISTING)


def page_backtest():
    from backtest import render_backtest

    render_backtest(load_prices, find_candidates, LISTING, risk_free_for)


def page_options():
    from options import render_options

    render_options()


def page_bonds():
    from bonds import render_bonds

    render_bonds()


def page_research():
    from research import render_research

    render_research()


PAGES = [
    (page_market, "시장", ":material/monitoring:", "market"),
    (page_single, "주가 조회", ":material/candlestick_chart:", "chart"),
    (page_research, "기업 리서치", ":material/business_center:", "research"),
    (page_screener, "스크리너", ":material/filter_list:", "screener"),
    (page_compare, "수익률 비교", ":material/compare_arrows:", "compare"),
    (page_portfolio, "포트폴리오", ":material/pie_chart:", "portfolio"),
    (page_backtest, "백테스트", ":material/history:", "backtest"),
    (page_options, "옵션", ":material/stacked_line_chart:", "options"),
    (page_bonds, "채권", ":material/account_balance:", "bonds"),
]


def _run_pages() -> None:
    """열린 페이지 하나만 실행한다 (예전엔 7개 탭을 매번 모두 계산해서 느렸음)."""
    if hasattr(st, "navigation") and hasattr(st, "Page"):
        pages = [st.Page(fn, title=t, icon=ic, url_path=u, default=(i == 0)) for i, (fn, t, ic, u) in enumerate(PAGES)]
        ui.PAGE_OBJS = {u: pg for (_, _, _, u), pg in zip(PAGES, pages)}  # 스크리너에서 다른 페이지로 넘어갈 때 씀
        try:
            nav = st.navigation(pages, position="top")
        except TypeError:
            nav = st.navigation(pages)
        ui.keep_state(getattr(nav, "url_path", None) or getattr(nav, "title", None))  # 다른 페이지에 갔다 와도 입력값 유지
        nav.run()
        return
    labels = [t for _, t, _, _ in PAGES]  # 아주 오래된 Streamlit
    ui.keep_state()
    pick = st.radio("메뉴", labels, horizontal=True, key="nav_fallback", label_visibility="collapsed")
    PAGES[labels.index(pick)][0]()


_run_pages()
ui.snapshot_state()
st.caption("일봉 기준 데이터이며 실시간 시세가 아닙니다. 장중에는 지연되거나 당일 값이 반영되지 않을 수 있습니다.")
