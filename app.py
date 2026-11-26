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
if getattr(autoreload, "ERRORS", None):
    st.warning("새로 올린 코드 일부를 적용하지 못해 이전 코드로 실행 중입니다 — "
               + " · ".join(f"{k}.py: {v}" for k, v in autoreload.ERRORS.items()))

import ui  # 화면 공통 요소 (ui.py)

ui.apply_theme()

_boot = st.empty()  # 종목 목록을 처음 받을 때(20~30초) 빈 화면 대신 보여줄 안내
if "_listing" not in st.session_state:
    _boot.info("데이터를 불러오는 중입니다. 처음 접속하면 20~30초 걸릴 수 있어요. 두 번째부터는 빨라집니다.")

# 파일 구성은 ARCHITECTURE.md (파트별 파일 지도 · 공통 도우미 · 세션 키 접두어 · 고칠 때 순서)
#   페이지 진입점: market.py 시장 · page_single.py 주가 조회 · research.py 기업 리서치 · screener.py 스크리너
#                  page_compare.py 수익률 비교 · page_portfolio.py 포트폴리오 · backtest.py · options.py 옵션 · bonds.py 채권
from common import find_candidates, load_listing, load_prices, risk_free_for

# ======================================================================
try:
    LISTING = load_listing()
    st.session_state["_listing"] = LISTING
except Exception as e:  # noqa: BLE001
    _boot.empty()
    st.error("종목 목록을 불러오지 못했습니다. 잠시 후 새로고침해 주세요.")
    st.stop()
_boot.empty()



def page_market():
    with ui.url_page("market"):
        from market import render_market

        render_market()


def page_single():
    with ui.url_page("chart"):
        from page_single import render_single

        render_single(LISTING)


def page_screener():
    with ui.url_page("screener"):
        from screener import render_screener

        render_screener()


def page_compare():
    with ui.url_page("compare"):
        from page_compare import render_compare

        render_compare(LISTING)


def page_portfolio():
    with ui.url_page("portfolio"):
        from page_portfolio import render_portfolio

        render_portfolio(LISTING)


def page_backtest():
    with ui.url_page("backtest"):
        from backtest import render_backtest

        render_backtest(load_prices, find_candidates, LISTING, risk_free_for)


def page_options():
    from options import render_options

    render_options()


def page_bonds():
    from bonds import render_bonds

    render_bonds()


def page_research():
    with ui.url_page("research"):
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


def _key_box() -> None:
    """오른쪽 위 '키 설정': OpenDART 키를 붙여 넣는 칸. 이 브라우저 탭에서만 쓰이고 저장되지 않는다 (새로고침하면 다시 입력)."""
    try:
        _l, _r = st.columns([6, 1])
        with _r.popover("키 설정", icon=":material/key:", use_container_width=True):
            st.text_input("OpenDART API 키", key="dart_key_global", type="password",
                          help="한국 지도의 시·군·구 위치, 재무제표 등에 쓰입니다. opendart.fss.or.kr에서 무료로 발급받을 수 있습니다.")
            st.caption("이 탭에서만 쓰이고 저장되지 않습니다. 새로고침하면 다시 넣어야 합니다. "
                       "계속 쓰려면 Streamlit 앱 설정의 Secrets에 DART_API_KEY로 넣으세요.")
            import kr

            st.caption("지금 키가 적용되어 있습니다." if kr.dart_key() else "키가 아직 없습니다.")
    except Exception:  # noqa: BLE001  (오래된 Streamlit에서는 칸만 생략)
        pass


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
        _key_box()
        nav.run()
        return
    labels = [t for _, t, _, _ in PAGES]  # 아주 오래된 Streamlit
    ui.keep_state()
    pick = st.radio("메뉴", labels, horizontal=True, key="nav_fallback", label_visibility="collapsed")
    PAGES[labels.index(pick)][0]()


_run_pages()
ui.snapshot_state()
st.caption("일봉 기준 데이터이며 실시간 시세가 아닙니다. 장중에는 지연되거나 당일 값이 반영되지 않을 수 있습니다.")
st.caption("이 앱의 점수·지표는 과거 데이터로 계산한 참고 정보이며 특정 종목의 매수·매도를 권유하지 않습니다. 계산 방식과 기준일은 각 화면에 적혀 있고, 과거 성과가 미래를 보장하지 않습니다.")
