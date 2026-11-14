"""화면 공통 요소: 테마, 페이지를 오가도 입력값 유지, 지연 로딩 탭·더보기, 시세 헤더, 핵심 지표 카드.

- 지연 로딩: Streamlit 1.55+의 st.tabs / st.expander(on_change="rerun")로 열린 탭·펼친 더보기만 계산한다.
  (예전 버전이면 자동으로 일반 탭·펼침으로 돌아가 모두 계산한다)
- 입력값 유지: 다른 페이지로 갔다 오면 Streamlit이 화면에 없던 위젯 값을 지우므로, 실행이 끝날 때 값을 기억해 두었다가 되살린다.
"""
from __future__ import annotations

import html
import inspect
from datetime import date

import streamlit as st

UP, DOWN, FLAT = "#EF5350", "#4D8DF7", "#8C877C"  # 국내 관행: 상승 빨강, 하락 파랑


def _has_param(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


LAZY_TABS = _has_param(st.tabs, "on_change")
LAZY_EXPANDER = _has_param(st.expander, "on_change")

THEME_CSS = """
<style>
/* 디자인: 검정 바탕 + 금색. 제목·현재가는 명조(세리프), 숫자는 고정폭 숫자.
   테마 색은 .streamlit/config.toml, 여기서는 글꼴·카드·표면만 다룬다. */
@import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css");
@import url("https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@600;700;900&display=swap");
:root { --paper:#0C0C0D; --card:#141413; --ink:#ECE6D8; --ink2:#9A9488; --rule:#2E2A22; --accent:#C9A44C; --gold:#D9B85E;
        --up:#EF5350; --down:#4D8DF7; --serif:"Noto Serif KR", "Nanum Myeongjo", serif;
        --sans:"Pretendard Variable", Pretendard, -apple-system, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif; }
html, body, .stApp, .stMarkdown, label, input, textarea, button,
[data-baseweb="select"], [data-testid="stCaptionContainer"] { font-family: var(--sans); font-size: 0.88rem; }
.stApp { color: var(--ink); }
.block-container { padding-top: 4.4rem !important; padding-bottom: 3rem !important; max-width: 1320px; }
[data-testid="stHeader"] { background: rgba(12,12,13,.92); backdrop-filter: blur(8px); border-bottom: 1px solid var(--rule); }
/* 제목: 명조 — 리포트의 목소리 */
h1, h2, h3, h4, h5, h6 { font-family: var(--serif) !important; color: var(--ink) !important; letter-spacing: -0.01em; }
h1 { font-size: 1.5rem !important; font-weight: 900 !important; color: var(--gold) !important; padding: 0.1rem 0 0.4rem 0 !important; }
h2, h3 { font-size: 1.12rem !important; font-weight: 700 !important; padding: 0.2rem 0 !important; }
h4, h5, h6 { font-size: 1rem !important; font-weight: 700 !important; padding: 1.1rem 0 0.35rem 0 !important;
             border-bottom: 1px solid var(--rule); margin-bottom: 0.5rem !important; }
[data-testid="stCaptionContainer"] { color: var(--ink2); opacity: 1; }
/* 탭 */
[data-baseweb="tab-list"] { gap: 0.2rem; border-bottom: 1px solid var(--rule); }
button[data-baseweb="tab"] { padding: 0.5rem 0.75rem !important; }
button[data-baseweb="tab"] p { font-size: 0.9rem !important; font-weight: 500; color: var(--ink2); }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--ink); font-weight: 700; }
/* 더보기 상자 */
[data-testid="stExpander"] details { border-radius: 8px; border: 1px solid var(--rule); background: var(--card); }
[data-testid="stExpander"] summary p { font-weight: 600; font-size: 0.88rem; }
/* 입력칸·버튼: 흰 바탕 */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] { background: var(--card) !important; }
.stButton > button, [data-testid="stPopover"] button, [data-testid="stDownloadButton"] button {
  background: var(--card); border: 1px solid var(--rule); font-weight: 600; }
.stButton > button:hover, [data-testid="stPopover"] button:hover { border-color: var(--accent); color: var(--accent); }
[data-testid="stBaseButton-primary"] { background: var(--accent) !important; border-color: var(--accent) !important; color: #0C0C0D !important; }
[data-baseweb="tag"] { background: rgba(201,164,76,.18) !important; border: 1px solid rgba(201,164,76,.55); }
[data-baseweb="tag"] span, [data-baseweb="tag"] svg { color: var(--gold) !important; fill: var(--gold) !important; }
:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 1px; }
/* 그래프·표: 흰 지면 위에 */
[data-testid="stPlotlyChart"] { background: var(--card); border: 1px solid var(--rule); border-radius: 8px; padding: 6px 4px 2px; }
[data-testid="stDataFrame"] { border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; background: var(--card); }
[data-testid="stAlert"] { border-radius: 8px; }
.qt:hover, .qt:focus { color: var(--accent) !important; border-color: var(--accent) !important; }
/* 지표 카드: 그림자 없이 흰 칸 + 가는 테두리 */
.kc { display: grid; gap: 8px; margin: 4px 0 14px; }
.kc-c { min-width: 0; padding: 10px 13px 9px; background: var(--card); border: 1px solid var(--rule); border-radius: 8px; }
.kc-l { font-size: 0.74rem; line-height: 1.2; white-space: nowrap; color: var(--ink2); }
.kc-v { font-size: 1.12rem; font-weight: 650; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        font-variant-numeric: tabular-nums; color: var(--ink); letter-spacing: -0.01em; }
.kc-s { font-size: 0.71rem; color: var(--ink2); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kc-d { font-size: 0.74rem; font-weight: 600; margin-top: 1px; }
.up { color: var(--up); } .down { color: var(--down); } .flat { color: #8C877C; }
/* 시세 헤더: 종목명·현재가는 명조로 크게 — 이 화면의 얼굴 */
.qh { margin: 4px 0 14px; padding-bottom: 12px; border-bottom: 1px solid var(--accent); }
.qh-n { font-family: var(--serif); font-size: 1.45rem; font-weight: 900; line-height: 1.3; color: var(--ink); letter-spacing: -0.02em; }
.qh-n .qh-code { font-family: var(--sans); font-size: 0.85rem; font-weight: 500; color: var(--ink2); margin-left: 8px; letter-spacing: 0; }
.qh-m { font-size: 0.78rem; color: var(--ink2); margin-top: 2px; }
.qh-p { display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 14px; margin-top: 6px; }
.qh-p .px { font-family: var(--serif); font-size: 2.4rem; font-weight: 900; letter-spacing: -0.03em; font-variant-numeric: tabular-nums; color: var(--gold); line-height: 1.1; }
.qh-p .ch { font-size: 1.02rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.qh-p .asof { font-size: 0.75rem; color: var(--ink2); }
/* 시장 타일 */
.mt { display: grid; gap: 10px; margin: 6px 0 18px; }
.mt-c { padding: 11px 13px 7px; border-radius: 8px; background: var(--card); border: 1px solid var(--rule); min-width: 0; }
.mt-n { font-size: 0.76rem; color: var(--ink2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.mt-v { font-size: 1.2rem; font-weight: 700; font-variant-numeric: tabular-nums; margin-top: 2px; color: var(--ink); letter-spacing: -0.01em; }
.mt-d { font-size: 0.78rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.mt-c svg { display: block; width: 100%; height: 34px; margin-top: 4px; }
.sec-t { font-family: var(--serif); font-size: 1rem; font-weight: 700; margin: 18px 0 6px; padding-bottom: 5px; border-bottom: 1px solid var(--rule); color: var(--ink); }
.sec-s { font-family: var(--sans); font-size: 0.75rem; color: var(--ink2); font-weight: 400; margin-left: 8px; }
@media (max-width: 640px) {
  .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
  .kc, .mt { grid-template-columns: repeat(2, minmax(0,1fr)) !important; }
  .qh-p .px { font-size: 1.9rem; } .qh-n { font-size: 1.2rem; }
}
</style>
"""


def apply_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 페이지를 오가도 입력값 유지
# ----------------------------------------------------------------------
_BOOL_KEYS = {"bt_bh", "bt_log", "hedge_round", "rs_ko"}
_BOOL_PREFIX = ("rs_daily_mean_", "ins_ps_")
_BOOL_SUFFIX = ("_rate_on", "_wman")


def _keepable(k, v) -> bool:
    if not isinstance(k, str) or k.startswith("_") or k.startswith("FormSubmitter"):
        return False
    if isinstance(v, bool):  # 버튼 값은 되살리면 오류가 나므로 스위치(토글)만
        return k in _BOOL_KEYS or k.startswith(_BOOL_PREFIX) or k.endswith(_BOOL_SUFFIX)
    if v is None:
        return False
    return isinstance(v, (str, int, float, date, list, tuple))


def keep_state(page: str | None = None) -> None:
    """실행을 시작할 때: 지난 실행에서 기억한 입력값 중 사라진 것을 되살린다.
    page가 바뀌었으면(다른 메뉴에서 돌아왔으면) 남아 있는 입력값도 다시 저장해 입력창에 값이 그대로 보이게 한다."""
    try:  # 되살린 값과 위젯 기본값이 겹칠 때 뜨는 노란 경고를 끈다 (값은 되살린 쪽이 쓰임)
        from streamlit.elements.lib import policies

        policies._shown_default_value_warning = True
    except Exception:  # noqa: BLE001
        pass
    snap = st.session_state.get("_ui_snap") or {}
    moved = page is not None and st.session_state.get("_ui_page") != page
    if page is not None:
        st.session_state["_ui_page"] = page
    for k, v in snap.items():
        if k not in st.session_state or moved:
            cur = st.session_state[k] if k in st.session_state else v
            if not _keepable(k, cur):
                continue
            try:
                st.session_state[k] = cur
            except Exception:  # noqa: BLE001
                pass


def snapshot_state() -> None:
    """실행이 끝날 때: 지금 입력값을 기억한다."""
    snap = {}
    for k in list(st.session_state.keys()):
        try:
            v = st.session_state[k]
        except Exception:  # noqa: BLE001
            continue
        if _keepable(k, v):
            snap[k] = v
    st.session_state["_ui_snap"] = snap


# ----------------------------------------------------------------------
# 지연 로딩 탭 · 더보기
# ----------------------------------------------------------------------
def lazy_tabs(labels: list, key: str) -> list:
    """[(탭, 열림 여부), ...]. 열린 탭만 계산하면 된다."""
    if LAZY_TABS:
        try:
            tabs = st.tabs(labels, key=key, on_change="rerun")
            opens = [bool(getattr(t, "open", None)) for t in tabs]
            if not any(opens):
                opens[0] = True
            return list(zip(tabs, opens))
        except Exception:  # noqa: BLE001
            pass
    return [(t, True) for t in st.tabs(labels)]


def more(label: str = "더보기", key: str | None = None, expanded: bool = False):
    """(접힘 상자, 펼쳐짐 여부). 펼쳤을 때만 안을 계산한다."""
    if LAZY_EXPANDER and key:
        try:
            e = st.expander(label, expanded=expanded, key=key, on_change="rerun")
            op = getattr(e, "open", None)
            return e, (expanded if op is None else bool(op))
        except Exception:  # noqa: BLE001
            pass
    return st.expander(label, expanded=expanded), True


# ----------------------------------------------------------------------
# 시세 헤더 · 지표 카드
# ----------------------------------------------------------------------
def _cls(x) -> str:
    if x is None or x != x or abs(x) < 1e-12:
        return "flat"
    return "up" if x > 0 else "down"


def change_html(chg_text: str | None, pct: float | None) -> str:
    """'▲ 1,200 (+1.53%)' 형태 (상승 빨강·하락 파랑)."""
    if pct is None or pct != pct:
        return ""
    arrow = "▲" if pct > 0 else ("▼" if pct < 0 else "")
    body = f"{arrow} {chg_text} ({pct:+.2f}%)" if chg_text else f"{arrow} {pct:+.2f}%"
    return f'<span class="ch {_cls(pct)}">{html.escape(body.strip())}</span>'


def quote_header(name: str, code: str = "", meta: str = "", price: str = "", chg_text: str | None = None,
                 pct: float | None = None, asof: str = "") -> None:
    """종목명·현재가·전일 대비를 MTS처럼 크게 보여준다."""
    meta_html = f'<div class="qh-m">{meta}</div>' if meta else ""  # meta는 링크 등 HTML 허용
    out = (f'<div class="qh"><div class="qh-n">{html.escape(name)}<span class="qh-code">{html.escape(code)}</span></div>{meta_html}'
           f'<div class="qh-p"><span class="px">{html.escape(price)}</span>{change_html(chg_text, pct)}'
           f'<span class="asof">{html.escape(asof)}</span></div></div>')
    st.markdown(out.replace("$", "&#36;"), unsafe_allow_html=True)


def cards(items: list, cols: int = 4) -> None:
    """지표 카드 격자. items: (라벨, 값[, 보조 문구[, 등락값(색)]]). 라벨 옆 ?에 설명."""
    from glossary import CSS, icon, tip

    cells = ""
    for i, it in enumerate(items):
        lab, val = it[0], it[1]
        sub = it[2] if len(it) > 2 and it[2] else ""
        sign = it[3] if len(it) > 3 else None
        right = " qr" if cols > 1 and (i % cols) >= cols / 2 else ""
        vcls = f" {_cls(sign)}" if sign is not None else ""
        cells += (f'<div class="kc-c{right}"><div class="kc-l"><span>{html.escape(lab)}</span>{icon(tip(lab))}</div>'
                  f'<div class="kc-v{vcls}">{html.escape(str(val))}</div>'
                  + (f'<div class="kc-s">{html.escape(str(sub))}</div>' if sub else "") + "</div>")
    st.markdown((CSS + f'<div class="kc" style="grid-template-columns:repeat({cols},minmax(0,1fr))">{cells}</div>').replace("$", "&#36;"),
                unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    s = f'<span class="sec-s">{html.escape(sub)}</span>' if sub else ""
    st.markdown(f'<div class="sec-t">{html.escape(title)}{s}</div>', unsafe_allow_html=True)


def sparkline(values, color: str) -> str:
    """작은 추세선 SVG (차트 라이브러리 없이 가볍게)."""
    v = [float(x) for x in values if x == x]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    rng = hi - lo or 1.0
    n = len(v)
    pts = " ".join(f"{i * 100 / (n - 1):.2f},{30 - (x - lo) / rng * 28 + 1:.2f}" for i, x in enumerate(v))
    return (f'<svg viewBox="0 0 100 32" preserveAspectRatio="none"><polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>')


def popover(label: str, icon: str | None = None, help: str | None = None):
    """작은 설정 창 (예전 버전이면 접힘 상자)."""
    if hasattr(st, "popover"):
        for kw in ({"icon": icon, "help": help, "width": "stretch"}, {"icon": icon, "help": help, "use_container_width": True},
                   {"use_container_width": True}, {}):
            try:
                return st.popover(label, **kw)
            except TypeError:
                continue
    return st.expander(label)
"""화면 공통 요소: 테마, 페이지를 오가도 입력값 유지, 지연 로딩 탭·더보기, 시세 헤더, 핵심 지표 카드.

- 지연 로딩: Streamlit 1.55+의 st.tabs / st.expander(on_change="rerun")로 열린 탭·펼친 더보기만 계산한다.
  (예전 버전이면 자동으로 일반 탭·펼침으로 돌아가 모두 계산한다)
- 입력값 유지: 다른 페이지로 갔다 오면 Streamlit이 화면에 없던 위젯 값을 지우므로, 실행이 끝날 때 값을 기억해 두었다가 되살린다.
"""
from __future__ import annotations

import html
import inspect
from datetime import date

import streamlit as st

UP, DOWN, FLAT = "#E5484D", "#3B82F6", "#8A8F98"  # 국내 관행: 상승 빨강, 하락 파랑


def _has_param(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


LAZY_TABS = _has_param(st.tabs, "on_change")
LAZY_EXPANDER = _has_param(st.expander, "on_change")

THEME_CSS = """
<style>
html, body, .stApp, .stMarkdown, label, input, textarea,
[data-baseweb="select"], [data-testid="stCaptionContainer"] { font-size: 0.86rem; }
.block-container { padding-top: 4.2rem !important; padding-bottom: 3rem !important; max-width: 1380px; }
h1 { font-size: 1.35rem !important; padding: 0.1rem 0 0.3rem 0 !important; }
h2, h3 { font-size: 1.05rem !important; padding: 0.15rem 0 !important; }
h4, h5, h6 { font-size: 0.95rem !important; padding: 0.6rem 0 0.15rem 0 !important; }
[data-testid="stCaptionContainer"] { opacity: 0.85; }
/* 탭: 글자 크게, 밑줄만 */
button[data-baseweb="tab"] p { font-size: 0.9rem !important; font-weight: 500; }
[data-baseweb="tab-list"] { gap: 0.4rem; }
/* 더보기(접힘 상자): 테두리만 얇게 */
[data-testid="stExpander"] details { border-radius: 10px; border: 1px solid rgba(128,128,128,.22); }
[data-testid="stExpander"] summary p { font-weight: 600; font-size: 0.88rem; }
/* 지표 카드 */
.kc { display: grid; gap: 8px; margin: 4px 0 14px; }
.kc-c { min-width: 0; padding: 9px 12px 8px; border-radius: 10px;
        background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.14); }
.kc-l { font-size: 0.74rem; line-height: 1.2; white-space: nowrap; }
.kc-l > span:first-child { opacity: .68; }
.kc-v { font-size: 1.08rem; font-weight: 650; margin-top: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        font-variant-numeric: tabular-nums; }
.kc-s { font-size: 0.7rem; opacity: .6; margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kc-d { font-size: 0.74rem; font-weight: 600; margin-top: 1px; }
.up { color: #E5484D; } .down { color: #3B82F6; } .flat { color: #8A8F98; }
/* 시세 헤더 */
.qh { margin: 2px 0 10px; }
.qh-n { font-size: 1.15rem; font-weight: 700; line-height: 1.3; }
.qh-n .qh-code { font-size: 0.85rem; font-weight: 500; opacity: .6; margin-left: 6px; }
.qh-m { font-size: 0.78rem; opacity: .65; margin-top: 1px; }
.qh-p { display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 12px; margin-top: 4px; }
.qh-p .px { font-size: 1.9rem; font-weight: 750; letter-spacing: -0.5px; font-variant-numeric: tabular-nums; }
.qh-p .ch { font-size: 1rem; font-weight: 650; font-variant-numeric: tabular-nums; }
.qh-p .asof { font-size: 0.75rem; opacity: .6; }
/* 시장 타일 */
.mt { display: grid; gap: 10px; margin: 6px 0 16px; }
.mt-c { padding: 10px 12px 6px; border-radius: 12px; background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.14); min-width: 0; }
.mt-n { font-size: 0.76rem; opacity: .7; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.mt-v { font-size: 1.15rem; font-weight: 700; font-variant-numeric: tabular-nums; margin-top: 2px; }
.mt-d { font-size: 0.78rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.mt-c svg { display: block; width: 100%; height: 34px; margin-top: 4px; }
@media (max-width: 640px) { .kc, .mt { grid-template-columns: repeat(2, minmax(0,1fr)) !important; } .qh-p .px { font-size: 1.5rem; } }
.sec-t { font-size: 0.95rem; font-weight: 700; margin: 14px 0 4px; }
.sec-s { font-size: 0.75rem; opacity: .6; font-weight: 400; margin-left: 6px; }
</style>
"""


def apply_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 페이지를 오가도 입력값 유지
# ----------------------------------------------------------------------
_BOOL_KEYS = {"bt_bh", "bt_log", "hedge_round", "rs_ko"}
_BOOL_PREFIX = ("rs_daily_mean_", "ins_ps_")
_BOOL_SUFFIX = ("_rate_on", "_wman")


def _keepable(k, v) -> bool:
    if not isinstance(k, str) or k.startswith("_") or k.startswith("FormSubmitter"):
        return False
    if isinstance(v, bool):  # 버튼 값은 되살리면 오류가 나므로 스위치(토글)만
        return k in _BOOL_KEYS or k.startswith(_BOOL_PREFIX) or k.endswith(_BOOL_SUFFIX)
    if v is None:
        return False
    return isinstance(v, (str, int, float, date, list, tuple))


def keep_state(page: str | None = None) -> None:
    """실행을 시작할 때: 지난 실행에서 기억한 입력값 중 사라진 것을 되살린다.
    page가 바뀌었으면(다른 메뉴에서 돌아왔으면) 남아 있는 입력값도 다시 저장해 입력창에 값이 그대로 보이게 한다."""
    try:  # 되살린 값과 위젯 기본값이 겹칠 때 뜨는 노란 경고를 끈다 (값은 되살린 쪽이 쓰임)
        from streamlit.elements.lib import policies

        policies._shown_default_value_warning = True
    except Exception:  # noqa: BLE001
        pass
    snap = st.session_state.get("_ui_snap") or {}
    moved = page is not None and st.session_state.get("_ui_page") != page
    if page is not None:
        st.session_state["_ui_page"] = page
    for k, v in snap.items():
        if k not in st.session_state or moved:
            cur = st.session_state[k] if k in st.session_state else v
            if not _keepable(k, cur):
                continue
            try:
                st.session_state[k] = cur
            except Exception:  # noqa: BLE001
                pass


def snapshot_state() -> None:
    """실행이 끝날 때: 지금 입력값을 기억한다."""
    snap = {}
    for k in list(st.session_state.keys()):
        try:
            v = st.session_state[k]
        except Exception:  # noqa: BLE001
            continue
        if _keepable(k, v):
            snap[k] = v
    st.session_state["_ui_snap"] = snap


# ----------------------------------------------------------------------
# 지연 로딩 탭 · 더보기
# ----------------------------------------------------------------------
def lazy_tabs(labels: list, key: str) -> list:
    """[(탭, 열림 여부), ...]. 열린 탭만 계산하면 된다."""
    if LAZY_TABS:
        try:
            tabs = st.tabs(labels, key=key, on_change="rerun")
            opens = [bool(getattr(t, "open", None)) for t in tabs]
            if not any(opens):
                opens[0] = True
            return list(zip(tabs, opens))
        except Exception:  # noqa: BLE001
            pass
    return [(t, True) for t in st.tabs(labels)]


def more(label: str = "더보기", key: str | None = None, expanded: bool = False):
    """(접힘 상자, 펼쳐짐 여부). 펼쳤을 때만 안을 계산한다."""
    if LAZY_EXPANDER and key:
        try:
            e = st.expander(label, expanded=expanded, key=key, on_change="rerun")
            op = getattr(e, "open", None)
            return e, (expanded if op is None else bool(op))
        except Exception:  # noqa: BLE001
            pass
    return st.expander(label, expanded=expanded), True


# ----------------------------------------------------------------------
# 시세 헤더 · 지표 카드
# ----------------------------------------------------------------------
def _cls(x) -> str:
    if x is None or x != x or abs(x) < 1e-12:
        return "flat"
    return "up" if x > 0 else "down"


def change_html(chg_text: str | None, pct: float | None) -> str:
    """'▲ 1,200 (+1.53%)' 형태 (상승 빨강·하락 파랑)."""
    if pct is None or pct != pct:
        return ""
    arrow = "▲" if pct > 0 else ("▼" if pct < 0 else "")
    body = f"{arrow} {chg_text} ({pct:+.2f}%)" if chg_text else f"{arrow} {pct:+.2f}%"
    return f'<span class="ch {_cls(pct)}">{html.escape(body.strip())}</span>'


def quote_header(name: str, code: str = "", meta: str = "", price: str = "", chg_text: str | None = None,
                 pct: float | None = None, asof: str = "") -> None:
    """종목명·현재가·전일 대비를 MTS처럼 크게 보여준다."""
    meta_html = f'<div class="qh-m">{meta}</div>' if meta else ""  # meta는 링크 등 HTML 허용
    out = (f'<div class="qh"><div class="qh-n">{html.escape(name)}<span class="qh-code">{html.escape(code)}</span></div>{meta_html}'
           f'<div class="qh-p"><span class="px">{html.escape(price)}</span>{change_html(chg_text, pct)}'
           f'<span class="asof">{html.escape(asof)}</span></div></div>')
    st.markdown(out.replace("$", "&#36;"), unsafe_allow_html=True)


def cards(items: list, cols: int = 4) -> None:
    """지표 카드 격자. items: (라벨, 값[, 보조 문구[, 등락값(색)]]). 라벨 옆 ?에 설명."""
    from glossary import CSS, icon, tip

    cells = ""
    for i, it in enumerate(items):
        lab, val = it[0], it[1]
        sub = it[2] if len(it) > 2 and it[2] else ""
        sign = it[3] if len(it) > 3 else None
        right = " qr" if cols > 1 and (i % cols) >= cols / 2 else ""
        vcls = f" {_cls(sign)}" if sign is not None else ""
        cells += (f'<div class="kc-c{right}"><div class="kc-l"><span>{html.escape(lab)}</span>{icon(tip(lab))}</div>'
                  f'<div class="kc-v{vcls}">{html.escape(str(val))}</div>'
                  + (f'<div class="kc-s">{html.escape(str(sub))}</div>' if sub else "") + "</div>")
    st.markdown((CSS + f'<div class="kc" style="grid-template-columns:repeat({cols},minmax(0,1fr))">{cells}</div>').replace("$", "&#36;"),
                unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    s = f'<span class="sec-s">{html.escape(sub)}</span>' if sub else ""
    st.markdown(f'<div class="sec-t">{html.escape(title)}{s}</div>', unsafe_allow_html=True)


def sparkline(values, color: str) -> str:
    """작은 추세선 SVG (차트 라이브러리 없이 가볍게)."""
    v = [float(x) for x in values if x == x]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    rng = hi - lo or 1.0
    n = len(v)
    pts = " ".join(f"{i * 100 / (n - 1):.2f},{30 - (x - lo) / rng * 28 + 1:.2f}" for i, x in enumerate(v))
    return (f'<svg viewBox="0 0 100 32" preserveAspectRatio="none"><polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>')


def popover(label: str, icon: str | None = None, help: str | None = None):
    """작은 설정 창 (예전 버전이면 접힘 상자)."""
    if hasattr(st, "popover"):
        for kw in ({"icon": icon, "help": help, "width": "stretch"}, {"icon": icon, "help": help, "use_container_width": True},
                   {"use_container_width": True}, {}):
            try:
                return st.popover(label, **kw)
            except TypeError:
                continue
    return st.expander(label)
