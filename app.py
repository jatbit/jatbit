"""주가 차트 조회 프로그램.

탭 1) 주가 조회: 종목 하나의 기간별 주가 차트
탭 2) 수익률 비교: 여러 종목을 기간 시작일 = 0% 기준으로 겹쳐서 비교

데이터: FinanceDataReader (국내 KRX 전 종목 + 해외 티커). 일봉 기준이며 실시간 틱 데이터는 아닙니다.
"""

import re
from datetime import date, timedelta

import FinanceDataReader as fdr
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

LINE_COLOR = "#2B6CB0"
UP_COLOR = "#D64545"      # 국내 관행: 상승 = 빨강
DOWN_COLOR = "#2B6CB0"    # 하락 = 파랑
VOLUME_COLOR = "#A0AEC0"
SERIES_COLORS = ["#2B6CB0", "#D64545", "#2F855A", "#B7791F", "#6B46C1", "#0987A0", "#DD6B20", "#4A5568"]
MAX_COMPARE = 8

st.set_page_config(page_title="주가 차트", layout="wide")

# 글자 크기 축소
st.markdown(
    """
    <style>
    html, body, .stApp, .stMarkdown, label, input, textarea,
    [data-baseweb="select"], [data-testid="stCaptionContainer"] { font-size: 0.85rem !important; }
    h1 { font-size: 1.4rem !important; padding: 0.2rem 0 0.4rem 0 !important; }
    h2, h3 { font-size: 1.05rem !important; padding: 0.2rem 0 !important; }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    [data-testid="stMetricLabel"] p { font-size: 0.72rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.72rem !important; }
    button[data-baseweb="tab"] p { font-size: 0.9rem !important; }
    .block-container { padding-top: 2.5rem !important; }
    .mg { display: grid; gap: 0.6rem 0.8rem; margin: 0.4rem 0 0.8rem 0; }
    .mg-cell { min-width: 0; }
    .mg-label { font-size: 0.72rem; opacity: 0.75; }
    .mg-value { font-size: clamp(0.85rem, 3.8vw, 1.1rem); font-weight: 500; word-break: keep-all; }
    .mg-delta { font-size: 0.72rem; }
    .mg-delta.up { color: #E5534B; }
    .mg-delta.down { color: #4C8DF6; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("주가 차트")


@st.cache_data(ttl=60 * 60 * 24, show_spinner="종목 목록을 불러오는 중...")
def load_listing() -> pd.DataFrame:
    df = fdr.StockListing("KRX")
    if "Symbol" in df.columns and "Code" not in df.columns:
        df = df.rename(columns={"Symbol": "Code"})
    cols = [c for c in ["Code", "Name", "Market"] if c in df.columns]
    df = df[cols].dropna(subset=["Code", "Name"]).copy()
    if "Market" not in df.columns:
        df["Market"] = "KRX"
    df["Code"] = df["Code"].astype(str)
    return df.reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner="주가 데이터를 불러오는 중...")
def load_prices(code: str, start: str, end: str) -> pd.DataFrame:
    return fdr.DataReader(code, start, end)


def find_candidates(query: str, listing: pd.DataFrame) -> list[dict]:
    """검색어로 종목 후보를 찾는다. (국내: 이름/코드, 해외: 티커)"""
    q = query.strip()
    if not q:
        return []

    by_code = listing[listing["Code"] == q]
    exact = listing[listing["Name"] == q]
    partial = listing[listing["Name"].str.contains(q, case=False, regex=False)]
    found = pd.concat([by_code, exact, partial]).drop_duplicates(subset="Code")

    candidates = [
        {"code": r.Code, "label": f"{r.Name} ({r.Code}, {r.Market})", "name": r.Name, "domestic": True}
        for r in found.head(30).itertuples()
    ]

    # 국내에서 못 찾았고 영문 티커 형태면 해외 티커로 시도
    if not candidates and re.fullmatch(r"[A-Za-z][A-Za-z0-9.\-]{0,9}", q):
        t = q.upper()
        candidates.append({"code": t, "label": f"{t} (해외 티커)", "name": t, "domestic": False})
    return candidates


def fmt_price(value: float, domestic: bool) -> str:
    if domestic:
        return f"{int(round(value)):,}원"
    return f"{value:,.2f}"


def metric_grid(items: list[tuple], cols: int = 3) -> None:
    """지표를 가로로 cols개씩 나열한다. 화면이 좁아도 세로로 쌓이지 않는다.
    items: (라벨, 값, 변동 문구 또는 None, 상승 여부)"""
    cells = ""
    for label, value, delta, is_up in items:
        d = ""
        if delta:
            cls = "up" if is_up else "down"
            arrow = "▲" if is_up else "▼"
            d = f'<div class="mg-delta {cls}">{arrow} {delta}</div>'
        cells += f'<div class="mg-cell"><div class="mg-label">{label}</div><div class="mg-value">{value}</div>{d}</div>'
    st.markdown(
        f'<div class="mg" style="grid-template-columns: repeat({cols}, minmax(0, 1fr));">{cells}</div>',
        unsafe_allow_html=True,
    )


def period_input(key: str):
    today = date.today()
    p = st.date_input(
        "기간",
        value=(today - timedelta(days=180), today),
        max_value=today,
        format="YYYY-MM-DD",
        key=key,
    )
    if isinstance(p, (tuple, list)) and len(p) == 2:
        return p
    return None


# ======================================================================
# 탭 1: 주가 조회
# ======================================================================
def render_single(listing: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns([3, 3, 2])
    query = c1.text_input(
        "종목 검색", key="single_query",
        placeholder="종목명 (예: 삼성전자) 또는 종목코드 / 해외 티커 (예: AAPL)",
    )
    with c2:
        period = period_input("single_period")
    chart_type = c3.radio("차트 종류", ["선", "캔들"], horizontal=True, key="single_type")

    if not query.strip():
        st.info("종목명을 입력하세요.")
        return
    if period is None:
        st.info("기간의 시작일과 종료일을 모두 선택하세요.")
        return
    start_d, end_d = period

    candidates = find_candidates(query, listing)
    if not candidates:
        st.warning("일치하는 종목이 없습니다. 종목명을 다시 확인해 주세요.")
        return

    if len(candidates) > 1:
        labels = [c["label"] for c in candidates]
        chosen = candidates[labels.index(st.selectbox("종목 선택", labels, key="single_pick"))]
    else:
        chosen = candidates[0]

    try:
        df = load_prices(chosen["code"], start_d.isoformat(), end_d.isoformat())
    except Exception as e:  # noqa: BLE001
        st.error(f"주가 데이터를 불러오지 못했습니다: {e}")
        return
    if df is None or df.empty:
        st.warning("선택한 기간에 데이터가 없습니다. 기간을 바꿔 보세요.")
        return

    domestic = chosen["domestic"]
    first_close = float(df["Close"].iloc[0])
    last_close = float(df["Close"].iloc[-1])
    change = last_close - first_close
    pct = change / first_close * 100 if first_close else 0.0

    st.subheader(chosen["label"])
    metric_grid([
        ("기간 시작 종가", fmt_price(first_close, domestic), None, True),
        ("기간 마지막 종가", fmt_price(last_close, domestic), None, True),
        ("기간 변동", fmt_price(change, domestic), f"{abs(pct):.2f}%", change >= 0),
        ("기간 최고가", fmt_price(float(df["High"].max()), domestic), None, True),
        ("기간 최저가", fmt_price(float(df["Low"].min()), domestic), None, True),
    ], cols=3)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    if chart_type == "캔들":
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
                name="주가",
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["Close"], mode="lines",
                line=dict(color=LINE_COLOR, width=2), name="종가",
                hovertemplate="%{x|%Y-%m-%d}<br>종가 %{y:,.0f}<extra></extra>" if domestic
                else "%{x|%Y-%m-%d}<br>종가 %{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
    if "Volume" in df.columns:
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], marker_color=VOLUME_COLOR, name="거래량",
                   hovertemplate="%{x|%Y-%m-%d}<br>거래량 %{y:,.0f}<extra></extra>"),
            row=2, col=1,
        )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider_visible=False)
    fig.update_yaxes(title_text="가격", row=1, col=1, tickformat=",")
    fig.update_yaxes(title_text="거래량", row=2, col=1, tickformat=",")
    fig.update_layout(
        height=560, showlegend=False, hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("데이터 표 보기"):
        st.dataframe(df.sort_index(ascending=False), use_container_width=True)
    st.download_button(
        "CSV 내려받기",
        df.to_csv().encode("utf-8-sig"),
        file_name=f"{chosen['code']}_{start_d}_{end_d}.csv",
        mime="text/csv",
    )


# ======================================================================
# 탭 2: 수익률 비교
# ======================================================================
def render_compare(listing: pd.DataFrame) -> None:
    ordered = listing.sort_values("Name")
    options = {f"{r.Name} ({r.Code})": r.Code for r in ordered.itertuples()}
    names = {r.Code: r.Name for r in ordered.itertuples()}

    c1, c2 = st.columns([3, 2])
    selected = c1.multiselect(
        f"비교할 종목 (최대 {MAX_COMPARE}개, 이름을 입력하면 검색됩니다)",
        list(options.keys()), key="cmp_selected",
    )
    with c2:
        period = period_input("cmp_period")
    extra = st.text_input(
        "해외 티커 또는 종목 추가 (쉼표로 구분)", key="cmp_extra",
        placeholder="예: AAPL, NVDA, 005930",
    )

    # 선택 항목 정리 (중복 제거)
    items: list[dict] = []
    seen: set[str] = set()
    for label in selected:
        code = options[label]
        if code not in seen:
            seen.add(code)
            items.append({"code": code, "name": names[code], "domestic": True})
    for token in [t.strip() for t in extra.split(",") if t.strip()]:
        cands = find_candidates(token, listing)
        if not cands:
            st.warning(f"'{token}' 종목을 찾지 못했습니다.")
            continue
        c = cands[0]
        if c["code"] not in seen:
            seen.add(c["code"])
            items.append({"code": c["code"], "name": c["name"], "domestic": c["domestic"]})

    if not items:
        st.info("비교할 종목을 선택하세요. 종목 2개 이상부터 비교 효과가 있습니다.")
        return
    if period is None:
        st.info("기간의 시작일과 종료일을 모두 선택하세요.")
        return
    if len(items) > MAX_COMPARE:
        st.warning(f"종목이 너무 많아 앞의 {MAX_COMPARE}개만 표시합니다.")
        items = items[:MAX_COMPARE]
    start_d, end_d = period

    fig = go.Figure()
    rows = []
    for i, it in enumerate(items):
        try:
            df = load_prices(it["code"], start_d.isoformat(), end_d.isoformat())
        except Exception as e:  # noqa: BLE001
            st.warning(f"{it['name']}: 데이터를 불러오지 못했습니다 ({e})")
            continue
        s = df["Close"].dropna() if df is not None and not df.empty else pd.Series(dtype=float)
        if s.empty or float(s.iloc[0]) == 0:
            st.warning(f"{it['name']}: 선택한 기간에 데이터가 없습니다.")
            continue

        base = float(s.iloc[0])
        ret = (s / base - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=ret.index, y=ret, mode="lines", name=it["name"],
                line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=2),
                hovertemplate="%{y:+.2f}%<extra>" + it["name"] + "</extra>",
            )
        )
        rows.append({
            "종목": it["name"],
            "시작일": s.index[0].strftime("%Y-%m-%d"),
            "시작 종가": fmt_price(base, it["domestic"]),
            "마지막 종가": fmt_price(float(s.iloc[-1]), it["domestic"]),
            "기간 수익률(%)": round(float(ret.iloc[-1]), 2),
            "기간 내 최고(%)": round(float(ret.max()), 2),
            "기간 내 최저(%)": round(float(ret.min()), 2),
        })

    if not rows:
        return

    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#718096")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="수익률 (기간 시작일 = 0%)", ticksuffix="%")
    fig.update_layout(
        height=520, hovermode="x unified", font=dict(size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("기간 수익률 요약")
    summary = pd.DataFrame(rows).sort_values("기간 수익률(%)", ascending=False)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption("수익률은 각 종목의 기간 첫 거래일 종가를 0%로 놓고 계산한 값입니다. 배당은 반영하지 않은 가격 기준입니다.")


# ======================================================================
# 실행
# ======================================================================
try:
    LISTING = load_listing()
except Exception as e:  # noqa: BLE001
    st.error(f"종목 목록을 불러오지 못했습니다: {e}")
    st.stop()

tab_single, tab_compare = st.tabs(["주가 조회", "수익률 비교"])
with tab_single:
    render_single(LISTING)
with tab_compare:
    render_compare(LISTING)

st.caption("일봉 기준 데이터이며 실시간 시세가 아닙니다. 장중에는 지연되거나 당일 값이 반영되지 않을 수 있습니다.")
