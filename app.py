"""주가 차트 조회 프로그램.

탭 1) 주가 조회: 종목 하나의 기간별 주가 차트 + 기간 통계
탭 2) 수익률 비교: 여러 종목을 기간 시작일 = 0% 기준으로 겹쳐서 비교

데이터: FinanceDataReader (국내 KRX 전 종목 + 해외 티커). 일봉 기준이며 실시간 틱 데이터는 아닙니다.
"""

import re
from datetime import date, datetime, timedelta, timezone

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


QUICK_RANGES = {"1개월": 1, "3개월": 3, "6개월": 6, "1년": 12, "5년": 60}


def kst_today() -> date:
    """한국 시간 기준 오늘 날짜 (서버가 UTC여도 날짜가 어긋나지 않게)."""
    return datetime.now(timezone(timedelta(hours=9))).date()


def _apply_quick(key: str) -> None:
    label = st.session_state.get(f"{key}_quick")
    if label in QUICK_RANGES:
        end = kst_today()
        start = (pd.Timestamp(end) - pd.DateOffset(months=QUICK_RANGES[label])).date()
        st.session_state[key] = (start, end)


def _clear_quick(key: str) -> None:
    st.session_state[f"{key}_quick"] = None


def period_input(key: str):
    """빠른 기간 버튼(오늘 기준) + 직접 선택 달력. 종료일은 오늘이며, 데이터는 가장 최근 영업일까지 나온다."""
    today = kst_today()
    if hasattr(st, "pills"):
        st.pills(
            "빠른 기간 (오늘 기준 ~ 가장 최근 영업일)", list(QUICK_RANGES),
            key=f"{key}_quick", on_change=_apply_quick, args=(key,),
        )
    p = st.date_input(
        "기간",
        value=(today - timedelta(days=180), today),
        max_value=today,
        format="YYYY-MM-DD",
        key=key,
        on_change=_clear_quick,
        args=(key,),
    )
    if isinstance(p, (tuple, list)) and len(p) == 2:
        return p
    return None


TRADING_DAYS = 252


def _pct(x: float, sign: bool = False) -> str:
    if x != x:  # NaN
        return "-"
    return f"{x * 100:+.2f}%" if sign else f"{x * 100:.2f}%"


def _with_date(text: str, d) -> str:
    return f'{text}<br><span style="font-size:0.7rem;opacity:0.7">{d:%Y-%m-%d}</span>'


# 무위험수익률: 한국 종목 = 국고채 3년물, 해외(미국) 종목 = 미국 국채 10년물
# 한국 국고채 3년물은 고정값이므로 금리가 크게 바뀌면 이 값을 갱신해야 한다.
RF_KR = 0.0374
RF_KR_ASOF = "2026-11-02"
# 미국 10년물은 FRED에서 가져오고, 실패하면 아래 고정값을 쓴다.
RF_US_FALLBACK = 0.0487
RF_US_FALLBACK_ASOF = "2026-11-06"


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_us10y() -> tuple:
    """(금리, 기준일, 실시간 여부). FRED DGS10 CSV를 읽고, 실패하면 고정값."""
    try:
        import io
        import urllib.request

        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
        with urllib.request.urlopen(url, timeout=6) as resp:
            raw = resp.read().decode("utf-8")
        data = pd.read_csv(io.StringIO(raw), na_values=["."]).dropna()
        return float(data.iloc[-1, 1]) / 100, str(data.iloc[-1, 0]), True
    except Exception:  # noqa: BLE001
        return RF_US_FALLBACK, RF_US_FALLBACK_ASOF, False


def risk_free_for(domestic: bool) -> tuple:
    """종목이 속한 나라에 맞는 (무위험수익률, 설명 문구)."""
    if domestic:
        return RF_KR, f"한국 국고채 3년물 {RF_KR * 100:.2f}% ({RF_KR_ASOF} 기준, 고정값)"
    rate, asof, live = get_us10y()
    src = "FRED" if live else "고정값"
    return rate, f"미국 국채 10년물 {rate * 100:.2f}% ({asof} 기준, {src}) · 해외 티커는 모두 미국 금리 적용"


def show_stats(df: pd.DataFrame, rf: float, rf_label: str) -> None:
    """조회 기간의 기본 통계 (일별 종가 수익률 기준)."""
    close = df["Close"].dropna()
    r = close.pct_change().dropna()
    if len(r) < 2:
        st.info("통계를 계산하기에 데이터가 부족합니다. 기간을 늘려 보세요.")
        return

    n = len(r)
    mean_d = float(r.mean())
    std_d = float(r.std(ddof=1))
    mean_a = mean_d * TRADING_DAYS
    std_a = std_d * TRADING_DAYS ** 0.5
    total = float(close.iloc[-1] / close.iloc[0] - 1)
    years = n / TRADING_DAYS
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 and years > 0 else float("nan")
    dd = close / close.cummax() - 1
    mdd, mdd_date = float(dd.min()), dd.idxmin()
    best_date, worst_date = r.idxmax(), r.idxmin()
    up_ratio = float((r > 0).mean())
    sharpe = (mean_a - rf) / std_a if std_a > 0 else float("nan")

    metric_grid([
        ("기간 수익률", _pct(total, True), None, True),
        ("연환산 수익률(복리)", _pct(cagr, True), None, True),
        ("일평균 수익률", _pct(mean_d, True), None, True),
        ("연환산 기대수익률", _pct(mean_a, True), None, True),
        ("일간 표준편차", _pct(std_d), None, True),
        ("연환산 표준편차", _pct(std_a), None, True),
        ("최대 낙폭(MDD)", _with_date(_pct(mdd), mdd_date), None, True),
        ("최대 일간 상승", _with_date(_pct(float(r.max()), True), best_date), None, True),
        ("최대 일간 하락", _with_date(_pct(float(r.min()), True), worst_date), None, True),
        ("상승일 비율", _pct(up_ratio), None, True),
        (f"샤프비율 (무위험 {rf * 100:.2f}%)", "-" if sharpe != sharpe else f"{sharpe:.2f}", None, True),
        ("수익률 표본 수", f"{n:,}일", None, True),
    ], cols=3)
    st.caption(
        "일별 종가 기준 단순 수익률로 계산한 과거 통계입니다. "
        "기대수익률 = 일별 수익률 평균, 연환산은 평균×252 · 표준편차×√252. "
        "샤프비율 = (연환산 기대수익률 − 무위험수익률) ÷ 연환산 표준편차. "
        f"무위험수익률: {rf_label}. "
        "조회 기간이 짧으면 연환산 값이 크게 부풀려질 수 있고, 미래 수익을 보장하지 않습니다."
    )


# ======================================================================
# 탭 1: 주가 조회
# ======================================================================
def render_single(listing: pd.DataFrame) -> None:
    c1, c3 = st.columns([3, 2])
    query = c1.text_input(
        "종목 검색", key="single_query",
        placeholder="종목명 (예: 삼성전자) 또는 종목코드 / 해외 티커 (예: AAPL)",
    )
    chart_type = c3.radio("차트 종류", ["선", "캔들"], horizontal=True, key="single_type")
    period = period_input("single_period")

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

    st.caption(f"조회 구간: {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d} (가장 최근 거래일 기준, {len(df):,}거래일)")

    # 캔들이 너무 많으면 주봉으로 묶어서 깨져 보이지 않게 한다
    plot_df = df
    weekly = False
    if chart_type == "캔들" and len(df) > 300:
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in df.columns:
            agg["Volume"] = "sum"
        plot_df = df.resample("W-FRI").agg(agg).dropna(subset=["Open", "Close"])
        weekly = True
    n_bars = len(plot_df)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    if chart_type == "캔들":
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"],
                increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
                increasing_line_width=1, decreasing_line_width=1,
                whiskerwidth=0.4,
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
    if "Volume" in plot_df.columns:
        fig.add_trace(
            go.Bar(x=plot_df.index, y=plot_df["Volume"], marker_color=VOLUME_COLOR, name="거래량",
                   hovertemplate="%{x|%Y-%m-%d}<br>거래량 %{y:,.0f}<extra></extra>"),
            row=2, col=1,
        )
    # 봉이 많을수록 간격을 넓혀 봉을 가늘게 (굵고 뭉개져 보이는 현상 방지)
    box_gap = 0.3 if n_bars <= 60 else 0.45 if n_bars <= 150 else 0.6
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider_visible=False)
    fig.update_yaxes(title_text="가격", row=1, col=1, tickformat=",")
    fig.update_yaxes(title_text="거래량", row=2, col=1, tickformat=",")
    fig.update_layout(
        height=560, showlegend=False, hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
        boxgap=box_gap, bargap=0.3 if n_bars <= 150 else 0.4,
    )
    st.plotly_chart(fig, use_container_width=True)
    if weekly:
        st.caption("기간이 길어 캔들을 주봉(1주 = 1개)으로 묶어서 표시합니다. 위 지표는 일별 데이터 기준입니다.")

    st.subheader("기간 통계")
    rf, rf_label = risk_free_for(domestic)
    show_stats(df, rf, rf_label)

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

    selected = st.multiselect(
        f"비교할 종목 (최대 {MAX_COMPARE}개, 이름을 입력하면 검색됩니다)",
        list(options.keys()), key="cmp_selected",
    )
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
