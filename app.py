"""주가 차트 조회 프로그램.

종목명(또는 종목코드/해외 티커)을 검색하고 기간을 고르면 해당 기간의 주가 차트를 보여줍니다.
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

st.set_page_config(page_title="주가 차트", layout="wide")
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


# ---------- 입력 ----------
today = date.today()
c1, c2, c3 = st.columns([3, 3, 2])
query = c1.text_input("종목 검색", placeholder="종목명 (예: 삼성전자) 또는 종목코드 / 해외 티커 (예: AAPL)")
period = c2.date_input(
    "기간",
    value=(today - timedelta(days=180), today),
    max_value=today,
    format="YYYY-MM-DD",
)
chart_type = c3.radio("차트 종류", ["선", "캔들"], horizontal=True)

if not query.strip():
    st.info("종목명을 입력하세요.")
    st.stop()

if not (isinstance(period, (tuple, list)) and len(period) == 2):
    st.info("기간의 시작일과 종료일을 모두 선택하세요.")
    st.stop()
start_d, end_d = period

try:
    listing = load_listing()
except Exception as e:  # noqa: BLE001
    st.error(f"종목 목록을 불러오지 못했습니다: {e}")
    st.stop()

candidates = find_candidates(query, listing)
if not candidates:
    st.warning("일치하는 종목이 없습니다. 종목명을 다시 확인해 주세요.")
    st.stop()

if len(candidates) > 1:
    labels = [c["label"] for c in candidates]
    chosen = candidates[labels.index(st.selectbox("종목 선택", labels))]
else:
    chosen = candidates[0]

# ---------- 데이터 ----------
try:
    df = load_prices(chosen["code"], start_d.isoformat(), end_d.isoformat())
except Exception as e:  # noqa: BLE001
    st.error(f"주가 데이터를 불러오지 못했습니다: {e}")
    st.stop()

if df is None or df.empty:
    st.warning("선택한 기간에 데이터가 없습니다. 기간을 바꿔 보세요.")
    st.stop()

domestic = chosen["domestic"]
first_close = float(df["Close"].iloc[0])
last_close = float(df["Close"].iloc[-1])
change = last_close - first_close
pct = change / first_close * 100 if first_close else 0.0

st.subheader(chosen["label"])
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("기간 시작 종가", fmt_price(first_close, domestic))
m2.metric("기간 마지막 종가", fmt_price(last_close, domestic))
m3.metric("기간 변동", fmt_price(change, domestic), f"{pct:+.2f}%")
m4.metric("기간 최고가", fmt_price(float(df["High"].max()), domestic))
m5.metric("기간 최저가", fmt_price(float(df["Low"].min()), domestic))

# ---------- 차트 ----------
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
fig.update_layout(height=640, showlegend=False, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---------- 표 / 내려받기 ----------
with st.expander("데이터 표 보기"):
    st.dataframe(df.sort_index(ascending=False), use_container_width=True)

st.download_button(
    "CSV 내려받기",
    df.to_csv().encode("utf-8-sig"),
    file_name=f"{chosen['code']}_{start_d}_{end_d}.csv",
    mime="text/csv",
)
st.caption("일봉 기준 데이터이며 실시간 시세가 아닙니다. 장중에는 지연되거나 당일 값이 반영되지 않을 수 있습니다.")
