"""금리 겹쳐 보기: 차트 오른쪽 축에 한국 국고채 3년 / 미국 국채 10년 금리를 점선으로 그린다. (app.py에서 가져다 씀)"""

import time
from datetime import datetime, timedelta

import FinanceDataReader as fdr
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _load_fred_dgs10() -> pd.Series:
    import io
    import urllib.request

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    with urllib.request.urlopen(url, timeout=8) as resp:
        raw = resp.read().decode("utf-8")
    data = pd.read_csv(io.StringIO(raw), na_values=["."]).dropna()
    return pd.Series(data.iloc[:, 1].astype(float).values / 100, index=pd.to_datetime(data.iloc[:, 0]))

RATE_SERIES = {"한국 국고채 3년": "KR3Y", "미국 국채 10년": "US10Y"}
RATE_COLORS = {"KR3Y": "#2F855A", "US10Y": "#B7791F"}


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _load_kr3y_range(start: str, end: str) -> list:
    """한국은행 ECOS 국고채 3년 일별 금리. sample 키는 한 번에 10건까지라 14일씩 나눠 동시에 요청한다."""
    import json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    s = datetime.strptime(start, "%Y%m%d").date()
    e = datetime.strptime(end, "%Y%m%d").date()
    wins, cur = [], s
    while cur <= e:
        w_end = min(cur + timedelta(days=13), e)
        wins.append((cur.strftime("%Y%m%d"), w_end.strftime("%Y%m%d")))
        cur = w_end + timedelta(days=1)

    def fetch(w):
        url = f"https://ecos.bok.or.kr/api/StatisticSearch/sample/json/kr/1/10/817Y002/D/{w[0]}/{w[1]}/010200000"
        for _ in range(2):
            try:
                with urllib.request.urlopen(url, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                rows = data.get("StatisticSearch", {}).get("row", [])
                return [(r["TIME"], float(r["DATA_VALUE"])) for r in rows if r.get("DATA_VALUE")]
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        return None

    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(fetch, wins))
    rows = [x for r in results if r for x in r]
    if not rows:
        raise RuntimeError("금리 데이터를 받지 못했습니다")
    return rows


@st.cache_data(ttl=60 * 60 * 6, show_spinner="금리 데이터를 불러오는 중...")
def load_rate(kind: str, start: str, end: str):
    """kind: KR3Y / US10Y. 반환: 날짜 인덱스, 값은 % 단위(예: 3.61). 실패하면 None."""
    try:
        if kind == "US10Y":
            s = _load_fred_dgs10()
            s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
            return s * 100
        try:
            d = fdr.DataReader("KR3YT=RR", start, end)
            s = d["Close"].dropna()
            if len(s) > 0:
                return s
        except Exception:  # noqa: BLE001
            pass
        rows = _load_kr3y_range(start.replace("-", ""), end.replace("-", ""))
        s = pd.Series({pd.to_datetime(t, format="%Y%m%d"): v for t, v in rows}).sort_index()
        return s
    except Exception:  # noqa: BLE001
        return None


def rate_controls(key: str) -> list:
    """'금리 함께 보기' 스위치와 표시할 금리 선택. 선택된 금리 이름 목록을 돌려준다."""
    on = st.toggle("금리 함께 보기 (오른쪽 축)", key=f"{key}_rate_on")
    if not on:
        return []
    return st.multiselect("표시할 금리", list(RATE_SERIES), default=list(RATE_SERIES), key=f"{key}_rate_sel")


def add_rates(fig, kinds: list, start_d, end_d, axis_n: int, domain=None) -> None:
    """차트에 금리 선(점선)을 오른쪽 축(yN)으로 겹쳐 그린다."""
    drawn = False
    for name in kinds:
        code = RATE_SERIES[name]
        s = load_rate(code, start_d.isoformat(), end_d.isoformat())
        if s is None or len(s) == 0:
            st.caption(f"{name}: 금리 데이터를 불러오지 못했거나 이 기간에 데이터가 없습니다.")
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=f"{name} (우축)", yaxis=f"y{axis_n}",
            line=dict(color=RATE_COLORS[code], width=1.5, dash="dash"),
            hovertemplate="%{y:.2f}%<extra>" + name + "</extra>",
        ))
        drawn = True
    if drawn:
        ax = dict(title="금리(%)", overlaying="y", side="right", showgrid=False, ticksuffix="%", anchor="x")
        if domain:
            ax["domain"] = domain
        fig.update_layout(**{f"yaxis{axis_n}": ax})
        st.caption("점선은 금리(오른쪽 축)입니다. 한국 국고채 3년은 한국은행 ECOS, 미국 국채 10년은 FRED 일별 값이며, 휴장일은 비어 있을 수 있습니다.")
