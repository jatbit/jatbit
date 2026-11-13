"""주가 차트 · 시장 분석 앱.

상단 메뉴: 시장(첫 화면) · 주가 조회 · 기업 리서치 · 스크리너 · 수익률 비교 · 포트폴리오 · 백테스트 · 옵션 · 채권.
열린 페이지와 열린 탭만 계산해서 빠르게 보여주고, 자주 보는 지표를 먼저, 세부 지표는 '더보기'에 접어 둔다.
수익률 비교와 포트폴리오의 종목·기간 입력은 서로 동기화된다.

데이터: FinanceDataReader (국내 KRX 전 종목 + 해외 티커 + 해외 선물), Yahoo Finance. 일봉 기준이며 실시간 틱 데이터는 아닙니다.
"""

import math
import re
import time
from datetime import date, datetime, timedelta, timezone

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import autoreload  # GitHub에 올린 새 코드를 새로고침만으로 반영 (autoreload.py)

autoreload.refresh()
from rates import add_rates, rate_controls
# 옵션·리서치·백테스트·채권·시장 모듈은 그 페이지를 열 때 불러온다 (첫 화면을 빠르게)

LINE_COLOR = "#2B6CB0"
UP_COLOR = "#D64545"      # 국내 관행: 상승 = 빨강
DOWN_COLOR = "#2B6CB0"    # 하락 = 파랑
VOLUME_COLOR = "#A0AEC0"
SERIES_COLORS = ["#2B6CB0", "#D64545", "#2F855A", "#B7791F", "#6B46C1", "#0987A0", "#DD6B20", "#4A5568"]


def series_color(i: int) -> str:
    """종목 수에 제한이 없으므로 기본 8색이 모자라면 색상환에서 추가로 만든다."""
    if i < len(SERIES_COLORS):
        return SERIES_COLORS[i]
    return f"hsl({(i * 47) % 360}, 60%, 45%)"

st.set_page_config(page_title="주가 차트", page_icon="📈", layout="wide")

import ui  # 화면 공통 요소 (ui.py)

ui.apply_theme()


@st.cache_data(ttl=60 * 60 * 24, show_spinner="종목 목록을 불러오는 중...")
def load_listing() -> pd.DataFrame:
    df = fdr.StockListing("KRX")
    if "Symbol" in df.columns and "Code" not in df.columns:
        df = df.rename(columns={"Symbol": "Code"})
    cols = [c for c in ["Code", "Name", "Market", "Marcap"] if c in df.columns]
    df = df[cols].dropna(subset=["Code", "Name"]).copy()
    if "Marcap" not in df.columns:
        df["Marcap"] = float("nan")
    if "Market" not in df.columns:
        df["Market"] = "KRX"
    df["Code"] = df["Code"].astype(str)
    return df.reset_index(drop=True)


def _extend_back(code: str, start: str, d):
    """국내 종목은 한 번에 약 3,000거래일까지만 오므로, 시작일보다 데이터가 짧으면 더 과거 구간을 야후에서 받아 앞에 이어 붙인다."""
    try:
        if d is None or len(d) < 2900 or not re.fullmatch(r"[0-9]{6}", str(code)):
            return d
        want = pd.Timestamp(start)
        first = d.index.min()
        if first <= want + pd.Timedelta(days=7):
            return d
        old = None
        for suf in (".KS", ".KQ"):
            try:
                y = fdr.DataReader("YAHOO:" + code + suf, start, (first + pd.Timedelta(days=5)).strftime("%Y-%m-%d"))
            except Exception:  # noqa: BLE001
                continue
            if y is not None and len(y) > 0 and "Close" in y.columns:
                old = y
                break
        if old is None:
            return d
        old = old[old["Close"] > 0]
        # 이어지는 날짜의 종가 비율로 야후 가격 수준을 맞춘다.
        ratio = 1.0
        if first in old.index and float(old.loc[first, "Close"]) > 0:
            ratio = float(d.loc[first, "Close"]) / float(old.loc[first, "Close"])
        old = old[old.index < first].copy()
        if len(old) == 0:
            return d
        for _c in ("Open", "High", "Low", "Close"):
            if _c in old.columns:
                old[_c] = old[_c] * ratio
        cols = [c for c in d.columns if c in old.columns]
        out = pd.concat([old[cols], d])
        return out[~out.index.duplicated(keep="last")].sort_index()
    except Exception:  # noqa: BLE001
        return d


@st.cache_data(ttl=300, show_spinner="주가 데이터를 불러오는 중...")
def load_prices(code: str, start: str, end: str) -> pd.DataFrame:
    """일시적인 네트워크 오류로 데이터가 빠지지 않도록 최대 3번까지 다시 시도한다. (성공한 결과만 캐시됨)"""
    last_err = None
    for attempt in range(3):
        try:
            d = _extend_back(code, start, fdr.DataReader(code, start, end))
            if d is not None and "Close" in d.columns:
                d = d[d["Close"] != 0].copy()  # 오래된 구간의 0원(거래정지 등) 행 제거
                for _c in ("Open", "High", "Low"):
                    if _c in d.columns:
                        d[_c] = d[_c].where(d[_c] != 0, d["Close"])
            return d
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise last_err


# 해외 선물: Yahoo 연속(근월물 이어붙인) 시세. 티커 → (표시 이름, 한글 별칭들). 목록에 없는 '=F' 심볼도 그대로 조회는 된다.
FUTURES = {
    # 귀금속·산업금속
    "GC=F": ("금 선물", ["금선물", "금"]),
    "SI=F": ("은 선물", ["은선물", "은"]),
    "HG=F": ("구리 선물", ["구리선물", "구리"]),
    "PL=F": ("백금 선물", ["백금선물", "백금"]),
    "PA=F": ("팔라듐 선물", ["팔라듐선물", "팔라듐"]),
    # 에너지
    "CL=F": ("WTI 원유 선물", ["원유선물", "원유", "wti", "wti원유", "wti원유선물", "유가"]),
    "BZ=F": ("브렌트유 선물", ["브렌트유선물", "브렌트유", "브렌트", "brent"]),
    "NG=F": ("천연가스 선물", ["천연가스선물", "천연가스", "가스선물"]),
    "HO=F": ("난방유 선물", ["난방유선물", "난방유"]),
    "RB=F": ("휘발유 선물", ["휘발유선물", "휘발유", "가솔린선물"]),
    # 곡물·농산물
    "ZC=F": ("옥수수 선물", ["옥수수선물", "옥수수"]),
    "ZW=F": ("밀 선물", ["밀선물", "소맥선물"]),
    "ZS=F": ("대두 선물", ["대두선물", "대두", "콩선물"]),
    "ZM=F": ("대두박 선물", ["대두박선물", "대두박"]),
    "ZL=F": ("대두유 선물", ["대두유선물", "대두유"]),
    "ZO=F": ("귀리 선물", ["귀리선물", "귀리"]),
    "ZR=F": ("쌀 선물", ["쌀선물", "쌀"]),
    # 소프트(기호식품·섬유)·목재
    "KC=F": ("커피 선물", ["커피선물", "커피"]),
    "SB=F": ("설탕 선물", ["설탕선물", "설탕"]),
    "CC=F": ("코코아 선물", ["코코아선물", "코코아"]),
    "CT=F": ("면화 선물", ["면화선물", "면화", "목화선물"]),
    "OJ=F": ("오렌지주스 선물", ["오렌지주스선물", "오렌지주스"]),
    "LBR=F": ("목재 선물", ["목재선물", "목재"]),
    # 축산
    "LE=F": ("생우(소) 선물", ["생우선물", "생우", "소선물"]),
    "GF=F": ("비육우 선물", ["비육우선물", "비육우"]),
    "HE=F": ("돼지고기(린호그) 선물", ["돼지고기선물", "돼지선물", "린호그선물"]),
    # 주가지수 선물
    "ES=F": ("S&P500 선물", ["s&p500선물", "sp500선물", "에스앤피선물"]),
    "NQ=F": ("나스닥 선물", ["나스닥선물", "나스닥100선물"]),
    "YM=F": ("다우존스 선물", ["다우선물", "다우존스선물", "다우지수선물"]),
    "RTY=F": ("러셀2000 선물", ["러셀2000선물", "러셀선물"]),
    # 채권·통화·가상자산 선물
    "ZT=F": ("미국 2년 국채 선물", ["2년국채선물", "미국2년국채선물"]),
    "ZF=F": ("미국 5년 국채 선물", ["5년국채선물", "미국5년국채선물"]),
    "ZN=F": ("미국 10년 국채 선물", ["10년국채선물", "미국10년국채선물", "10년물선물"]),
    "ZB=F": ("미국 30년 국채 선물", ["30년국채선물", "미국30년국채선물", "30년물선물"]),
    "DX-Y.NYB": ("달러인덱스(DXY, 지수)", ["달러인덱스선물", "달러지수선물", "달러인덱스", "달러지수", "dxy"]),
    "BTC=F": ("비트코인 선물(CME)", ["비트코인선물", "비트코인"]),
    "ETH=F": ("이더리움 선물(CME)", ["이더리움선물", "이더리움"]),
}
FUTURES_NAMES = {t: v[0] for t, v in FUTURES.items()}
# 한글로 '금', '은'만 치면 일반 종목명(예: ~금융, ~은행)과 겹치므로, 한 글자 별칭은 '금 선물'처럼 '선물'을 붙였을 때만 쓴다.
_FUT_ALIAS = {}
for _t, (_n, _al) in FUTURES.items():
    for _a in _al:
        if len(_a) >= 2 or _a.endswith("선물"):
            _FUT_ALIAS[_a] = _t
    _FUT_ALIAS[_n.replace(" ", "").lower()] = _t


FUT_OPTIONS = {f"{v[0]} ({t})": t for t, v in FUTURES.items()}  # 목록에서 골라 넣는 선물


def _futures_items(labels) -> list[dict]:
    return [{"code": FUT_OPTIONS[l], "name": FUTURES_NAMES[FUT_OPTIONS[l]], "domestic": False, "market": "US"}
            for l in labels if l in FUT_OPTIONS]


def _futures_candidate(t: str) -> dict:
    nm = FUTURES_NAMES.get(t, t)
    label = f"{nm} ({t}, 해외 선물)" if t in FUTURES_NAMES else f"{t} (해외 티커)"
    return {"code": t, "label": label, "name": nm, "domestic": False, "market": "US"}


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
        {"code": r.Code, "label": f"{r.Name} ({r.Code}, {r.Market})", "name": r.Name, "domestic": True, "market": str(r.Market)}
        for r in found.head(30).itertuples()
    ]

    # 한글/영문 별칭으로 해외 선물 찾기 (예: '원유 선물' → CL=F). 별칭과 정확히 일치할 때만.
    key = q.replace(" ", "").lower()
    if key in _FUT_ALIAS:
        candidates.insert(0, _futures_candidate(_FUT_ALIAS[key]))
    # 국내에서 못 찾았고 영문 티커 형태면 해외 티커로 시도
    elif not candidates and re.fullmatch(r"[A-Za-z][A-Za-z0-9.\-=]{0,9}", q):
        candidates.append(_futures_candidate(q.upper()))
    return candidates


def fmt_price(value: float, domestic: bool) -> str:
    if domestic:
        return f"{int(round(value)):,}원"
    return f"{value:,.2f}"


def metric_grid(items: list[tuple], cols: int = 3) -> None:
    """지표 카드 격자. items: (라벨, 값, 변동 문구 또는 None, 상승 여부). 값에는 HTML(날짜 줄바꿈 등)을 쓸 수 있다."""
    from glossary import CSS, icon, tip

    cells = ""
    for i, (label, value, delta, is_up) in enumerate(items):
        d = ""
        if delta:
            cls = "up" if is_up else "down"
            arrow = "▲" if is_up else "▼"
            d = f'<div class="kc-d {cls}">{arrow} {delta}</div>'
        right = " qr" if cols > 1 and (i % cols) >= cols / 2 else ""
        cells += (f'<div class="kc-c{right}"><div class="kc-l"><span>{label}</span>{icon(tip(label))}</div>'
                  f'<div class="kc-v">{value}</div>{d}</div>')
    st.markdown(CSS + f'<div class="kc" style="grid-template-columns:repeat({cols},minmax(0,1fr))">{cells}</div>',
                unsafe_allow_html=True)


QUICK_RANGES = {"1개월": 1, "3개월": 3, "6개월": 6, "1년": 12, "5년": 60, "10년": 120, "20년": 240, "전체": 720}


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


def _mirror(src: str, dst: str) -> None:
    """src 위젯의 값을 dst 위젯으로 복사한다 (탭 사이 입력 동기화)."""
    if src in st.session_state:
        st.session_state[dst] = st.session_state[src]


def _sync_period(key: str, mirror: str) -> None:
    v = st.session_state.get(key)
    if isinstance(v, (tuple, list)) and len(v) == 2:  # 시작일만 고른 중간 상태는 복사하지 않는다
        st.session_state[mirror] = v
    st.session_state[f"{mirror}_quick"] = st.session_state.get(f"{key}_quick")


def _on_quick(key: str, mirror) -> None:
    _apply_quick(key)
    if mirror:
        _sync_period(key, mirror)


def _on_date(key: str, mirror) -> None:
    _clear_quick(key)
    if mirror:
        _sync_period(key, mirror)


US_TOP5 = ["NVDA", "AAPL", "GOOGL", "MSFT", "AMZN"]  # 고정 목록(2026-11 기준 시가총액 상위, 수시로 바뀔 수 있음)


def _preset(kind: str) -> None:
    """시가총액 상위 5개 종목을 두 탭(수익률 비교·포트폴리오)에 동시에 채운다."""
    listing = st.session_state.get("_listing")
    if kind == "KR":
        labels = []
        if listing is not None and "Marcap" in listing.columns:
            d = listing.dropna(subset=["Marcap"])
            d = d[d["Code"].astype(str).str.endswith("0")]  # 우선주 제외(보통주 코드는 0으로 끝남)
            d = d.sort_values("Marcap", ascending=False).head(5)
            labels = [f"{r.Name} ({r.Code})" for r in d.itertuples()]
        for k in ("cmp_selected", "pf_selected"):
            st.session_state[k] = labels
        for k in ("cmp_extra", "pf_extra"):
            st.session_state[k] = ""
    else:
        for k in ("cmp_selected", "pf_selected"):
            st.session_state[k] = []
        for k in ("cmp_extra", "pf_extra"):
            st.session_state[k] = ", ".join(US_TOP5)


def _row(spec):
    try:
        return st.columns(spec, vertical_alignment="bottom")
    except TypeError:
        return st.columns(spec)


def period_input(key: str, mirror: str = None, presets: bool = False):
    """빠른 기간 버튼(오늘 기준) + 직접 선택 달력을 한 줄에. 종료일은 오늘이며, 데이터는 가장 최근 영업일까지 나온다.
    mirror: 같은 기간을 공유할 다른 페이지의 period_input 키."""
    today = kst_today()
    cols = _row([3.4, 2.2] + ([1, 1] if presets else []))
    if hasattr(st, "pills"):
        with cols[0]:
            st.pills(
                "빠른 기간", list(QUICK_RANGES), key=f"{key}_quick", on_change=_on_quick, args=(key, mirror),
                label_visibility="collapsed", help="오늘 기준 ~ 가장 최근 영업일",
            )
    with cols[1]:
        p = st.date_input(
            "기간",
            value=(today - timedelta(days=180), today),
            min_value=date(1950, 1, 1),
            max_value=today,
            format="YYYY-MM-DD",
            key=key,
            on_change=_on_date,
            args=(key, mirror),
            label_visibility="collapsed",
        )
    if presets:
        for col, kind, label in ((cols[2], "KR", "한국 Top5"), (cols[3], "US", "미국 Top5")):
            with col:
                st.button(label, key=f"{key}_preset_{kind}", on_click=_preset, args=(kind,),
                          use_container_width=True, help="시가총액 상위 5개 종목을 한 번에 입력")
    if isinstance(p, (tuple, list)) and len(p) == 2:
        return p
    return None


TRADING_DAYS = 252


def _pct(x: float, sign: bool = False) -> str:
    if x != x:  # NaN
        return "-"
    return f"{x * 100:+.2f}%" if sign else f"{x * 100:.2f}%"


def _with_date(text: str, d) -> str:
    return f'{text}<br><span style="font-size:0.6rem;opacity:0.7">{d:%Y-%m-%d}</span>'


# 무위험수익률: 조회 기간의 마지막 영업일 기준 금리를 가져온다.
#  - 한국 종목: 국고채 3년물 (한국은행 ECOS 일별 시장금리, 공개 sample 인증키)
#  - 해외 티커: 미국 국채 10년물 (FRED DGS10)
# 금리를 가져오지 못하면 아래 고정값을 쓰고, 화면에 그 사실을 표시한다.
RF_KR = 0.03613
RF_KR_ASOF = "2026-11-06"
RF_US_FALLBACK = 0.0487
RF_US_FALLBACK_ASOF = "2026-11-06"


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _load_fred_dgs10() -> pd.Series:
    import io
    import urllib.request

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    with urllib.request.urlopen(url, timeout=8) as resp:
        raw = resp.read().decode("utf-8")
    data = pd.read_csv(io.StringIO(raw), na_values=["."]).dropna()
    return pd.Series(data.iloc[:, 1].astype(float).values / 100, index=pd.to_datetime(data.iloc[:, 0]))


def us10y_asof(asof: date):
    """asof 이전(당일 포함) 가장 최근 미국 10년물 금리 → (금리, 기준일) 또는 None."""
    try:
        s = _load_fred_dgs10()
        s = s[s.index <= pd.Timestamp(asof)]
        if s.empty:
            return None
        return float(s.iloc[-1]), s.index[-1].strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _load_kr3y(start: str, end: str) -> list:
    import json
    import urllib.request

    url = f"https://ecos.bok.or.kr/api/StatisticSearch/sample/json/kr/1/10/817Y002/D/{start}/{end}/010200000"
    with urllib.request.urlopen(url, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    rows = data["StatisticSearch"]["row"]
    return [(r["TIME"], float(r["DATA_VALUE"])) for r in rows if r.get("DATA_VALUE")]


def kr3y_asof(asof: date):
    """asof 이전(당일 포함) 가장 최근 국고채 3년물 금리 → (금리, 기준일) 또는 None.
    sample 인증키는 한 번에 10건까지라서 최근 12일 구간만 요청한다."""
    try:
        start = (asof - timedelta(days=12)).strftime("%Y%m%d")
        end = asof.strftime("%Y%m%d")
        rows = [x for x in _load_kr3y(start, end) if x[0] <= end]
        if not rows:
            return None
        t, v = max(rows)
        return v / 100, f"{t[:4]}-{t[4:6]}-{t[6:]}"
    except Exception:  # noqa: BLE001
        return None


def risk_free_for(domestic: bool, asof: date) -> tuple:
    """종목이 속한 나라의 (조회 기간 마지막 영업일 기준 무위험수익률, 설명 문구)."""
    if domestic:
        got = kr3y_asof(asof)
        if got:
            return got[0], f"한국 국고채 3년물 {got[0] * 100:.2f}% ({got[1]} 기준 · 조회 기간 마지막 영업일 · 한국은행 ECOS)"
        return RF_KR, (
            f"한국 국고채 3년물 {RF_KR * 100:.2f}% (금리 조회 실패로 {RF_KR_ASOF} 고정값 사용, "
            "조회 기간 마지막 영업일 값이 아님)"
        )
    got = us10y_asof(asof)
    if got:
        return got[0], (
            f"미국 국채 10년물 {got[0] * 100:.2f}% ({got[1]} 기준 · 조회 기간 마지막 영업일 · FRED) "
            "· 해외 티커는 모두 미국 금리 적용"
        )
    return RF_US_FALLBACK, (
        f"미국 국채 10년물 {RF_US_FALLBACK * 100:.2f}% (금리 조회 실패로 {RF_US_FALLBACK_ASOF} 고정값 사용, "
        "조회 기간 마지막 영업일 값이 아님)"
    )


def benchmark_for(market: str, domestic: bool) -> tuple:
    """종목이 속한 시장의 벤치마크 지수 (코드, 이름)."""
    if not domestic:
        return "US500", "S&P 500"
    if str(market).upper().startswith("KOSDAQ"):
        return "KQ11", "코스닥 지수"
    return "KS11", "코스피 지수"


def capm_stats(r: pd.Series, m: pd.Series, rf: float):
    """같은 날짜로 맞춘 일별 수익률 r(종목/포트폴리오), m(시장)의 CAPM 지표 딕셔너리."""
    var_m = float(m.var(ddof=1))
    if var_m <= 0:
        return None
    beta = float(r.cov(m)) / var_m
    corr = float(r.corr(m))
    mean_a = float(r.mean()) * TRADING_DAYS
    mean_m = float(m.mean()) * TRADING_DAYS
    std_a = float(r.std(ddof=1)) * TRADING_DAYS ** 0.5
    std_m = float(m.std(ddof=1)) * TRADING_DAYS ** 0.5
    capm = rf + beta * (mean_m - rf)
    sys_risk = abs(beta) * std_m
    return {
        "beta": beta, "corr": corr, "r2": corr ** 2,
        "mean_a": mean_a, "mean_m": mean_m, "std_a": std_a, "std_m": std_m,
        "capm": capm, "alpha": mean_a - capm,
        "treynor": (mean_a - rf) / beta if beta != 0 else float("nan"),
        "sys": sys_risk, "idio": max(std_a ** 2 - sys_risk ** 2, 0.0) ** 0.5,
        "n": len(r),
    }


Z95, Z99 = 1.6449, 2.3263


def risk_metrics(r: pd.Series) -> dict:
    """1일 기준 VaR / ES (손실률, 양수). 과거 시뮬레이션(실제 수익률 분포)과 정규분포 가정 두 가지."""
    out = {}
    mu_d, sd_d = float(r.mean()), float(r.std(ddof=1))
    for cl, z, tag in ((0.95, Z95, "95"), (0.99, Z99, "99")):
        q = float(r.quantile(1 - cl))
        tail = r[r <= q]
        out[f"var{tag}"] = -q
        out[f"es{tag}"] = -float(tail.mean()) if len(tail) else -q
        out[f"nvar{tag}"] = -(mu_d - z * sd_d)
    out["worst"] = -float(r.min())
    out["n"] = len(r)
    return out


def risk_grid(k: dict) -> None:
    metric_grid([
        ("VaR 95% (1일, 과거)", _pct(k["var95"]), None, True),
        ("ES 95% (1일, 과거)", _pct(k["es95"]), None, True),
        ("VaR 99% (1일, 과거)", _pct(k["var99"]), None, True),
        ("ES 99% (1일, 과거)", _pct(k["es99"]), None, True),
        ("VaR 95% (정규분포)", _pct(k["nvar95"]), None, True),
        ("VaR 99% (정규분포)", _pct(k["nvar99"]), None, True),
        ("최악의 하루 손실", _pct(k["worst"]), None, True),
        ("표본 수", f"{k['n']:,}일", None, True),
    ], cols=4)


RISK_NOTE = "손실은 양수로 표시합니다. 조회 기간이 짧으면 99% 값은 최악의 하루 한두 개에 좌우되므로 참고용으로만 보세요."


def capm_grid(c: dict, name: str) -> None:
    metric_grid([
        ("벤치마크", name, None, True),
        ("베타 (β)", f"{c['beta']:.2f}", None, True),
        ("상관계수 (ρ)", f"{c['corr']:.2f}", None, True),
        ("결정계수 (R²)", f"{c['r2']:.2f}", None, True),
        ("시장 연환산 수익률", _pct(c["mean_m"], True), None, True),
        ("시장 연환산 표준편차", _pct(c["std_m"]), None, True),
        ("CAPM 기대수익률", _pct(c["capm"], True), None, True),
        ("젠센 알파 (연)", _pct(c["alpha"], True), None, True),
        ("트레이너 비율", "-" if c["treynor"] != c["treynor"] else f"{c['treynor']:.3f}", None, True),
        ("체계적 위험 (|β|σm)", _pct(c["sys"]), None, True),
        ("비체계적 위험", _pct(c["idio"]), None, True),
        ("공통 표본 수", f"{c['n']:,}일", None, True),
    ], cols=4)


def capm_note(name: str) -> str:
    return f"벤치마크: {name} · 조회 기간이 짧으면 값이 크게 흔들립니다."


def show_capm(df: pd.DataFrame, rf: float, bench: tuple) -> None:
    """CAPM 기본 지표: 벤치마크 지수 대비 베타, 알파 등 (일별 수익률 회귀)."""
    code, name = bench
    try:
        mk = load_prices(code, df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"))
        m = mk["Close"].dropna().pct_change().dropna()
    except Exception as e:  # noqa: BLE001
        st.caption(f"CAPM 지표: 벤치마크({name}) 데이터를 불러오지 못했습니다 ({e})")
        return
    r = df["Close"].dropna().pct_change().dropna()
    j = pd.concat([r.rename("r"), m.rename("m")], axis=1, join="inner").dropna()
    if len(j) < 20:
        st.caption(f"CAPM 지표: 벤치마크({name})와 겹치는 거래일이 {len(j)}일뿐이라 계산하지 않았습니다. 기간을 늘려 보세요.")
        return
    c = capm_stats(j["r"], j["m"], rf)
    if c is None:
        return
    st.markdown("**CAPM 기본 지표**")
    capm_grid(c, name)
    st.caption(capm_note(name))


def show_stats(df: pd.DataFrame, rf: float, rf_label: str, bench: tuple = None) -> None:
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
    ], cols=4)
    st.caption(f"일별 종가 기준 과거 통계 · 무위험수익률: {rf_label}")
    if bench:
        show_capm(df, rf, bench)


# ======================================================================
# 탭 1: 주가 조회
# ======================================================================
IND_OPTS = ["이동평균선", "볼린저밴드", "RSI", "MACD"]
IND_DEFAULTS = {"single_ma_n": "5, 20, 60, 120", "single_bb_n": 20, "single_bb_k": 2.0, "single_rsi_n": 14, "single_rsi_hi": 70,
                "single_rsi_lo": 30, "single_macd_f": 12, "single_macd_s": 26, "single_macd_g": 9}


def _clean_ind_state() -> None:
    """예전 버전의 지표 이름(괄호 안 숫자 포함)이 남아 있으면 새 이름으로 바꾼다."""
    v = st.session_state.get("single_ind")
    if isinstance(v, (list, tuple)):
        st.session_state["single_ind"] = [o for o in IND_OPTS if any(str(x).startswith(o) for x in v)]


def _reset_ind() -> None:
    for k, v in IND_DEFAULTS.items():
        st.session_state[k] = v


def _ind_settings(inds: list) -> dict:
    """선택한 보조지표의 기간·기준값 입력칸 (기본값이 채워져 있고, 지우고 원하는 값을 넣으면 됨)."""
    import ta

    P = {}
    if not inds:
        return P
    st.markdown("**보조지표 설정**")
    if IND_OPTS[0] in inds:
        txt = st.text_input("이동평균 기간 (일, 쉼표로 구분)", value=IND_DEFAULTS["single_ma_n"], key="single_ma_n",
                            help="예: 5, 20, 60, 120. 2~400일, 최대 8개")
        P["ma"] = ta.parse_periods(txt)
    if IND_OPTS[1] in inds:
        a, b = st.columns(2)
        P["bb_n"] = int(a.number_input("볼린저 기간", 2, 400, IND_DEFAULTS["single_bb_n"], step=1, key="single_bb_n"))
        P["bb_k"] = float(b.number_input("표준편차 배수", 0.5, 5.0, IND_DEFAULTS["single_bb_k"], step=0.5, key="single_bb_k"))
    if IND_OPTS[2] in inds:
        a, b, c = st.columns(3)
        P["rsi_n"] = int(a.number_input("RSI 기간", 2, 200, IND_DEFAULTS["single_rsi_n"], step=1, key="single_rsi_n"))
        P["rsi_hi"] = int(b.number_input("과매수", 50, 100, IND_DEFAULTS["single_rsi_hi"], step=5, key="single_rsi_hi"))
        P["rsi_lo"] = int(c.number_input("과매도", 0, 50, IND_DEFAULTS["single_rsi_lo"], step=5, key="single_rsi_lo"))
    if IND_OPTS[3] in inds:
        a, b, c = st.columns(3)
        P["macd_f"] = int(a.number_input("MACD 단기", 2, 200, IND_DEFAULTS["single_macd_f"], step=1, key="single_macd_f"))
        P["macd_s"] = int(b.number_input("MACD 장기", 3, 400, IND_DEFAULTS["single_macd_s"], step=1, key="single_macd_s"))
        P["macd_g"] = int(c.number_input("시그널", 2, 100, IND_DEFAULTS["single_macd_g"], step=1, key="single_macd_g"))
        if P["macd_f"] >= P["macd_s"]:
            st.caption("단기 기간이 장기 기간보다 짧아야 합니다. 두 값을 바꿔서 계산합니다.")
            P["macd_f"], P["macd_s"] = min(P["macd_f"], P["macd_s"]), max(P["macd_f"], P["macd_s"]) + (P["macd_f"] == P["macd_s"])
    st.button("기본값으로 되돌리기", key="single_ind_reset", on_click=_reset_ind)
    return P


def _indicator_base(code: str, start_d, end_d, weekly: bool, longest: int = 120):
    """보조지표 계산용 종가: 앞쪽 계산 구간(가장 긴 기간)이 비지 않게 조회 시작일보다 넉넉히 앞에서부터."""
    back = int(longest * (7 if weekly else 1.5)) + 60  # 거래일 → 달력일 여유
    try:
        ext = load_prices(code, (pd.Timestamp(start_d) - pd.Timedelta(days=back)).date().isoformat(),
                          end_d.isoformat())
    except Exception:  # noqa: BLE001
        return None
    if ext is None or ext.empty:
        return None
    c = ext["Close"].dropna()
    return c.resample("W-FRI").last().dropna() if weekly else c


def render_single(listing: pd.DataFrame) -> None:
    c1, c2 = _row([5, 1])
    query = c1.text_input(
        "종목 검색", key="single_query", label_visibility="collapsed",
        placeholder="종목명 (예: 삼성전자) 또는 종목코드 / 해외 티커 (예: AAPL, 원유 선물)",
    )
    with c2:
        with ui.popover("차트 설정", icon=":material/tune:"):
            chart_type = st.radio("차트 종류", ["선", "캔들"], horizontal=True, key="single_type")
            basis = st.radio("표시 기준", ["주가", "수익률"], horizontal=True, key="single_basis")
            _clean_ind_state()
            inds = st.multiselect("보조지표", IND_OPTS, key="single_ind", placeholder="이동평균선, 볼린저밴드, RSI, MACD")
            IP = _ind_settings(inds)
            rate_sel = rate_controls("single")
    period = period_input("single_period")

    if not query.strip():
        st.info("종목명이나 코드를 입력하세요. 예: 삼성전자, 005930, AAPL, 원유 선물")
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
    # 최근 1년(52주) 시세: 현재가·전일 대비·52주 범위용
    today = kst_today()
    y1 = df
    if (pd.Timestamp(df.index[0]) > pd.Timestamp(today - timedelta(days=360))) or end_d < today:
        try:
            y1 = load_prices(chosen["code"], (today - timedelta(days=372)).isoformat(), today.isoformat())
        except Exception:  # noqa: BLE001
            y1 = df
        if y1 is None or y1.empty:
            y1 = df
    y1 = y1[y1.index >= y1.index[-1] - pd.Timedelta(days=365)]
    last = y1.iloc[-1]
    px = float(last["Close"])
    prev = float(y1["Close"].iloc[-2]) if len(y1) >= 2 else float("nan")
    chg = px - prev
    pct_d = chg / prev * 100 if prev == prev and prev else float("nan")
    name_part = chosen["label"].split(" (")[0]
    code_part = chosen["label"][len(name_part):].strip(" ()")
    ui.quote_header(name_part, code_part, "", fmt_price(px, domestic), fmt_price(abs(chg), domestic) if chg == chg else None,
                    pct_d, f"{y1.index[-1]:%Y-%m-%d} 종가")

    first_close = float(df["Close"].iloc[0])
    period_ret = float(df["Close"].iloc[-1]) / first_close - 1 if first_close else float("nan")
    mdd = float((df["Close"] / df["Close"].cummax() - 1).min())
    hi52, lo52 = float(y1["High"].max()), float(y1["Low"].min())
    vol_txt, vol_sub = "-", ""
    if "Volume" in y1.columns and float(last.get("Volume", 0) or 0) > 0:
        v = float(last["Volume"])
        avg20 = float(y1["Volume"].iloc[-21:-1].mean()) if len(y1) > 21 else float("nan")
        vol_txt = f"{v:,.0f}"
        vol_sub = f"20일 평균의 {v / avg20 * 100:.0f}%" if avg20 == avg20 and avg20 > 0 else ""
    ui.cards([
        ("시가", fmt_price(float(last["Open"]), domestic)),
        ("고가", fmt_price(float(last["High"]), domestic)),
        ("저가", fmt_price(float(last["Low"]), domestic)),
        ("거래량", vol_txt, vol_sub),
        ("52주 최고", fmt_price(hi52, domestic), f"현재가 대비 {(px / hi52 - 1) * 100:+.1f}%" if hi52 else ""),
        ("52주 최저", fmt_price(lo52, domestic), f"현재가 대비 {(px / lo52 - 1) * 100:+.1f}%" if lo52 else ""),
        ("기간 수익률", _pct(period_ret, True), f"{df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}", period_ret),
        ("최대 낙폭(MDD)", _pct(mdd), "조회 기간 고점 대비 최대 하락"),
    ], cols=4)

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
    pct_mode = basis == "수익률"
    if pct_mode:
        base0 = float(df["Close"].dropna().iloc[0])
        plot_df = plot_df.copy()
        for _c in ("Open", "High", "Low", "Close"):
            if _c in plot_df.columns:
                plot_df[_c] = (plot_df[_c] / base0 - 1) * 100

    import ta

    want_ma = IND_OPTS[0] in inds
    want_bb = IND_OPTS[1] in inds
    want_rsi = IND_OPTS[2] in inds
    want_macd = IND_OPTS[3] in inds
    longest = max([*IP.get("ma", [0]), IP.get("bb_n", 0), IP.get("rsi_n", 0) * 3, IP.get("macd_s", 0) * 3 + IP.get("macd_g", 0), 20])
    base_c = _indicator_base(chosen["code"], start_d, end_d, weekly, longest) if inds else None
    x0 = plot_df.index[0]

    def _disp(sr):  # 가격 단위 → 화면 단위(수익률 모드면 %)
        sr = sr[sr.index >= x0].reindex(plot_df.index)
        return (sr / base0 - 1) * 100 if pct_mode else sr

    n_rows = 2 + int(want_rsi and base_c is not None) + int(want_macd and base_c is not None)
    heights = {2: [0.75, 0.25], 3: [0.62, 0.17, 0.21], 4: [0.54, 0.14, 0.16, 0.16]}[n_rows]
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.03)
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
                x=plot_df.index, y=plot_df["Close"], mode="lines",
                line=dict(color=LINE_COLOR, width=2), name="종가",
                hovertemplate="%{x|%Y-%m-%d}<br>수익률 %{y:+.2f}%<extra></extra>" if pct_mode else "%{x|%Y-%m-%d}<br>종가 %{y:,.0f}<extra></extra>" if domestic
                else "%{x|%Y-%m-%d}<br>종가 %{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
    if base_c is not None:
        if want_ma:
            unit = "주" if weekly else "일"
            for i_ma, n in enumerate(IP["ma"]):
                fig.add_trace(go.Scatter(x=plot_df.index, y=_disp(ta.sma(base_c, n)), mode="lines", name=f"{n}{unit}선",
                                         line=dict(color=ta.ma_color(n, i_ma), width=1.1),
                                         hovertemplate=f"{n}{unit}선 %{{y:,.2f}}<extra></extra>"), row=1, col=1)
        if want_bb:
            mid, upb, lob = ta.bollinger(base_c, IP["bb_n"], IP["bb_k"])
            fig.add_trace(go.Scatter(x=plot_df.index, y=_disp(upb), mode="lines", name="볼린저 상단",
                                     line=dict(color="#A0AEC0", width=1, dash="dot"), hovertemplate="상단 %{y:,.2f}<extra></extra>"), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=_disp(lob), mode="lines", name="볼린저 하단", fill="tonexty",
                                     fillcolor="rgba(160,174,192,0.10)", line=dict(color="#A0AEC0", width=1, dash="dot"),
                                     hovertemplate="하단 %{y:,.2f}<extra></extra>"), row=1, col=1)
            if not want_ma:
                fig.add_trace(go.Scatter(x=plot_df.index, y=_disp(mid), mode="lines", name=f"볼린저 중심({IP['bb_n']})",
                                         line=dict(color=ta.MA_COLORS[20], width=1), hovertemplate="중심 %{y:,.2f}<extra></extra>"), row=1, col=1)
        r_i = 3
        if want_rsi:
            rv = ta.rsi(base_c, IP["rsi_n"])
            rv = rv[rv.index >= x0].reindex(plot_df.index)
            fig.add_trace(go.Scatter(x=plot_df.index, y=rv, mode="lines", name="RSI", line=dict(color="#6B46C1", width=1.3),
                                     hovertemplate="RSI %{y:.1f}<extra></extra>"), row=r_i, col=1)
            for lv, col in ((IP["rsi_hi"], UP_COLOR), (IP["rsi_lo"], DOWN_COLOR)):
                fig.add_hline(y=lv, row=r_i, col=1, line_width=1, line_dash="dot", line_color=col)
            fig.update_yaxes(title_text=f"RSI({IP['rsi_n']})", range=[0, 100], tickvals=sorted({IP["rsi_lo"], 50, IP["rsi_hi"]}), row=r_i, col=1)
            r_i += 1
        if want_macd:
            ml, sg, hist = (x[x.index >= x0].reindex(plot_df.index) for x in ta.macd(base_c, IP["macd_f"], IP["macd_s"], IP["macd_g"]))
            fig.add_trace(go.Bar(x=plot_df.index, y=hist, name="MACD 히스토그램",
                                 marker_color=np.where(hist.fillna(0) >= 0, UP_COLOR, DOWN_COLOR), opacity=0.5,
                                 hovertemplate="히스토그램 %{y:,.2f}<extra></extra>"), row=r_i, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=ml, mode="lines", name="MACD", line=dict(color="#2B6CB0", width=1.3),
                                     hovertemplate="MACD %{y:,.2f}<extra></extra>"), row=r_i, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=sg, mode="lines", name="시그널", line=dict(color="#DD6B20", width=1.1),
                                     hovertemplate="시그널 %{y:,.2f}<extra></extra>"), row=r_i, col=1)
            fig.update_yaxes(title_text=f"MACD({IP['macd_f']},{IP['macd_s']},{IP['macd_g']})", row=r_i, col=1)
    elif inds:
        st.caption("보조지표 계산에 필요한 과거 시세를 불러오지 못했습니다.")
    if "Volume" in plot_df.columns:
        fig.add_trace(
            go.Bar(x=plot_df.index, y=plot_df["Volume"], marker_color=VOLUME_COLOR, name="거래량",
                   hovertemplate="%{x|%Y-%m-%d}<br>거래량 %{y:,.0f}<extra></extra>"),
            row=2, col=1,
        )
    # 봉이 많을수록 간격을 넓혀 봉을 가늘게 (굵고 뭉개져 보이는 현상 방지)
    box_gap = 0.3 if n_bars <= 60 else 0.45 if n_bars <= 150 else 0.6
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider_visible=False)
    if rate_sel:
        add_rates(fig, rate_sel, start_d, end_d, n_rows + 1, domain=list(fig.layout.yaxis.domain))
    fig.update_yaxes(title_text="수익률(%)" if pct_mode else "가격", row=1, col=1, tickformat=",.1f" if pct_mode else ",", ticksuffix="%" if pct_mode else "")
    if pct_mode:
        fig.add_hline(y=0, row=1, col=1, line_width=1, line_dash="dot", line_color="#718096")
    fig.update_yaxes(title_text="거래량", row=2, col=1, tickformat=",")
    fig.update_layout(
        height=560 + 130 * (n_rows - 2), showlegend=bool(base_c is not None and (want_ma or want_bb)), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=10)),
        margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
        boxgap=box_gap, bargap=0.3 if n_bars <= 150 else 0.4,
    )
    st.plotly_chart(fig, use_container_width=True)
    if weekly:
        st.caption("기간이 길어 캔들을 주봉(1주 = 1개)으로 묶어서 표시합니다. 위 지표는 일별 데이터 기준입니다.")

    exp, op = ui.more("상세 통계 더보기 · 변동성 · 샤프비율 · CAPM · VaR", key="single_more_stats")
    if op:
        with exp:
            rf, rf_label = risk_free_for(domestic, df.index[-1].date())
            show_stats(df, rf, rf_label, benchmark_for(chosen.get("market", ""), domestic))
            r_d = df["Close"].dropna().pct_change().dropna()
            if len(r_d) >= 20:
                st.markdown("**VaR · ES (1일 손실률)**")
                risk_grid(risk_metrics(r_d))
                st.caption(RISK_NOTE)
    exp2, op2 = ui.more("일별 시세 표 · CSV 내려받기", key="single_more_data")
    if op2:
        with exp2:
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

    c1, c2, c3 = _row([3, 2, 1])
    selected = c1.multiselect(
        "국내 종목 (이름을 입력하면 검색됩니다)",
        list(options.keys()), key="cmp_selected",
        on_change=_mirror, args=("cmp_selected", "pf_selected"),
    )
    extra = c2.text_input(
        "해외 티커 · 선물 · 종목 추가 (쉼표로 구분)", key="cmp_extra",
        placeholder="예: AAPL, NVDA, 원유 선물, 금 선물",
        on_change=_mirror, args=("cmp_extra", "pf_extra"),
    )
    with c3:
        with ui.popover("설정", icon=":material/tune:"):
            fut_sel = st.multiselect(
                "해외 선물 (목록에서 선택)", list(FUT_OPTIONS), key="cmp_fut",
                on_change=_mirror, args=("cmp_fut", "pf_fut"),
            )
            rate_sel = rate_controls("cmp")
    period = period_input("cmp_period", mirror="pf_period", presets=True)

    # 선택 항목 정리 (중복 제거)
    items: list[dict] = []
    seen: set[str] = set()
    for label in selected:
        code = options[label]
        if code not in seen:
            seen.add(code)
            items.append({"code": code, "name": names[code], "domestic": True})
    for it in _futures_items(fut_sel):
        if it["code"] not in seen:
            seen.add(it["code"])
            items.append(it)
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
                line=dict(color=series_color(i), width=2),
                hovertemplate="%{y:+.2f}%<extra>" + it["name"] + "</extra>",
            )
        )
        rd = s.pct_change().dropna()
        rows.append({
            "종목": it["name"],
            "마지막 종가": fmt_price(float(s.iloc[-1]), it["domestic"]),
            "기간 수익률(%)": round(float(ret.iloc[-1]), 2),
            "연 변동성(%)": round(float(rd.std(ddof=1)) * TRADING_DAYS ** 0.5 * 100, 1) if len(rd) > 2 else None,
            "최대 낙폭(%)": round(float((s / s.cummax() - 1).min()) * 100, 1),
            "기간 내 최고(%)": round(float(ret.max()), 2),
            "기간 내 최저(%)": round(float(ret.min()), 2),
            "시작일": s.index[0].strftime("%Y-%m-%d"),
        })

    if not rows:
        return

    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#718096")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="수익률 (기간 시작일 = 0%)", ticksuffix="%")
    if rate_sel:
        add_rates(fig, rate_sel, start_d, end_d, 2)
    fig.update_layout(
        height=520, hovermode="x unified", font=dict(size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    summary = pd.DataFrame(rows).sort_values("기간 수익률(%)", ascending=False)
    ui.section("기간 수익률 순위")
    from glossary import col_help

    cfg = col_help(summary.columns)
    for c in ("기간 수익률(%)", "기간 내 최고(%)", "기간 내 최저(%)"):
        cfg[c] = st.column_config.NumberColumn(c, format="%+.2f%%")
    cfg["연 변동성(%)"] = st.column_config.NumberColumn("연 변동성(%)", format="%.1f%%")
    cfg["최대 낙폭(%)"] = st.column_config.NumberColumn("최대 낙폭(%)", format="%.1f%%")
    st.dataframe(summary, use_container_width=True, hide_index=True, column_config=cfg)
    st.caption("수익률은 각 종목의 기간 첫 거래일 종가를 0%로 놓고 계산한 값입니다. 배당은 반영하지 않은 가격 기준입니다.")


# ======================================================================
# 탭 3: 포트폴리오 최적화
# ======================================================================
def _port_stats(W: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float):
    ret = W @ mu
    vol = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", W, cov, W), 1e-18))
    return ret, vol, (ret - rf) / vol


def _refine(w: np.ndarray, score_batch, cap: float) -> np.ndarray:
    """한 종목에서 다른 종목으로 비중을 옮기는 모든 경우를 한꺼번에 평가해, 점수가 가장 많이 오르는 이동을 반복한다.
    (공매도 불가, 종목당 상한 cap) 종목 수가 많아도 빠르게 돌도록 벡터화했다."""
    w = w.copy()
    n = len(w)
    best = float(score_batch(w[None, :])[0])
    ii, jj = np.where(~np.eye(n, dtype=bool))
    for step in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        for _ in range(5000):
            amt = np.minimum(step, np.minimum(w[jj], cap - w[ii]))
            ok = amt > 1e-12
            if not ok.any():
                break
            I, J, A = ii[ok], jj[ok], amt[ok]
            C = np.tile(w, (len(I), 1))
            idx = np.arange(len(I))
            C[idx, I] += A
            C[idx, J] -= A
            sc = score_batch(C)
            k = int(np.argmax(sc))
            if sc[k] > best + 1e-12:
                w, best = C[k], float(sc[k])
            else:
                break
    return w


@st.cache_data(show_spinner="최적 비중을 계산하는 중...", max_entries=30, ttl=60 * 60)
def optimize_portfolios(mu: np.ndarray, cov: np.ndarray, rf: float, cap: float, seed: int = 42):
    """무작위 후보 + 비중 이동 탐색으로 최대 샤프 / 최소 분산 포트폴리오를 찾는다. (SciPy 없이 numpy만 사용)"""
    n = len(mu)
    rng = np.random.default_rng(seed)
    per = 15000  # 정확도를 위해 후보를 넉넉히 만든다 (시간이 조금 더 걸림)
    W = np.vstack([rng.dirichlet(np.ones(n) * a, size=per) for a in (0.3, 1.0, 3.0)] + [np.full((1, n), 1.0 / n)])
    W = W[W.max(axis=1) <= cap + 1e-9]
    ret, vol, sh = _port_stats(W, mu, cov, rf)

    def sharpe_f(Wb):
        return _port_stats(Wb, mu, cov, rf)[2]

    def negvar(Wb):
        return -np.einsum("ij,jk,ik->i", Wb, cov, Wb)

    # 한 군데에서만 출발하면 국소 최적에 머물 수 있어, 상위 후보 여러 개에서 각각 다듬은 뒤 가장 좋은 것을 고른다
    def best_of(starts, score):
        cands = [_refine(W[i], score, cap) for i in starts]
        return max(cands, key=lambda x: float(score(x[None, :])[0]))

    k = min(8, len(W))
    w_sh = best_of(np.argsort(-sh)[:k], sharpe_f)
    w_mv = best_of(np.argsort(vol)[:k], negvar)
    return {
        "max_sharpe": w_sh, "min_var": w_mv, "equal": np.full(n, 1.0 / n),
        "cloud": (vol[::6], ret[::6]),
    }


def render_portfolio(listing: pd.DataFrame) -> None:
    ordered = listing.sort_values("Name")
    options = {f"{r.Name} ({r.Code})": (r.Code, r.Name, str(r.Market)) for r in ordered.itertuples()}

    c1, c2, c3 = _row([3, 2, 1])
    selected = c1.multiselect(
        "국내 종목 (2개 이상, 이름을 입력하면 검색됩니다)",
        list(options.keys()), key="pf_selected",
        on_change=_mirror, args=("pf_selected", "cmp_selected"),
        help="수익률 비교 페이지와 종목·기간 입력이 함께 바뀝니다",
    )
    extra = c2.text_input(
        "해외 티커 · 선물 · 종목 추가 (쉼표로 구분)", key="pf_extra",
        placeholder="예: AAPL, NVDA, 원유 선물, 금 선물",
        on_change=_mirror, args=("pf_extra", "cmp_extra"),
    )
    with c3:
        with ui.popover("설정", icon=":material/tune:"):
            fut_sel = st.multiselect(
                "해외 선물 (목록에서 선택)", list(FUT_OPTIONS), key="pf_fut",
                on_change=_mirror, args=("pf_fut", "cmp_fut"),
            )
            pf_set = st.container()
    period = period_input("pf_period", mirror="cmp_period", presets=True)

    items: list[dict] = []
    seen: set[str] = set()
    for label in selected:
        code, name, market = options[label]
        if code not in seen:
            seen.add(code)
            items.append({"code": code, "name": name, "domestic": True, "market": market})
    for it in _futures_items(fut_sel):
        if it["code"] not in seen:
            seen.add(it["code"])
            items.append(it)
    for token in [t.strip() for t in extra.split(",") if t.strip()]:
        cands = find_candidates(token, listing)
        if not cands:
            st.warning(f"'{token}' 종목을 찾지 못했습니다.")
            continue
        c = cands[0]
        if c["code"] not in seen:
            seen.add(c["code"])
            items.append({"code": c["code"], "name": c["name"], "domestic": c["domestic"], "market": c.get("market", "")})

    if len(items) < 2:
        st.info("종목을 2개 이상 선택하세요.")
        return
    if period is None:
        st.info("기간의 시작일과 종료일을 모두 선택하세요.")
        return
    start_d, end_d = period
    n = len(items)

    # 벤치마크: 전부 국내 종목이면 국내 지수, 해외가 섞이면 S&P 500 (직접 바꿀 수 있음)
    bench_opts = {"코스피 지수": "KS11", "코스닥 지수": "KQ11", "S&P 500": "US500"}
    if all(it["domestic"] for it in items):
        default_bench = 1 if all(it["market"].upper().startswith("KOSDAQ") for it in items) else 0
    else:
        default_bench = 2
    with pf_set:
        bench_name = st.selectbox("CAPM 벤치마크(시장 지수)", list(bench_opts), index=default_bench, key="pf_bench")
        cap_pct = st.slider("종목당 최대 비중 (%)", min_value=int(math.ceil(100 / n)), max_value=100, value=100, step=1)
    cap = cap_pct / 100
    bench_code = bench_opts[bench_name]

    names = [it["name"] for it in items]
    if len(names) != len(set(names)):
        st.error("이름이 같은 종목이 있습니다. 종목을 다시 선택하세요.")
        return

    # 가격 불러오기 → 공통 거래일로 정렬
    series = {}
    for it in items + [{"code": bench_code, "name": bench_name}]:
        try:
            d = load_prices(it["code"], start_d.isoformat(), end_d.isoformat())
        except Exception as e:  # noqa: BLE001
            st.error(f"{it['name']}: 데이터를 불러오지 못했습니다 ({e})")
            return
        if d is None or d.empty or "Close" not in d:
            st.error(f"{it['name']}: 선택한 기간에 데이터가 없습니다.")
            return
        series[it["name"]] = d["Close"].dropna()
    P = pd.concat(series, axis=1, join="inner").dropna()
    R = P.pct_change().dropna()
    if len(R) < 30:
        st.warning(f"모든 종목이 겹치는 거래일이 {len(R)}일뿐입니다. 기간을 늘리거나 종목을 바꿔 보세요.")
        return
    Rs, m = R[names], R[bench_name]

    rf, rf_label = risk_free_for(bench_code != "US500", R.index[-1].date())
    mu = Rs.mean().values * TRADING_DAYS
    cov = Rs.cov().values * TRADING_DAYS

    # ---- 최적화 (같은 입력이면 다시 계산하지 않음)
    res = optimize_portfolios(mu, cov, rf, cap)
    ports = {
        "최대 샤프 (최적)": res["max_sharpe"],
        "최소 분산": res["min_var"],
        "동일 비중": res["equal"],
    }

    st.session_state["pf_opt_result"] = {
        "items": [{"code": it["code"], "name": it["name"], "domestic": bool(it["domestic"])} for it in items],
        "weights": [float(x) for x in res["max_sharpe"]],
        "weight_sets": {k: [float(x) for x in w] for k, w in ports.items()},
        "start": f"{P.index[0]:%Y-%m-%d}",
        "prices": {nm: float(P[nm].iloc[-1]) for nm in names},
        "asof": f"{P.index[-1]:%Y-%m-%d}",
    }

    pseries, pstat = {}, {}
    for k, w in ports.items():
        rp = pd.Series(Rs.values @ w, index=Rs.index)
        pseries[k] = rp
        c = capm_stats(rp, m, rf)
        cum = (1 + rp).cumprod()
        pstat[k] = {"c": c, "rk": risk_metrics(rp), "total": float(cum.iloc[-1] - 1),
                    "mdd": float((cum / cum.cummax() - 1).min()),
                    "sharpe": (c["mean_a"] - rf) / c["std_a"] if c["std_a"] > 0 else float("nan")}

    # ---- 핵심 결과: 최대 샤프 포트폴리오
    o, e = pstat["최대 샤프 (최적)"], pstat["동일 비중"]
    ui.section("최적 포트폴리오 (최대 샤프)", f"{P.index[0]:%Y-%m-%d} ~ {P.index[-1]:%Y-%m-%d} · 공통 거래일 {len(R):,}일")
    ui.cards([
        ("연환산 기대수익률", _pct(o["c"]["mean_a"], True), f"동일 비중 {_pct(e['c']['mean_a'], True)}", o["c"]["mean_a"]),
        ("연환산 표준편차", _pct(o["c"]["std_a"]), f"동일 비중 {_pct(e['c']['std_a'])}"),
        ("샤프비율", f"{o['sharpe']:.2f}" if o["sharpe"] == o["sharpe"] else "-",
         f"동일 비중 {e['sharpe']:.2f}" if e["sharpe"] == e["sharpe"] else ""),
        ("최대 낙폭(MDD)", _pct(o["mdd"]), f"동일 비중 {_pct(e['mdd'])}"),
        ("기간 수익률", _pct(o["total"], True), "매일 같은 비중 유지 가정", o["total"]),
        ("베타 (β)", f"{o['c']['beta']:.2f}", bench_name),
        ("젠센 알파 (연)", _pct(o["c"]["alpha"], True), "CAPM 대비 초과수익", o["c"]["alpha"]),
        ("VaR 95% (1일, 과거)", _pct(o["rk"]["var95"]), "하루 손실이 이보다 클 확률 5%"),
    ], cols=4)

    if n > len(R):
        st.warning(
            f"종목 수({n}개)가 공통 거래일 수({len(R)}일)보다 많아 공분산이 불안정합니다. "
            "기간을 늘리거나 종목을 줄이면 결과가 더 믿을 만합니다."
        )

    cw, cf = st.columns([1, 1.35])
    with cw:
        ui.section("투자 비중")
        wdf = pd.DataFrame({"종목": names, **{k: [round(x * 100, 1) for x in w] for k, w in ports.items()}})
        wdf = wdf.sort_values("최대 샤프 (최적)", ascending=False)
        st.dataframe(wdf, use_container_width=True, hide_index=True,
                     column_config={k: st.column_config.ProgressColumn(k, format="%.1f%%", min_value=0, max_value=100) if k == "최대 샤프 (최적)"
                                    else st.column_config.NumberColumn(k, format="%.1f%%") for k in ports})
        st.caption(f"공매도 없이 비중 합 100%, 종목당 최대 {cap_pct}%. 기간이 짧으면 한두 종목에 비중이 쏠릴 수 있습니다.")
    with cf:
        ui.section("위험-수익 평면")
        vol_c, ret_c = res["cloud"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vol_c * 100, y=ret_c * 100, mode="markers", name="무작위 포트폴리오",
                                 marker=dict(size=4, color="#CBD5E0", opacity=0.6), hoverinfo="skip"))
        sd_i = np.sqrt(np.diag(cov))
        fig.add_trace(go.Scatter(x=sd_i * 100, y=mu * 100, mode="markers+text", text=names, textposition="top center",
                                 name="개별 종목", marker=dict(size=9, symbol="diamond", color=SERIES_COLORS[2])))
        markers = {"최대 샤프 (최적)": ("star", 16, SERIES_COLORS[1]), "최소 분산": ("circle", 11, SERIES_COLORS[0]),
                   "동일 비중": ("square", 9, "#718096")}
        for k, w in ports.items():
            sym, size, col = markers[k]
            fig.add_trace(go.Scatter(x=[float(np.sqrt(w @ cov @ w)) * 100], y=[float(w @ mu) * 100], mode="markers", name=k,
                                     marker=dict(size=size, symbol=sym, color=col, line=dict(width=1, color="white"))))
        w_o = ports["최대 샤프 (최적)"]
        s_o, r_o = float(np.sqrt(w_o @ cov @ w_o)), float(w_o @ mu)
        x_max = max(float(vol_c.max()), float(sd_i.max())) * 1.1
        if s_o > 0:
            fig.add_trace(go.Scatter(x=[0, x_max * 100], y=[rf * 100, (rf + (r_o - rf) / s_o * x_max) * 100], mode="lines",
                                     name="자본시장선", line=dict(color=SERIES_COLORS[1], dash="dot", width=1)))
        fig.update_xaxes(title_text="연환산 표준편차(%)", ticksuffix="%")
        fig.update_yaxes(title_text="연환산 기대수익률(%)", ticksuffix="%")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                          legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True)

    # ---- 세부 (더보기)
    exp, op = ui.more("포트폴리오 3종 비교 더보기 · CAPM · VaR · ES", key="pf_more_ports")
    if op:
        with exp:
            prow = {}
            for k, d in pstat.items():
                c, rk = d["c"], d["rk"]
                prow[k] = {
                    "기간 수익률": _pct(d["total"], True),
                    "연환산 기대수익률": _pct(c["mean_a"], True),
                    "연환산 표준편차": _pct(c["std_a"]),
                    "샤프비율": f"{d['sharpe']:.2f}" if d["sharpe"] == d["sharpe"] else "-",
                    "MDD": _pct(d["mdd"]),
                    "베타": f"{c['beta']:.2f}",
                    "상관계수": f"{c['corr']:.2f}",
                    "R²": f"{c['r2']:.2f}",
                    "CAPM 기대수익률": _pct(c["capm"], True),
                    "젠센 알파": _pct(c["alpha"], True),
                    "트레이너": f"{c['treynor']:.3f}" if c["treynor"] == c["treynor"] else "-",
                    "VaR 95% (1일, 과거)": _pct(rk["var95"]),
                    "ES 95% (1일, 과거)": _pct(rk["es95"]),
                    "VaR 99% (1일, 과거)": _pct(rk["var99"]),
                    "ES 99% (1일, 과거)": _pct(rk["es99"]),
                    "VaR 95% (1일, 정규)": _pct(rk["nvar95"]),
                    "VaR 99% (1일, 정규)": _pct(rk["nvar99"]),
                }
            st.dataframe(pd.DataFrame(prow), use_container_width=True)
            st.caption("각 포트폴리오는 위 비중을 기간 내내 매일 유지(일별 리밸런싱)한다고 가정했습니다. 거래비용·세금은 반영하지 않았습니다.")
            st.markdown("**최적 포트폴리오(최대 샤프)의 CAPM**")
            capm_grid(o["c"], bench_name)
            st.caption(capm_note(bench_name))
            st.markdown("**최적 포트폴리오(최대 샤프)의 VaR · ES**")
            risk_grid(o["rk"])
            st.caption(RISK_NOTE)

    exp, op = ui.more("종목별 수치 더보기 · 수익률 · 변동성 · 베타 · 알파 · VaR", key="pf_more_items")
    if op:
        with exp:
            rows = []
            for nm in names:
                r = Rs[nm]
                c = capm_stats(r, m, rf)
                rk = risk_metrics(r)
                px = P[nm]
                mdd = float((px / px.cummax() - 1).min())
                sd = float(r.std(ddof=1)) * TRADING_DAYS ** 0.5
                mean_a = float(r.mean()) * TRADING_DAYS
                rows.append({
                    "종목": nm,
                    "기간 수익률": _pct(float(px.iloc[-1] / px.iloc[0] - 1), True),
                    "연환산 기대수익률": _pct(mean_a, True),
                    "연환산 표준편차": _pct(sd),
                    "샤프비율": f"{(mean_a - rf) / sd:.2f}" if sd > 0 else "-",
                    "베타": f"{c['beta']:.2f}",
                    "상관계수": f"{c['corr']:.2f}",
                    "R²": f"{c['r2']:.2f}",
                    "CAPM 기대수익률": _pct(c["capm"], True),
                    "젠센 알파": _pct(c["alpha"], True),
                    "트레이너": f"{c['treynor']:.3f}" if c["treynor"] == c["treynor"] else "-",
                    "MDD": _pct(mdd),
                    "VaR 95%(1일)": _pct(rk["var95"]),
                    "ES 95%(1일)": _pct(rk["es95"]),
                    "VaR 99%(1일)": _pct(rk["var99"]),
                    "ES 99%(1일)": _pct(rk["es99"]),
                })
            from glossary import col_help

            tdf = pd.DataFrame(rows)
            st.dataframe(tdf, use_container_width=True, hide_index=True, column_config=col_help(tdf.columns))
            st.caption(f"무위험수익률: {rf_label}. 벤치마크: {bench_name}. VaR·ES는 1일 기준 과거 시뮬레이션 손실률입니다.")


# ======================================================================
# 실행
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
    render_single(LISTING)


def page_screener():
    from screener import render_screener

    render_screener()


def page_compare():
    render_compare(LISTING)


def page_portfolio():
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
