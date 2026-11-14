"""종목 목록·주가 불러오기, 해외 선물 목록, 종목 검색, 가격 표시."""
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

import ui  # 화면 공통 요소 (ui.py)
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


__all__ = [n for n in dir() if not n.startswith("__")]  # 밑줄로 시작하는 도우미 함수도 다른 모듈에서 쓸 수 있게
