"""기간 입력 (빠른 선택 버튼 · 날짜 · 프리셋), 페이지 간 기간 동기화."""
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


__all__ = [n for n in dir() if not n.startswith("__")]  # 밑줄로 시작하는 도우미 함수도 다른 모듈에서 쓸 수 있게
