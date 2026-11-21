"""옵션 페이지 — 맨 위 공통 바(티커 한 번 입력)와 탭 4개
1) 체인·전략: 콜/풋 시장가·BSM 이론가·IV·그릭스(간단히/전체 보기) + 전략 만들기(템플릿·손익·이익 확률·합성 백테스트)
2) 내 포지션: 주식·미국 옵션·국내 옵션(KRX CSV·직접 입력) 장부 → 헷지 / 손익·위험 / 헷지 시뮬레이션
3) 변동성: 기간구조·스마일·곡면·스큐·GEX, 모형 보정, 위험중립 분포
4) 계산기: 유럽식(BSM)·미국식(이항트리) 단일 옵션
데이터: 옵션 = Yahoo Finance(yfinance, 약 15분 지연), 기초자산 가격 = FinanceDataReader, 무위험금리 = FRED 3개월 국채.
"""
# 파일 구성: options_data.py(데이터·계산) · options_chain.py(옵션 체인) · hedge.py(델타·감마 헷지) · 이 파일(탭 묶기)
import html
import io
import math
import re
import time
import urllib.request
from datetime import datetime, timedelta

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import streamlit as st

import ui
import streamlit.components.v1 as components

import bsm

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None

try:
    from zoneinfo import ZoneInfo

    NY = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001
    NY = None

from options_data import *  # noqa: F401,F403
from options_chain import *  # noqa: F401,F403
from hedge import *  # noqa: F401,F403


def _ticker_bar() -> str:
    """모든 탭이 같이 쓰는 티커 (opt_ticker). 국내 옵션은 '내 포지션' 탭에서 CSV·직접 입력."""
    ss = st.session_state
    ss["_opt_shared"] = True
    pf = _pf_us_tickers()
    presets = list(dict.fromkeys(pf + OPT_PRESETS))[:10]
    c1, c2, c3 = st.columns([3.2, 1.6, 1])
    with c1:
        if hasattr(st, "pills"):
            st.pills("빠른 선택", presets, key="opt_pill", label_visibility="collapsed",
                     on_change=lambda: _set_ticker(ss.get("opt_pill")) if ss.get("opt_pill") else None)
    c2.text_input("미국 티커", key="opt_ticker", placeholder="AAPL", label_visibility="collapsed")
    fxv = load_usdkrw()
    ss["_opt_fx"] = fxv[0] if fxv else 1400.0
    c3.toggle("원화", key="opt_krw", help=f"달러 금액을 원화로 표시 (원/달러 {ss['_opt_fx']:,.1f})")
    return (ss.get("opt_ticker") or "").strip().upper()


def render_options() -> None:
    if yf is None:
        st.error("yfinance 패키지가 설치되지 않았습니다. requirements.txt에 yfinance를 추가해 주세요.")
        return
    from american import render_american
    from optbook import render_book
    from optvol import render_vol

    ticker = _ticker_bar()
    st.caption("미국 옵션은 Yahoo Finance(약 15분 지연). 국내 옵션은 '내 포지션' 탭에서 KRX CSV를 올리거나 직접 입력합니다.")
    (t1, o1), (t2, o2), (t3, o3), (t4, o4) = ui.lazy_tabs(["체인·전략", "내 포지션", "변동성", "계산기"], key="opt_tabs4")
    if o1:
        with t1:
            render_chain()
    if o2:
        with t2:
            render_book(ticker)
    if o3:
        with t3:
            render_vol()
            if ticker and is_us_ticker(ticker):
                exp, op = ui.more("옵션이 말하는 주가 분포 (위험중립, Breeden–Litzenberger)", key="opt_more_rnd")
                if op:
                    with exp:
                        import hedgesim

                        with st.spinner("만기별 체인을 불러오는 중..."):
                            P, chains = hedgesim._chains_for(ticker, n=8)
                        if chains:
                            hedgesim.render_rnd(ticker, chains, P["S"], P["r"], P["q"])
    if o4:
        with t4:
            render_american()
