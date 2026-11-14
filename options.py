"""옵션 탭
1) 옵션 체인: 미국 상장 종목·ETF의 콜/풋 시장가, 블랙-숄즈-머튼 이론가, 내재변동성, 역사적 변동성, 그릭스(마우스 오버 팝업)
2) 델타·감마 헷지: 최적 포트폴리오(또는 직접 입력한 보유 주식) + 옵션 포지션의 그릭스를 구하고, 델타와 감마를 동시에 0으로 만드는 헷지 수량 계산
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


def render_options() -> None:
    if yf is None:
        st.error("yfinance 패키지가 설치되지 않았습니다. requirements.txt에 yfinance를 추가해 주세요.")
        return
    st.caption("미국 상장 종목·ETF 옵션 (Yahoo Finance, 약 15분 지연). 국내 개별주식옵션은 무료 데이터가 없어 지원하지 않습니다. "
               "조회는 느릴 수 있으며, 실패하면 자동으로 여러 번 다시 시도합니다.")
    from capm_opt import render_capm_opt

    from optvol import render_vol

    from american import render_american

    (t1, t1_open), (t4, t4_open), (t5, t5_open), (t2, t2_open), (t3, t3_open) = ui.lazy_tabs(["옵션 체인", "변동성·수급 분석", "아메리칸 옵션", "델타·감마 헷지", "옵션 포함 CAPM·손익"], key="opt_tabs")
    if t1_open:
        with t1:
            render_chain()
    if t4_open:
        with t4:
            render_vol()
    if t5_open:
        with t5:
            render_american()
    if t2_open:
        with t2:
            render_hedge()
    if t3_open:
        with t3:
            render_capm_opt()
