"""종목 스크리너 (조건검색): Finviz·증권사 조건검색처럼 시세·기술적 조건으로 종목을 걸러 본다.

- 국내: KRX 전 종목 시세(시가총액·등락률·거래대금) + 시가총액 상위 종목의 1년 일봉으로 계산한 기술적 지표
- 미국: S&P 500 구성 종목의 1년 일봉으로 계산한 기술적 지표
기술적 지표는 Yahoo 일봉을 한 번에 받아 계산하고 1시간 동안 재사용한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import ta
import ui

MA_KEYS = ["none", "above_s", "below_s", "above_m", "above_l", "below_l", "aligned", "gc", "dc"]
TECH_DEFAULTS = {"sc_rsi_n": 14, "sc_ma_s": 20, "sc_ma_m": 60, "sc_ma_l": 200, "sc_vol_n": 20, "sc_hv_n": 60}


def _ma_label(k: str) -> str:
    s_, m_, l_ = (st.session_state.get(x, TECH_DEFAULTS[x]) for x in ("sc_ma_s", "sc_ma_m", "sc_ma_l"))
    return {"none": "상관없음", "above_s": f"{s_}일선 위", "below_s": f"{s_}일선 아래", "above_m": f"{m_}일선 위",
            "above_l": f"{l_}일선 위", "below_l": f"{l_}일선 아래", "aligned": f"정배열 ({s_} > {m_} > {l_}일선)",
            "gc": f"최근 5일 골든크로스 ({s_}일선이 {m_}일선 상향 돌파)",
            "dc": f"최근 5일 데드크로스 ({s_}일선이 {m_}일선 하향 돌파)"}[k]


_BASE = {"sc_rsi": (0, 100), "sc_ma": "none", "sc_dd": 100, "sc_volx": 0.0, "sc_r1m": (-50, 200)}
PRESETS = {
    "52주 신고가 근처": {**_BASE, "sc_dd": 5},
    "과매도 (RSI 30 이하)": {**_BASE, "sc_rsi": (0, 30)},
    "과매수 (RSI 70 이상)": {**_BASE, "sc_rsi": (70, 100)},
    "정배열 상승추세": {**_BASE, "sc_ma": "aligned"},
    "골든크로스": {**_BASE, "sc_ma": "gc"},
    "거래량 급증 (2배 이상)": {**_BASE, "sc_volx": 2.0},
}


def _apply_preset() -> None:
    p = PRESETS.get(st.session_state.get("sc_preset"))
    if p:
        for k, v in p.items():
            st.session_state[k] = v


def _reset() -> None:
    for k, v in {**_BASE, "sc_preset": None}.items():
        st.session_state[k] = v


def _reset_params() -> None:
    for k, v in TECH_DEFAULTS.items():
        st.session_state[k] = v


# ----------------------------------------------------------------------
# 데이터
# ----------------------------------------------------------------------
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def us_universe() -> pd.DataFrame:
    import FinanceDataReader as fdr

    d = fdr.StockListing("S&P500")
    if "Symbol" not in d.columns and "Code" in d.columns:
        d = d.rename(columns={"Code": "Symbol"})
    cols = [c for c in ("Symbol", "Name", "Sector", "Industry") if c in d.columns]
    d = d[cols].dropna(subset=["Symbol"]).copy()
    d["Symbol"] = d["Symbol"].astype(str).str.replace(".", "-", regex=False)  # BRK.B → BRK-B (Yahoo 표기)
    from sectors import GICS_IND_KO, GICS_KO

    if "Sector" not in d.columns:
        d["Sector"] = "분류 없음"
    d["SectorEn"] = d["Sector"].astype(str)
    d["IndustryEn"] = d["Industry"].astype(str) if "Industry" in d.columns else d["SectorEn"]
    d["Sector"] = d["Sector"].map(lambda x: GICS_KO.get(str(x), str(x) if x == x else "분류 없음"))
    if "Industry" not in d.columns:
        d["Industry"] = d["Sector"]
    d["Industry"] = d["Industry"].map(lambda x: GICS_IND_KO.get(str(x), str(x)))
    return d.reset_index(drop=True)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _download(symbols: tuple, period: str):
    """(종가, 거래량) 일봉. 여러 종목을 한 번에 받는다."""
    import yfinance as yf

    d = yf.download(list(symbols), period=period, interval="1d", auto_adjust=True, progress=False, threads=True)
    if d is None or d.empty:
        return pd.DataFrame(), None
    if isinstance(d.columns, pd.MultiIndex):
        close = d["Close"]
        vol = d["Volume"] if "Volume" in d.columns.get_level_values(0) else None
    else:  # 심볼이 하나뿐일 때
        close = d[["Close"]].rename(columns={"Close": symbols[0]})
        vol = d[["Volume"]].rename(columns={"Volume": symbols[0]}) if "Volume" in d.columns else None
    return close, vol


@st.cache_data(ttl=60 * 60, show_spinner=False)
def tech_table(symbols: tuple, rsi_n: int = 14, ma_s: int = 20, ma_m: int = 60, ma_l: int = 200, vol_n: int = 20,
               hv_n: int = 60) -> pd.DataFrame:
    """심볼별 기술적 지표. 기간을 바꿔도 시세는 다시 받지 않고 계산만 새로 한다."""
    if not symbols:
        return pd.DataFrame()
    period = "2y" if max(ma_l, ma_m, ma_s) > 230 else "1y"  # 긴 이평은 1년 일봉으로 부족
    close, vol = _download(symbols, period)
    if close is None or close.empty:
        return pd.DataFrame()
    rows = {}
    for sym in close.columns:
        c = close[sym].dropna()
        if len(c) < 30:
            continue
        last = float(c.iloc[-1])
        c1y = c[c.index >= c.index[-1] - pd.Timedelta(days=365)]
        mas, mam, mal = (ta.sma(c, n) for n in (ma_s, ma_m, ma_l))
        fin = lambda x: np.isfinite(x.iloc[-1])  # noqa: E731
        diff = (mas - mam).dropna()
        cross = np.sign(diff)
        recent = cross.iloc[-6:] if len(cross) >= 6 else cross
        gc = bool(len(recent) >= 2 and (recent.diff() > 0).iloc[1:].any() and recent.iloc[-1] > 0)
        dc = bool(len(recent) >= 2 and (recent.diff() < 0).iloc[1:].any() and recent.iloc[-1] < 0)
        r = c.pct_change().dropna()
        v_ratio = np.nan
        if vol is not None and sym in vol.columns:
            v = vol[sym].dropna()
            if len(v) > vol_n + 1 and float(v.iloc[-vol_n - 1:-1].mean()) > 0:
                v_ratio = float(v.iloc[-1]) / float(v.iloc[-vol_n - 1:-1].mean())

        def ago(n):
            return float(c.iloc[-1] / c.iloc[-1 - n] - 1) * 100 if len(c) > n else np.nan

        rows[sym] = {
            "last": last,
            "r1m": ago(21), "r3m": ago(63), "r1y": float(c1y.iloc[-1] / c1y.iloc[0] - 1) * 100,
            "rsi": float(ta.rsi(c, rsi_n).iloc[-1]),
            "gap_s": (last / float(mas.iloc[-1]) - 1) * 100 if fin(mas) else np.nan,
            "above_s": bool(last > mas.iloc[-1]) if fin(mas) else None,
            "above_m": bool(last > mam.iloc[-1]) if fin(mam) else None,
            "above_l": bool(last > mal.iloc[-1]) if fin(mal) else None,
            "aligned": bool(mas.iloc[-1] > mam.iloc[-1] > mal.iloc[-1]) if fin(mal) and fin(mam) and fin(mas) else None,
            "gc": gc, "dc": dc,
            "dd52": (last / float(c1y.max()) - 1) * 100,
            "up52": (last / float(c1y.min()) - 1) * 100,
            "vol_ratio": v_ratio,
            "hv": float(r.iloc[-hv_n:].std(ddof=1)) * np.sqrt(252) * 100 if len(r) > max(hv_n // 3, 10) else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _kr_universe(top_n: int) -> pd.DataFrame:
    from market import _sector_map, load_krx_snapshot

    s = load_krx_snapshot().copy()
    s["Code"] = s["Code"].astype(str).str.zfill(6)
    s = s[s["Market"].astype(str).str.upper().str.match(r"^(KOSPI|KOSDAQ)")]
    try:
        s = s.merge(_sector_map(), on="Code", how="left")
    except Exception:  # noqa: BLE001
        s["Sector"] = np.nan
    s["Sector"] = s["Sector"].fillna("분류 없음")
    if "Industry" not in s.columns:
        s["Industry"] = s["Sector"]
    s["Industry"] = s["Industry"].fillna(s["Sector"])
    s = s.dropna(subset=["Marcap"]).sort_values("Marcap", ascending=False)
    s["Symbol"] = s["Code"] + np.where(s["Market"].astype(str).str.upper().str.startswith("KOSPI"), ".KS", ".KQ")
    return s.head(top_n)


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
_go = ui.go  # 다른 페이지로 넘어가면서 입력칸 채우기 (ui.py)


def render_screener() -> None:
    try:
        c1, c2, c3 = st.columns([1.6, 1.2, 1], vertical_alignment="bottom")
    except TypeError:
        c1, c2, c3 = st.columns([1.6, 1.2, 1])
    with c1:
        uni = ui.seg("시장", ["국내 (KRX)", "미국 (S&P 500)"], default="국내 (KRX)", key="sc_uni") or "국내 (KRX)"
    domestic = uni.startswith("국내")
    with c2:
        top_n = st.selectbox("검색 대상", [200, 400, 800], index=1, key="sc_topn", format_func=lambda x: f"시가총액 상위 {x}종목",
                             disabled=not domestic, label_visibility="collapsed")
    with c3:
        st.button("조건 초기화", on_click=_reset, use_container_width=True, key="sc_reset_btn")
    if hasattr(st, "pills"):
        st.pills("빠른 조건", list(PRESETS), key="sc_preset", on_change=_apply_preset, label_visibility="collapsed")

    # ---- 기본 조건
    try:
        if domestic:
            U = _kr_universe(int(top_n))
        else:
            U = us_universe()
            U = U.assign(Code=U["Symbol"])
    except Exception as e:  # noqa: BLE001
        st.error(f"종목 목록을 불러오지 못했습니다: {e}")
        return
    if U.empty:
        st.warning("종목 목록이 비어 있습니다. 잠시 후 다시 시도해 주세요.")
        return

    with ui.popover("기본 조건 (시장 · 업종 · 규모)", icon=":material/filter_alt:"):
        secs = sorted(U["Sector"].dropna().astype(str).unique())
        for _k in (f"sc_sec_{'kr' if domestic else 'us'}", f"sc_ind_{'kr' if domestic else 'us'}"):  # 예전 분류 이름 정리
            if isinstance(st.session_state.get(_k), list):
                _ok = set(secs) if "_sec_" in _k else set(U["Industry"].dropna().astype(str)) if "Industry" in U.columns else set()
                st.session_state[_k] = [x for x in st.session_state[_k] if x in _ok]
        sec_sel = st.multiselect("섹터", secs, key=f"sc_sec_{'kr' if domestic else 'us'}", placeholder="전체 섹터")
        pool = U[U["Sector"].astype(str).isin(sec_sel)] if sec_sel else U
        inds = sorted(pool["Industry"].dropna().astype(str).unique()) if "Industry" in U.columns else []
        ind_sel = st.multiselect("업종", inds, key=f"sc_ind_{'kr' if domestic else 'us'}", placeholder="전체 업종")
        if domestic:
            mk_sel = st.multiselect("시장 구분", ["KOSPI", "KOSDAQ"], default=["KOSPI", "KOSDAQ"], key="sc_mk")
            cap_min = st.number_input("시가총액 최소 (억원)", value=1000, step=500, min_value=0, key="sc_capmin")
            amt_min = st.number_input("오늘 거래대금 최소 (억원)", value=0, step=10, min_value=0, key="sc_amtmin")
            chg_rng = st.slider("오늘 등락률 (%)", -30.0, 30.0, (-30.0, 30.0), step=0.5, key="sc_chg")

    with ui.popover("지표 기간 설정", icon=":material/tune:"):
        st.caption("기본값이 채워져 있습니다. 지우고 원하는 숫자를 넣으면 그 기간으로 다시 계산합니다.")
        a, b = st.columns(2)
        rsi_n = int(a.number_input("RSI 기간 (일)", 2, 200, TECH_DEFAULTS["sc_rsi_n"], step=1, key="sc_rsi_n"))
        vol_n = int(b.number_input("거래량 평균 기간 (일)", 2, 120, TECH_DEFAULTS["sc_vol_n"], step=1, key="sc_vol_n"))
        a, b, c = st.columns(3)
        ma_s = int(a.number_input("단기 이평 (일)", 2, 400, TECH_DEFAULTS["sc_ma_s"], step=1, key="sc_ma_s"))
        ma_m = int(b.number_input("중기 이평 (일)", 2, 400, TECH_DEFAULTS["sc_ma_m"], step=1, key="sc_ma_m"))
        ma_l = int(c.number_input("장기 이평 (일)", 2, 400, TECH_DEFAULTS["sc_ma_l"], step=1, key="sc_ma_l"))
        hv_n = int(st.number_input("변동성 계산 기간 (일)", 10, 250, TECH_DEFAULTS["sc_hv_n"], step=5, key="sc_hv_n"))
        st.button("기본값으로 되돌리기", key="sc_params_reset", on_click=_reset_params)
    if not (ma_s < ma_m < ma_l):
        ma_s, ma_m, ma_l = sorted((ma_s, ma_m, ma_l))
        st.caption(f"이동평균 기간은 단기 < 중기 < 장기 순서로 맞춰 {ma_s}·{ma_m}·{ma_l}일로 계산합니다.")

    if st.session_state.get("sc_ma") not in MA_KEYS:  # 예전 버전에서 남은 값 정리
        st.session_state.pop("sc_ma", None)
    with ui.popover("기술적 조건 (RSI · 이동평균 · 52주 · 거래량)", icon=":material/insights:"):
        rsi_rng = st.slider(f"RSI ({rsi_n}일)", 0, 100, (0, 100), key="sc_rsi", help="30 이하 과매도, 70 이상 과매수로 흔히 봅니다")
        ma_rule = st.selectbox("이동평균선 조건", MA_KEYS, key="sc_ma", format_func=_ma_label)
        dd_max = st.slider("52주 최고가 대비 하락 폭 최대 (%)", 0, 100, 100, key="sc_dd",
                           help="5로 두면 52주 최고가에서 5% 이내인 종목(신고가 근처)만")
        r1m_rng = st.slider("1개월 수익률 (%)", -50, 200, (-50, 200), key="sc_r1m")
        volx = st.number_input(f"거래량: {vol_n}일 평균 대비 최소 배수 (0 = 상관없음)", value=0.0, step=0.5, min_value=0.0, key="sc_volx")

    D = U.copy()
    if sec_sel:
        D = D[D["Sector"].astype(str).isin(sec_sel)]
    if ind_sel:
        D = D[D["Industry"].astype(str).isin(ind_sel)]
    if domestic:
        if mk_sel:
            D = D[D["Market"].astype(str).str.upper().str.startswith(tuple(mk_sel))]
        D = D[D["Marcap"] >= cap_min * 1e8]
        if amt_min:
            D = D[D["Amount"].fillna(0) >= amt_min * 1e8]
        D = D[(D["ChangesRatio"] >= chg_rng[0]) & (D["ChangesRatio"] <= chg_rng[1])]

    # ---- 기술적 지표 (대상 전체를 한 번에 받아 캐시)
    with st.spinner(f"{len(U)}개 종목의 1년 일봉으로 기술적 지표를 계산하는 중... (처음 한 번 10~30초, 이후 1시간 재사용)"):
        try:
            T = tech_table(tuple(U["Symbol"]), rsi_n, ma_s, ma_m, ma_l, vol_n, hv_n)
        except Exception as e:  # noqa: BLE001
            T = pd.DataFrame()
            st.warning(f"기술적 지표를 계산하지 못해 기본 조건만 적용했습니다: {e}")
    if not T.empty:
        D = D.merge(T, left_on="Symbol", right_index=True, how="inner")
        D = D[(D["rsi"] >= rsi_rng[0]) & (D["rsi"] <= rsi_rng[1])]
        rule = None if ma_rule == "none" else (D[ma_rule.replace("below_", "above_")] == (not ma_rule.startswith("below_")))
        if rule is not None:
            D = D[rule]
        if dd_max < 100:
            D = D[D["dd52"] >= -dd_max]
        D = D[(D["r1m"].fillna(0) >= r1m_rng[0]) & (D["r1m"].fillna(0) <= r1m_rng[1])]
        if volx > 0:
            D = D[D["vol_ratio"] >= volx]

    # ---- 결과
    if D.empty:
        ui.section("검색 결과 0종목", f"대상 {len(U):,}종목 중")
        st.info("조건에 맞는 종목이 없습니다. 조건을 넓혀 보세요.")
        return
    store = st.session_state.get("sc_scores", {})
    scores = {s: _total(store[(domestic, s)]) for s in D["Symbol"].astype(str)
              if (domestic, s) in store and not store[(domestic, s)].get("err")}
    (t1, o1), (t2, o2) = ui.lazy_tabs(["검색 결과", "점수 순위 TOP 20"], key="sc_res_tabs")
    if o2:
        with t2:
            _render_scores(D, domestic)
    if not o1:
        return
    with t1:
        _results(D, U, T, domestic, scores, rsi_n, ma_s, vol_n, hv_n)


def _results(D, U, T, domestic, scores, rsi_n, ma_s, vol_n, hv_n) -> None:
    ui.section(f"검색 결과 {len(D):,}종목", f"대상 {len(U):,}종목 중 · 머리글을 누르면 정렬 · 행을 고르면 차트·리서치로 이동")
    if domestic:
        out = pd.DataFrame({
            "종목": D["Name"], "코드": D["Code"], "섹터": D["Sector"], "업종": D["Industry"] if "Industry" in D.columns else D["Sector"],
            "현재가": D["Close"], "등락률(%)": D["ChangesRatio"], "시가총액(억원)": D["Marcap"] / 1e8,
            "거래대금(억원)": D["Amount"] / 1e8 if "Amount" in D.columns else np.nan,
        })
    else:
        out = pd.DataFrame({"종목": D["Name"], "코드": D["Symbol"], "섹터": D["Sector"], "업종": D["Industry"] if "Industry" in D.columns else D["Sector"], "현재가": D.get("last", np.nan)})
    if not T.empty:
        out = out.assign(**{"1개월(%)": D["r1m"], "3개월(%)": D["r3m"], "1년(%)": D["r1y"], f"RSI({rsi_n})": D["rsi"],
                            f"{ma_s}일선 이격(%)": D["gap_s"], "52주 고점 대비(%)": D["dd52"], "거래량 배수": D["vol_ratio"],
                            "변동성(%)": D["hv"]})
    if scores:
        out.insert(2, "점수", [round(scores[s][0]) if s in scores else np.nan for s in D["Symbol"].astype(str)])
    out = out.reset_index(drop=True)
    for c in ("시가총액(억원)", "거래대금(억원)"):
        if c in out.columns:
            out[c] = out[c].round(0)
    if domestic:
        out["현재가"] = out["현재가"].round(0)
    pct = st.column_config.NumberColumn
    cfg = {"점수": st.column_config.ProgressColumn("점수", min_value=0, max_value=100, format="%d",
                                                 help="'점수 순위' 탭에서 계산한 6개월 종합 점수 (0~100, 50 = 중립)"),
           "현재가": pct("현재가", format="localized" if domestic else "%.2f"),
           "등락률(%)": pct("등락률(%)", format="%+.2f"), "시가총액(억원)": pct("시가총액(억원)", format="localized"),
           "거래대금(억원)": pct("거래대금(억원)", format="localized"),
           "1개월(%)": pct("1개월(%)", format="%+.1f"), "3개월(%)": pct("3개월(%)", format="%+.1f"),
           "1년(%)": pct("1년(%)", format="%+.1f"),
           f"RSI({rsi_n})": pct(f"RSI({rsi_n})", format="%.0f", help=f"{rsi_n}일 RSI. 30 이하 과매도, 70 이상 과매수"),
           f"{ma_s}일선 이격(%)": pct(f"{ma_s}일선 이격(%)", format="%+.1f", help=f"현재가가 {ma_s}일 이동평균보다 몇 % 위(+)·아래(−)인지"),
           "52주 고점 대비(%)": pct("52주 고점 대비(%)", format="%+.1f", help="0에 가까울수록 52주 신고가 근처"),
           "거래량 배수": pct("거래량 배수", format="%.1f배", help=f"오늘 거래량 ÷ 최근 {vol_n}일 평균. 2배 이상이면 관심이 몰린 것"),
           "변동성(%)": pct("변동성(%)", format="%.0f", help=f"최근 {hv_n}일 일별 수익률의 연환산 표준편차")}
    kw = dict(use_container_width=True, hide_index=True, column_config=cfg, height=min(38 * (len(out) + 1) + 4, 620))
    i = ui.table_pick(out, "sc_table", **kw)
    if i is not None:
        r = out.iloc[i]
        q = str(r["종목"]) if domestic else str(r["코드"])
        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button(f"{r['종목']} 차트 보기", key="sc_go_chart", use_container_width=True):
            _go("chart", "single_query", q)
        if b2.button("기업 리서치에서 보기", key="sc_go_rs", use_container_width=True):
            _go("research", "rs_ticker", str(r["코드"]))
    st.download_button("결과 CSV 내려받기", out.to_csv(index=False).encode("utf-8-sig"), file_name="screener.csv", mime="text/csv")
    st.caption("국내는 KRX 당일 시세(장중엔 지연)와 Yahoo 일봉, 미국은 Yahoo 일봉 기준입니다. 기술적 지표는 1시간마다 새로 계산합니다. "
               "조건에 맞는다는 것이 매수·매도 신호를 뜻하지는 않습니다."
               + ("" if scores else " 매수·매도 점수는 '점수 순위' 탭에서 계산하면 이 표에도 붙습니다."))


# ----------------------------------------------------------------------
# 매수·매도 점수 순위
# ----------------------------------------------------------------------
SCORE_COLS = {"abs": "CAPM 미포함", "ind": "업종 평균 대비", "capm": "CAPM 대비"}


def _score_jobs(D: pd.DataFrame, domestic: bool) -> list:
    jobs = []
    for _, r in D.iterrows():
        sym = str(r["Symbol"])
        if domestic:
            ksec = " ".join(str(x) for x in (r.get("Sector"), r.get("Industry")) if x == x and x and x != "분류 없음")
            jobs.append(dict(sym=sym, name=str(r["Name"]), code=str(r["Code"]), sector="", industry="", ksec=ksec))
        else:
            jobs.append(dict(sym=sym, name=str(r["Name"]), code=sym, sector=str(r.get("SectorEn") or ""),
                             industry=str(r.get("IndustryEn") or ""), ksec=""))
    return jobs


def _run_scores(jobs: list, domestic: bool) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import outlook

    ss = st.session_state
    store = ss.setdefault("sc_scores", {})
    dkey = None
    if domestic:
        try:
            import kr

            dkey = kr.dart_key()
        except Exception:  # noqa: BLE001
            dkey = None
    bar = st.progress(0.0, text="준비 중...")
    if not domestic:  # 같은 업종 대표 종목 사례 (업종 유형마다 한 번, 7일 재사용)
        profs = sorted({outlook.screen_profile(False, j["sector"], j["industry"]) for j in jobs})
        for i, p in enumerate(profs):
            bar.progress(0.0, text=f"업종별 기준 사례 준비 중 ({i + 1}/{len(profs)}: {outlook.PROFILES[p]['name']}) — 처음 한 번만 1~2분씩 걸립니다")
            try:
                outlook.validate_profile(p)
            except Exception:  # noqa: BLE001
                pass
    t0, done, ok = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(outlook.screen_score, j["sym"], domestic, j["sector"], j["industry"], j["ksec"], dkey): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            done += 1
            try:
                store[(domestic, j["sym"])] = dict(j, **f.result(), err=None)
                ok += 1
            except Exception as e:  # noqa: BLE001
                store[(domestic, j["sym"])] = dict(j, err=str(e)[:60])
            el = time.time() - t0
            eta = el / done * (len(jobs) - done)
            bar.progress(done / len(jobs), text=f"{done}/{len(jobs)}종목 계산 (성공 {ok}) · 남은 시간 약 {eta / 60:.0f}분 {eta % 60:.0f}초")
    bar.empty()


def _total(r: dict):
    """점수 3개(CAPM 미포함 · 업종 평균 대비 · CAPM 대비)의 평균 → (점수, 80% 하단, 상단, 좁은지)."""
    import outlook

    mid = float(np.mean([r[m][0] for m in SCORE_COLS]))
    lo = float(np.mean([r[m][1] for m in SCORE_COLS]))
    hi = float(np.mean([r[m][2] for m in SCORE_COLS]))
    return mid, lo, hi, hi - lo <= outlook.RELIABLE_WIDTH


def _render_scores(D: pd.DataFrame, domestic: bool) -> None:
    import outlook

    ss = st.session_state
    jobs = _score_jobs(D, domestic)
    store = ss.get("sc_scores", {})
    have = [store[(domestic, j["sym"])] for j in jobs if (domestic, j["sym"]) in store]
    todo = [j for j in jobs if (domestic, j["sym"]) not in store]
    est = max(1, round(len(todo or jobs) * (6 if domestic else 4) / 3 / 60))
    if not have:
        with st.container(border=True):
            st.markdown(f"**검색 결과 {len(jobs):,}종목의 매수·매도 점수를 매겨 상위 20종목을 보여줍니다**")
            st.caption(f"종목마다 10년치 주가·재무제표로 '6개월 뒤 오를지'를 판정합니다. 약 {est}분 걸리고, 그동안 이 화면을 그대로 두세요."
                       + (" 조건을 좁혀 종목 수를 줄이면 빨라집니다." if len(jobs) > 150 else ""))
            go_ = st.button("점수 계산 시작", key="sc_score_btn", type="primary")
    else:
        n_ok = sum(1 for r in have if not r.get("err"))
        c1, c2 = st.columns([3, 1])
        c1.caption(f"{n_ok:,}종목 점수 계산됨" + (f" · 조건을 바꿔 새로 걸린 {len(todo):,}종목은 아직 (약 {est}분)" if todo else ""))
        go_ = c2.button(f"{len(todo):,}종목 추가 계산" if todo else "다시 계산", key="sc_score_btn",
                        type="primary" if todo else "secondary", use_container_width=True)
    if go_:
        if not todo:  # 다시 계산 (6시간 안이면 저장된 계산을 다시 써서 빠름)
            for j in jobs:
                ss.get("sc_scores", {}).pop((domestic, j["sym"]), None)
            todo = jobs
        _run_scores(todo, domestic)
        st.rerun()
    if not have:
        return
    good = [r for r in have if not r.get("err")]
    if not good:
        st.warning(f"{len(have)}종목 모두 점수를 내지 못했습니다 (재무제표·주가 이력 부족 등).")
        return
    detail = st.toggle("세부 점수 보기", key="sc_score_detail", help="종합 점수를 만든 세 점수와 80% 구간")
    rows = []
    for r in good:
        mid, lo, hi, rel = _total(r)
        row = {"종목": r["name"], "코드": r["code"], "업종 유형": outlook.PROFILES[r["prof"]]["name"], "종합 점수": round(mid),
               "판정": outlook.score_label(mid)}
        if detail:
            row.update({SCORE_COLS[m]: round(r[m][0]) for m in SCORE_COLS})
            row["80% 구간"] = f"{lo:.0f}–{hi:.0f}"
        row["위험 신호"] = ", ".join(r["flags"]) if r["flags"] else ""
        row["_k"] = mid
        rows.append(row)
    top = pd.DataFrame(rows).sort_values("_k", ascending=False).head(20).drop(columns="_k").reset_index(drop=True)
    top.insert(0, "순위", range(1, len(top) + 1))
    nc = st.column_config.NumberColumn
    cfg = {"순위": nc("순위", format="%d", width="small"),
           "종합 점수": st.column_config.ProgressColumn("종합 점수", min_value=0, max_value=100, format="%d점",
                                                     help="50 = 중립, 55 이상 약한 매수, 65 이상 매수, 45 이하 약한 매도, 35 이하 매도"),
           "위험 신호": st.column_config.TextColumn("위험 신호", help="부실 위험 등이 켜지면 점수 상한이 50점")}
    cfg.update({v: nc(v, format="%d") for v in SCORE_COLS.values()})
    kw = dict(use_container_width=True, hide_index=True, column_config=cfg, height=min(36 * (len(top) + 1) + 4, 760))
    i = ui.table_pick(top, "sc_score_table", **kw)
    if i is not None:
        r = top.iloc[i]
        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button(f"{r['종목']} 기업 리서치", key="sc_score_go", type="primary", use_container_width=True):
            _go("research", "rs_ticker", str(r["코드"]))
        if b2.button("차트 보기", key="sc_score_chart", use_container_width=True):
            _go("chart", "single_query", str(r["종목"]) if domestic else str(r["코드"]))
    else:
        st.caption("행을 누르면 기업 리서치·차트로 바로 갈 수 있습니다.")
    n_err = len(have) - len(good)
    st.caption(f"{len(good):,}종목 중 상위 20" + (f" · {n_err}종목은 재무제표·주가 이력이 부족해 빠짐" if n_err else "")
               + ". 종합 점수 = 기업 리서치의 세 점수(CAPM 미포함 · 업종 평균 대비 · CAPM 대비) 평균. "
               "종목이 많아 컨센서스(12M 선행 PER)와 실적 서프라이즈 신호는 빼고 계산해 리서치 화면과 몇 점 다를 수 있습니다. "
               "투자 권유가 아닌 과거 데이터 기반 모델 점수입니다.")
