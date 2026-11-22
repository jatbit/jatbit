"""신용위험(부도확률·회사채 금리) — 기업 리서치 탭의 하위 탭
- 시장 회사채 금리: FRED의 ICE BofA 미국 회사채 등급별 스프레드(OAS)·실효수익률, 무디스 Aaa/Baa 수익률
- 추정 신용등급: 이자보상배율 → 등급·스프레드 (Damodaran, 2026년 1월 표)
- 등급별 역사적 누적 부도율 (S&P Global Ratings, 1981–2024 전 세계 기업)
- 스프레드에 내재된 부도확률 (위험중립, 회수율 가정)
- 머튼(KMV) 구조모형: 시가총액·주가 변동성 → 자산가치·자산 변동성 → 부도거리(DD)·부도확률
- Altman Z-점수(제조업) / Z''-점수(비제조업)
- CDS 프리미엄: 부도강도(해저드) 곡선으로 CDS 프리미엄 레그·보호 레그를 계산해 적정 CDS 스프레드(만기별)·업프런트·RPV01·평가손익
# 파일 구성: credit_model.py(계산·표) · 이 파일(데이터·화면)
"""
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

from bsm import ncdf
from glossary import col_help
from ui import mgrid


from credit_model import *  # noqa: F401,F403

# ----------------------------------------------------------------------
# 데이터
# ----------------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner="FRED에서 회사채 금리를 불러오는 중...")
def load_credit_market(years: int = 3) -> pd.DataFrame:
    from bonds import _fred_csv

    start = (datetime.now() - timedelta(days=365 * years + 10)).strftime("%Y-%m-%d")
    ids = [s for pair in ICE.values() for s in pair] + list(OTHER)
    frames = _fred_csv(ids, start)  # 시리즈별 병렬 조회, 실패한 시리즈는 빠짐
    return frames / 100


def _row(df: pd.DataFrame, keys) -> pd.Series | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for k in keys:
        if k in df.index:
            return pd.to_numeric(df.loc[k], errors="coerce")
    return None


def annual_frame(R: dict, sec: pd.DataFrame | None) -> pd.DataFrame:
    """연도별 신용 지표 계산용 항목. SEC 10-K를 우선, 빠진 항목은 Yahoo 연간 재무제표로 채움."""
    names = {"총자산": (["Total Assets"], "balance"), "총부채": (["Total Liabilities Net Minority Interest"], "balance"),
             "유동자산": (["Current Assets"], "balance"), "유동부채": (["Current Liabilities"], "balance"),
             "이익잉여금": (["Retained Earnings"], "balance"), "자기자본": (["Stockholders Equity", "Common Stock Equity"], "balance"),
             "단기차입금": (["Current Debt", "Current Debt And Capital Lease Obligation"], "balance"),
             "장기차입금": (["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], "balance"),
             "현금성자산": (["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"], "balance"),
             "매출": (["Total Revenue", "Operating Revenue"], "income"), "영업이익": (["EBIT", "Operating Income"], "income"),
             "이자비용": (["Interest Expense", "Interest Expense Non Operating"], "income"),
             "감가상각비": (["Reconciled Depreciation", "Depreciation And Amortization In Income Statement"], "income"),
             "영업현금흐름": (["Operating Cash Flow"], "cashflow")}
    if sec is not None and not sec.empty:
        out = pd.DataFrame(index=list(names), columns=sec.columns, dtype=float)
        for n in names:
            if n in sec.index:
                out.loc[n] = pd.to_numeric(sec.loc[n], errors="coerce")
        for k in ("희석주식수", "_filed_희석주식수"):
            if k in sec.index:
                out.loc[k] = sec.loc[k]
    else:
        out = None
    # Yahoo로 채우기
    for n, (keys, src) in names.items():
        s = _row(R.get(src), keys)
        if s is None:
            continue
        s.index = pd.to_datetime(s.index)
        if out is None:
            out = pd.DataFrame(index=list(names), columns=sorted(s.index), dtype=float)
        for d, v in s.items():
            if not np.isfinite(v):
                continue
            near = [c for c in out.columns if abs((c - d).days) <= 10]
            if near:
                if not np.isfinite(out.loc[n, near[0]]):
                    out.loc[n, near[0]] = v
            elif sec is None or sec.empty:
                out[d] = np.nan
                out.loc[n, d] = v
    if out is None:
        return pd.DataFrame()
    out = out[sorted(out.columns)]
    if "이자비용" in out.index:
        out.loc["이자비용"] = out.loc["이자비용"].abs()
    return out


def _price_at(hist: pd.Series, d: pd.Timestamp):
    if not isinstance(hist, pd.Series) or hist.empty:
        return np.nan
    h = hist.copy()
    h.index = pd.to_datetime(h.index)
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    b = h[h.index <= d]
    return float(b.iloc[-1]) if len(b) and (d - b.index[-1]).days <= 10 else np.nan


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:  # noqa: BLE001
        return None


def _big(x):
    import money

    return money.big(x)


def _pp(x, nd=2):
    if x is None or not np.isfinite(x):
        return "-"
    if abs(x) < 1e-12:
        return f"{0:.{nd}f}%"
    if abs(x) < 0.0001:
        return f"{x * 100:.1e}%"
    return f"{x * 100:.{nd}f}%"


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
def render_credit(ticker: str, R: dict, sec: pd.DataFrame | None, domestic: bool = False) -> None:
    import money

    info = R.get("info") or {}
    financial = (info.get("sector") or "") in FIN_SECTORS
    A = annual_frame(R, sec)
    try:
        M = load_credit_market()
    except Exception as e:  # noqa: BLE001
        M = pd.DataFrame()
        st.caption(f"FRED 회사채 금리를 불러오지 못했습니다: {e}")
    last = lambda c: float(M[c].dropna().iloc[-1]) if c in M and M[c].dropna().size else np.nan  # noqa: E731
    rf1 = last("DGS1") if np.isfinite(last("DGS1")) else 0.04
    rf10 = last("DGS10") if np.isfinite(last("DGS10")) else 0.042
    rf5_kr = None
    if domestic:  # 국내 종목: 금리는 국고채, 등급별 스프레드는 국내 무료 자료가 없어 미국 회사채 지수로 대신
        from kr import kr_rate

        k10, k3 = kr_rate("10Y"), kr_rate("3Y")
        rf10 = k10[0] if k10 else 0.03
        rf1 = k3[0] if k3 else rf10
        rf5_kr = k3[0] if k3 else rf10
        st.caption("국내 회사채 등급별 시장 금리는 무료 자료가 없어 미국 회사채 지수의 스프레드를 참고값으로 씁니다. 금리는 국고채 기준입니다.")

    g = lambda n: A.loc[n] if (not A.empty and n in A.index) else pd.Series(dtype=float)  # noqa: E731
    lastv = lambda n: (float(g(n).dropna().iloc[-1]) if g(n).dropna().size else np.nan)  # noqa: E731
    ebit, intr = lastv("영업이익"), lastv("이자비용")
    cov = ebit / intr if np.isfinite(ebit) and np.isfinite(intr) and intr > 0 else (np.inf if np.isfinite(ebit) and ebit > 0 else np.nan)
    srt = synthetic_rating(cov, financial)

    # ---- 머튼 입력
    E = _num(info.get("marketCap"))
    hist = R.get("hist")
    sigE = np.nan
    if isinstance(hist, pd.Series) and len(hist) > 260:
        lr = np.log(hist.astype(float)).diff().dropna().iloc[-252:]
        sigE = float(lr.std(ddof=1) * math.sqrt(252))
    st_debt, lt_debt = lastv("단기차입금"), lastv("장기차입금")
    tot_debt = _num(info.get("totalDebt"))
    if not np.isfinite(st_debt):
        st_debt = 0.0
    if not np.isfinite(lt_debt):
        lt_debt = max((tot_debt or 0) - st_debt, 0.0)

    k = f"cr_{ticker}"
    with st.expander("머튼 모형 가정 바꾸기"):
        c1, c2, c3, c4 = st.columns(4)
        sc, un = money.unit()
        st_in = c1.number_input(f"단기 부채 ({un})", value=round(st_debt / sc, 1), step=100.0, key=k + "_st") * sc
        lt_in = c2.number_input(f"장기 부채 ({un})", value=round(lt_debt / sc, 1), step=100.0, key=k + "_lt") * sc
        lt_w = c3.number_input("장기 부채 반영 비율", value=0.5, min_value=0.0, max_value=1.0, step=0.1, key=k + "_w",
                               help="KMV 부도점 = 단기 부채 + 0.5 × 장기 부채 (실무 관행)")
        T = c4.number_input("기간 (년)", value=1.0, min_value=0.25, max_value=10.0, step=0.25, key=k + "_T")
        c5, c6, c7 = st.columns(3)
        sig_in = c5.number_input("주가 변동성 (%)", value=round(sigE * 100, 1) if np.isfinite(sigE) else 30.0, step=1.0, key=k + "_sig",
                                 help="최근 1년 일간 로그수익률의 연환산 표준편차") / 100
        r_in = c6.number_input("무위험금리 (%)", value=round(rf1 * 100, 2), step=0.05, key=k + "_r", help="국고채 3년" if domestic else "미 국채 1년") / 100
        rec = c7.number_input("회수율 가정 (%)", value=40.0, step=5.0, key=k + "_rec", help="스프레드 → 부도확률 환산에 사용. 선순위 무담보 채권은 보통 40% 내외") / 100
    D = st_in + lt_w * lt_in
    beta = _num(info.get("beta")) or 1.0
    mu = r_in + beta * 0.05
    MT = merton(E or 0, sig_in, D, r_in, T, mu) if E and D > 0 else {"ok": False}

    # ---- Altman (최근 연도)
    Z = None
    if not A.empty:
        d = A.columns[-1]
        ta, tl = A.loc["총자산", d], A.loc["총부채", d]
        eq = A.loc["자기자본", d]
        if not np.isfinite(tl) and np.isfinite(ta) and np.isfinite(eq):
            tl = ta - eq
        vals = [A.loc["유동자산", d] - A.loc["유동부채", d], A.loc["이익잉여금", d], A.loc["영업이익", d], E or np.nan,
                A.loc["매출", d], ta, tl, eq]
        if all(np.isfinite(v) for v in vals) and ta > 0 and tl > 0:
            Z = altman(*vals)

    # ---- 요약
    sp_mkt = np.nan
    if srt:
        bk = bucket(srt[0])
        ice_key = "CCC 이하" if bk == "CCC/C" else bk
        sp_mkt = last(ICE[ice_key][0])
    items = [("추정 신용등급", srt[0] if srt else "-",
              f"이자보상배율 {cov:.1f}배 ({'금융업 표' if financial else '대형 비금융 기업 표'}, Damodaran 2026.1)" if srt and np.isfinite(cov)
              else ("이자비용이 거의 없어 최상위로 분류" if srt else "영업이익·이자비용 자료 없음")),
             ("추정 세전 차입금리", _pp(rf10 + srt[1]) if srt else "-",
              f"국채 10년 {rf10 * 100:.2f}% + 등급 스프레드 {srt[1] * 100:.2f}%" if srt else ""),
             ("같은 등급 회사채 시장 스프레드", _pp(sp_mkt) if np.isfinite(sp_mkt) else "-", "ICE BofA 등급별 지수 OAS (FRED, 최신)"),
             ("역사적 1년 부도율", f"{SP_CUM[bucket(srt[0])][0]:.2f}%" if srt else "-", "S&P 1981–2024 같은 등급 평균"),
             ("역사적 5년 누적 부도율", f"{SP_CUM[bucket(srt[0])][4]:.2f}%" if srt else "-")]
    if MT.get("ok"):
        items += [("머튼 부도거리(DD)", f"{MT['DD']:.2f}σ", "자산가치가 부도점까지 떨어지려면 표준편차 몇 배가 필요한지"),
                  (f"머튼 {T:g}년 부도확률", _pp(MT["PD"], 3), "위험중립 기준 N(−d2)"),
                  ("머튼 부도확률 (실제 기대수익률)", _pp(MT["PD_p"], 3), f"자산 기대수익률 = 무위험 + 베타 {beta:.2f} × 5%"),
                  ("머튼 내재 신용스프레드", _pp(MT["spread"], 3), "부채를 '무위험채권 − 풋옵션'으로 본 가격에서 계산")]
    r5 = rf5_kr if rf5_kr is not None else (last("DGS5") if np.isfinite(last("DGS5")) else rf10)
    cds5 = cds_price(surv_flat(sp_mkt / (1 - rec)), 5, r5, rec)["par"] if np.isfinite(sp_mkt) else np.nan
    items.append(("추정 5년 CDS 프리미엄", f"{cds5 * 1e4:,.0f}bp" if np.isfinite(cds5) else "-",
                  "같은 등급 회사채 스프레드에서 구한 부도강도로 계산 (개별 CDS 시세는 유료)"))
    if Z:
        items += [("Altman Z (제조업)", f"{Z['Z']:.2f} · {z_zone(Z['Z'])}", "2.99 초과 안전 · 1.81 미만 부실 위험"),
                  ("Altman Z'' (비제조업)", f"{Z['Z2']:.2f} · {z_zone(Z['Z2'], 'Z2')}", "2.60 초과 안전 · 1.10 미만 부실 위험")]
    mgrid(items, cols=4)
    if financial:
        st.warning("금융회사는 부채가 영업의 일부라 이자보상배율·Altman Z·머튼 모형이 잘 맞지 않습니다. 추정 등급은 금융업 표를 썼지만 참고만 하세요.")

    tabs = ui.lazy_tabs(["회사채 시장 금리", "등급·부도확률", "CDS 프리미엄", "머튼 구조모형", "Altman Z-점수", "신용 지표 추이"], key="cr_tabs")
    if tabs[2][1]:
        with tabs[2][0]:
            _render_cds(ticker, srt, sp_mkt, MT, D, r_in, r5, rec)
    tabs = [tabs[0], tabs[1], tabs[3], tabs[4], tabs[5]]
    if tabs[0][1]:
        with tabs[0][0]:
            if M.empty:
                st.info("시장 금리 자료가 없습니다.")
            else:
                rows = []
                for nm, (oas, ey) in ICE.items():
                    s = M[oas].dropna() if oas in M else pd.Series(dtype=float)
                    if s.empty:
                        continue
                    y1 = s[s.index <= s.index[-1] - pd.Timedelta(days=365)]
                    rows.append({"등급": nm, "스프레드(OAS)": f"{s.iloc[-1] * 100:.2f}%",
                                 "실효수익률": f"{last(ey) * 100:.2f}%" if np.isfinite(last(ey)) else "-",
                                 "1년 전 스프레드": f"{y1.iloc[-1] * 100:.2f}%" if len(y1) else "-",
                                 "3년 범위": f"{s.min() * 100:.2f}% ~ {s.max() * 100:.2f}%",
                                 "내재 1년 부도확률": _pp(spread_pd(float(s.iloc[-1]), 1, rec)),
                                 "내재 5년 부도확률": _pp(spread_pd(float(s.iloc[-1]), 5, rec)),
                                 "역사적 1년 부도율": f"{SP_CUM['CCC/C' if nm == 'CCC 이하' else nm][0]:.2f}%",
                                 "역사적 5년 부도율": f"{SP_CUM['CCC/C' if nm == 'CCC 이하' else nm][4]:.2f}%"})
                asof = M.dropna(how="all").index[-1]
                st.markdown(f"###### 미국 회사채 등급별 금리 ({asof:%Y-%m-%d})")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, column_config=col_help(pd.DataFrame(rows).columns))
                show = st.radio("그래프", ["등급별 스프레드", "등급별 실효수익률", "무디스 Aaa·Baa와 국채"], horizontal=True, key="cr_mk_show")
                f = go.Figure()
                if show == "무디스 Aaa·Baa와 국채":
                    for c in ("DAAA", "DBAA", "DGS10", "DGS1"):
                        if c in M:
                            s = M[c].dropna()
                            f.add_trace(go.Scatter(x=s.index, y=s * 100, name=OTHER[c], mode="lines",
                                                   hovertemplate="%{x|%Y-%m-%d} %{y:.2f}%<extra>" + OTHER[c] + "</extra>"))
                else:
                    j = 0 if show == "등급별 스프레드" else 1
                    for nm, pair in ICE.items():
                        if pair[j] in M:
                            s = M[pair[j]].dropna()
                            f.add_trace(go.Scatter(x=s.index, y=s * 100, name=nm, mode="lines", line=dict(color=COLORS[nm], width=1.5),
                                                   hovertemplate="%{x|%Y-%m-%d} %{y:.2f}%<extra>" + nm + "</extra>"))
                if srt and show != "무디스 Aaa·Baa와 국채":
                    f.add_annotation(text=f"{ticker} 추정 등급: {srt[0]}", xref="paper", yref="paper", x=0.01, y=0.98, showarrow=False,
                                     font=dict(size=11), bgcolor="rgba(128,128,128,0.15)")
                f.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                                yaxis=dict(ticksuffix="%", type="log" if show == "등급별 스프레드" else "linear"),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
                st.plotly_chart(f, use_container_width=True)
                if show == "등급별 스프레드":
                    st.caption("세로축은 로그 눈금입니다. ICE BofA 지수는 FRED에서 최근 몇 년치만 공개됩니다.")

    if tabs[1][1]:
        with tabs[1][0]:
            tbl = DAMO_FIN if financial else DAMO_LARGE
            rows = []
            for i, (lo, rt, sp) in enumerate(tbl):
                if i == 0:
                    rng = f"{lo:g} 초과"
                elif lo < -1e8:
                    rng = f"{tbl[i - 1][0]:g} 이하"
                else:
                    rng = f"{lo:g} 초과 ~ {tbl[i - 1][0]:g} 이하"
                rows.append({"이자보상배율": rng, "등급": rt, "스프레드": f"{sp * 100:.2f}%", "세전 차입금리": f"{(rf10 + sp) * 100:.2f}%",
                             "역사적 1년 부도율": f"{SP_CUM[bucket(rt)][0]:.2f}%", "역사적 5년 부도율": f"{SP_CUM[bucket(rt)][4]:.2f}%",
                             "": "◀ 이 회사" if srt and rt == srt[0] else ""})
            st.markdown("###### 이자보상배율 → 추정 등급 (Damodaran, 2026년 1월)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, column_config=col_help(pd.DataFrame(rows).columns))
            f = go.Figure()
            yrs = list(range(1, 16))
            for nm, v in SP_CUM.items():
                f.add_trace(go.Scatter(x=yrs, y=v, name=nm, mode="lines+markers", marker=dict(size=4),
                                       line=dict(width=3 if srt and bucket(srt[0]) == nm else 1.3),
                                       hovertemplate="%{x}년 누적 %{y:.2f}%<extra>" + nm + "</extra>"))
            f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                            xaxis=dict(title="기간(년)"), yaxis=dict(title="누적 부도율(%)", type="log", ticksuffix="%"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(f, use_container_width=True)
            st.caption("출처: S&P Global Ratings 2024 부도·등급 전이 연구 (전 세계 기업 1981–2024 평균)")

    if tabs[2][1]:
        with tabs[2][0]:
            if not MT.get("ok"):
                st.info("시가총액·주가 변동성·부채 자료가 부족해 계산하지 못했습니다. 위 '머튼 모형 가정 바꾸기'에서 직접 넣어 보세요.")
            else:
                mgrid([("자기자본 시장가치 E", _big(E)), ("부도점 D", _big(D), f"단기 {_big(st_in)} + {lt_w:g} × 장기 {_big(lt_in)}"),
                       ("자산가치 V (추정)", _big(MT["V"])), ("자산 변동성 σV", _pp(MT["sigV"], 1), f"주가 변동성 {sig_in * 100:.1f}%"),
                       ("레버리지 D/V", _pp(MT["lev"], 1)), ("부도거리 DD", f"{MT['DD']:.2f}σ"),
                       ("부도확률 (위험중립)", _pp(MT["PD"], 4)), ("부도확률 (실제 기대수익률)", _pp(MT["PD_p"], 4)),
                       ("부채 시장가치", _big(MT["debt_val"]), "V − E"), ("내재 신용스프레드", _pp(MT["spread"], 3))], cols=5)
                # 주가 변동성 민감도
                sigs = np.linspace(max(0.05, sig_in * 0.4), sig_in * 2.5, 30)
                pds, dds = [], []
                for s_ in sigs:
                    m = merton(E, s_, D, r_in, T, mu)
                    pds.append(m["PD"] * 100 if m.get("ok") else np.nan)
                    dds.append(m["DD"] if m.get("ok") else np.nan)
                f = go.Figure()
                f.add_trace(go.Scatter(x=sigs * 100, y=pds, name="부도확률(%)", mode="lines", line=dict(color="#E5534B"),
                                       hovertemplate="주가 변동성 %{x:.0f}% → 부도확률 %{y:.4f}%<extra></extra>"))
                f.add_trace(go.Scatter(x=sigs * 100, y=dds, name="부도거리(σ)", mode="lines", yaxis="y2", line=dict(color="#4C9BE8"),
                                       hovertemplate="주가 변동성 %{x:.0f}% → DD %{y:.2f}σ<extra></extra>"))
                f.add_vline(x=sig_in * 100, line_dash="dot", line_color="#888", annotation_text="현재")
                f.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                                xaxis=dict(title="주가 변동성(%)"), yaxis=dict(title="부도확률(%)"),
                                yaxis2=dict(title="부도거리(σ)", overlaying="y", side="right", showgrid=False),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
                st.plotly_chart(f, use_container_width=True)

    if tabs[3][1]:
        with tabs[3][0]:
            if A.empty:
                st.info("재무상태표 자료가 없습니다.")
            else:
                fac = None
                if "_filed_희석주식수" in A.index:
                    from research import _split_factor
                    fac = _split_factor(A.loc["_filed_희석주식수"], R.get("splits")).reindex(A.columns).fillna(1.0)
                zr = {}
                for d in A.columns:
                    ta, tl, eq = A.loc["총자산", d], A.loc["총부채", d], A.loc["자기자본", d]
                    if not np.isfinite(tl) and np.isfinite(ta) and np.isfinite(eq):
                        tl = ta - eq
                    mve = np.nan
                    if d == A.columns[-1] and E:
                        mve = E
                    elif "희석주식수" in A.index and np.isfinite(A.loc["희석주식수", d]):
                        mve = _price_at(hist, d) * A.loc["희석주식수", d] * (fac[d] if fac is not None else 1.0)
                    vals = [A.loc["유동자산", d] - A.loc["유동부채", d], A.loc["이익잉여금", d], A.loc["영업이익", d], mve,
                            A.loc["매출", d], ta, tl, eq]
                    if all(np.isfinite(v) for v in vals) and ta > 0 and tl > 0:
                        zr[d] = altman(*vals)
                if not zr:
                    st.info("Altman Z를 계산할 항목(유동자산·유동부채·이익잉여금·영업이익·매출·총자산·총부채)이 부족합니다.")
                else:
                    Zd = pd.DataFrame(zr).T
                    xs = [d.strftime("%Y-%m") for d in Zd.index]
                    f = go.Figure()
                    f.add_trace(go.Scatter(x=xs, y=Zd["Z"], name="Z (제조업)", mode="lines+markers", line=dict(color="#4C9BE8", width=2)))
                    f.add_trace(go.Scatter(x=xs, y=Zd["Z2"], name="Z'' (비제조업)", mode="lines+markers", line=dict(color="#9B6FD6", width=2, dash="dot")))
                    for y, c, t in ((2.99, "#1B9E4B", "Z 안전 2.99"), (1.81, "#E5534B", "Z 부실 1.81"), (2.60, "#1B9E4B", "Z'' 안전 2.60"),
                                    (1.10, "#E5534B", "Z'' 부실 1.10")):
                        f.add_hline(y=y, line_dash="dash" if "Z''" not in t else "dot", line_color=c, line_width=1,
                                    annotation_text=t, annotation_font_size=9)
                    f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
                    st.plotly_chart(f, use_container_width=True)
                    tbl = Zd[["X1", "X2", "X3", "X4", "X5", "X4b", "Z", "Z2"]].T
                    tbl.columns = xs
                    tbl.index = ["X1 운전자본/총자산", "X2 이익잉여금/총자산", "X3 영업이익/총자산", "X4 시가총액/총부채", "X5 매출/총자산",
                                 "X4'' 장부 자기자본/총부채", "Altman Z (제조업)", "Altman Z'' (비제조업)"]
                    st.dataframe(tbl.map(lambda v: f"{v:.3f}") if hasattr(tbl, "map") else tbl.applymap(lambda v: f"{v:.3f}"),
                                 use_container_width=True)

    if tabs[4][1]:
        with tabs[4][0]:
            if A.empty:
                st.info("재무제표 자료가 없습니다.")
            else:
                debt = g("단기차입금").fillna(0) + g("장기차입금").fillna(0)
                ebitda = g("영업이익") + g("감가상각비").fillna(0)
                cash = g("현금성자산")
                ratios = {"이자보상배율(배)": g("영업이익") / g("이자비용"), "EBITDA/이자비용(배)": ebitda / g("이자비용"),
                          "총차입금/EBITDA(배)": (debt / ebitda).where(ebitda > 0), "순차입금/EBITDA(배)": ((debt - cash) / ebitda).where(ebitda > 0),
                          "영업현금흐름/총차입금(%)": (g("영업현금흐름") / debt * 100).where(debt > 0),
                          "부채비율(%)": g("총부채") / g("자기자본") * 100, "유동비율(%)": g("유동자산") / g("유동부채") * 100,
                          "실효 이자율(%)": (g("이자비용") / debt * 100).where(debt > 0)}
                Rt = pd.DataFrame(ratios).T
                Rt = Rt.loc[:, Rt.notna().any()]
                Rt.columns = [pd.Timestamp(c).strftime("%Y-%m") for c in Rt.columns]
                pick = st.multiselect("그래프로 볼 지표", list(Rt.index), default=["이자보상배율(배)", "순차입금/EBITDA(배)"], key="cr_ratio_pick")
                if pick:
                    f = go.Figure()
                    for nm in pick:
                        f.add_trace(go.Scatter(x=list(Rt.columns), y=Rt.loc[nm], name=nm, mode="lines+markers",
                                               yaxis="y2" if nm.endswith("(%)") else "y"))
                    f.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                                    yaxis=dict(title="배"), yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
                    st.plotly_chart(f, use_container_width=True)
                st.dataframe(Rt.map(lambda v: f"{v:,.2f}" if np.isfinite(v) else "-") if hasattr(Rt, "map")
                             else Rt.applymap(lambda v: f"{v:,.2f}" if np.isfinite(v) else "-"), use_container_width=True)


def _render_cds(ticker, srt, sp_mkt, MT, D, r1, r5, rec) -> None:
    yrs = [1, 2, 3, 5, 7, 10]
    curves = {}
    if np.isfinite(sp_mkt):
        curves[f"시장 회사채 스프레드 기준 ({bucket(srt[0]) if srt else '-'})"] = surv_flat(sp_mkt / (1 - rec))
    if srt:
        curves[f"추정 등급 스프레드 기준 ({srt[0]}, Damodaran)"] = surv_flat(srt[1] / (1 - rec))
        hist_pd = {float(i + 1): v / 100 for i, v in enumerate(SP_CUM[bucket(srt[0])])}
        curves[f"역사적 부도율 기준 ({bucket(srt[0])}, S&P)"] = surv_from_cum(hist_pd)
    if MT.get("ok"):
        curves["머튼 구조모형"] = surv_from_cum(merton_cum_pd(MT["V"], MT["sigV"], D, r1, [0.25, 0.5] + list(range(1, 11))))

    st.markdown("###### 직접 넣는 시장 CDS (선택)")
    c1, c2, c3 = st.columns(3)
    mk = c1.number_input("시장 CDS 프리미엄 (bp)", value=0.0, min_value=0.0, step=5.0, key=f"cds_mk_{ticker}",
                         help="블룸버그·증권사 리포트 등에서 본 실제 CDS 시세가 있으면 넣으세요. 0이면 사용하지 않습니다.")
    mk_T = c2.selectbox("그 CDS 만기", [1, 3, 5, 7, 10], index=2, key=f"cds_mkT_{ticker}", format_func=lambda x: f"{x}년")
    std_c = c3.selectbox("표준 쿠폰", [100, 500], index=0 if (not srt or bucket(srt[0]) in ("AAA", "AA", "A", "BBB")) else 1,
                         key=f"cds_c_{ticker}", format_func=lambda x: f"{x}bp", help="북미 표준 계약: 투자등급 100bp, 하이일드 500bp")
    if mk > 0:
        lam = implied_hazard(mk / 1e4, mk_T, r5, rec)
        curves[f"입력한 시장 CDS {mk:g}bp ({mk_T}년)"] = surv_flat(lam)
        mgrid([("내재 부도강도 λ", f"{lam * 100:.3f}%/년", f"신용 삼각형 근사 s/(1−R) = {mk / 1e4 / (1 - rec) * 100:.3f}%"),
               ("내재 1년 부도확률", _pp(1 - math.exp(-lam)), "위험중립"), ("내재 5년 누적 부도확률", _pp(1 - math.exp(-5 * lam))),
               ("머튼 모형 대비", f"{mk / (cds_price(curves['머튼 구조모형'], mk_T, r5, rec)['par'] * 1e4):,.1f}배"
                if "머튼 구조모형" in curves and cds_price(curves["머튼 구조모형"], mk_T, r5, rec)["par"] > 0 else "-",
                "시장 CDS ÷ 머튼 모형 적정 CDS")], cols=4)
    if not curves:
        st.info("등급·스프레드·머튼 모형 중 계산 가능한 것이 없어 CDS를 추정하지 못했습니다.")
        return

    rows, plot = [], {}
    for nm, Q in curves.items():
        pr = {y: cds_price(Q, y, r5, rec) for y in yrs}
        plot[nm] = [pr[y]["par"] * 1e4 for y in yrs]
        p5 = pr[5]
        rows.append({"기준": nm, **{f"{y}년": f"{pr[y]['par'] * 1e4:,.1f}" for y in yrs},
                     "5년 RPV01": f"{p5['rpv01']:.3f}", f"5년 업프런트 (쿠폰 {std_c}bp)": f"{(p5['par'] - std_c / 1e4) * p5['rpv01'] * 100:+.2f}%",
                     "5년 누적 부도확률": _pp(p5["pd"])})
    st.markdown("###### 적정 CDS 프리미엄 (bp, 연간)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, column_config=col_help(pd.DataFrame(rows).columns))
    f = go.Figure()
    for nm, v in plot.items():
        f.add_trace(go.Scatter(x=yrs, y=v, name=nm, mode="lines+markers", hovertemplate="%{x}년 %{y:,.1f}bp<extra>" + nm + "</extra>"))
    f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                    xaxis=dict(title="만기(년)"), yaxis=dict(title="CDS 프리미엄(bp)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(f, use_container_width=True)

    st.markdown("###### CDS 평가손익 계산기")
    nm0 = list(curves)[0]
    base5 = cds_price(curves[nm0], 5, r5, rec)
    c1, c2, c3, c4 = st.columns(4)
    side = c1.radio("포지션", ["보호 매수", "보호 매도"], horizontal=True, key=f"cds_side_{ticker}")
    import money

    notional = c2.number_input(f"명목금액 ({money.sym()})", value=10_000_000_000 if money.is_krw() else 10_000_000,
                               step=1_000_000_000 if money.is_krw() else 1_000_000, key=f"cds_n_{ticker}", format="%d")
    k_bp = c3.number_input("계약 스프레드 (bp)", value=round(base5["par"] * 1e4, 1), step=5.0, key=f"cds_k_{ticker}")
    now_bp = c4.number_input("현재 스프레드 (bp)", value=round(base5["par"] * 1e4 + 50, 1), step=5.0, key=f"cds_s_{ticker}")
    lam_now = implied_hazard(now_bp / 1e4, 5, r5, rec)
    rp = cds_price(surv_flat(lam_now), 5, r5, rec)["rpv01"]
    mtm = (now_bp - k_bp) / 1e4 * rp * notional * (1 if side == "보호 매수" else -1)
    cs01 = rp * notional / 1e4
    u = money.sym()
    mgrid([("평가손익", f"{mtm:+,.0f} {u}"),
           ("CS01 (1bp당)", f"{cs01:,.0f} {u}"),
           ("부도 시 손익", f"{(1 - rec) * notional * (1 if side == '보호 매수' else -1):+,.0f} {u}", f"회수율 {rec * 100:.0f}%"),
           ("연간 프리미엄", f"{k_bp / 1e4 * notional:,.0f} {u}")], cols=4)
    st.caption("개별 기업 CDS 시세는 무료로 공개되지 않아 등급 스프레드·부도율·머튼 모형으로 계산한 이론값입니다.")
