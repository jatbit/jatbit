"""옵션을 포함한 포트폴리오의 CAPM·손익 분석
- 대상: '델타·감마 헷지' 탭에 입력된 보유 주식 + 옵션 포지션(+ 선택하면 계산된 헷지 거래) 중 미국 종목
- 손익 그래프: 기초자산 가격별 손익 (오늘 / 선택 시점 / 가장 가까운 만기), 손익분기점, 최대 이익·손실
- CAPM: 옵션 베타 = 탄력성(Ω = Δ·S / 옵션가) × 주식 베타, 포트폴리오 베타·달러 베타·기대수익
- 위험: 몬테카를로(상관 반영, 전체 재평가) VaR·ES, 델타-정규 VaR, 기대 손익·이익 확률
"""
import math
from datetime import datetime, timedelta

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bsm
from options import (MULT, TYPE_KO, _row_for, build_chain, is_us_ticker, load_chain_raw, load_expiries,
                     load_underlying, load_us_rate_3m, mgrid, years_to_expiry)

def _usd(x) -> str:
    x = float(x)
    return ("-$" if x < 0 else "$") + f"{abs(x):,.0f}"


COLORS = {"today": "#4C9BE8", "horizon": "#F2A93B", "expiry": "#E5534B"}


# ----------------------------------------------------------------------
# 데이터
# ----------------------------------------------------------------------
@st.cache_data(ttl=60 * 60, show_spinner="S&P 500 지수를 불러오는 중...")
def load_spx() -> pd.Series:
    start = (datetime.now() - timedelta(days=800)).strftime("%Y-%m-%d")
    for _ in range(3):
        try:
            d = fdr.DataReader("US500", start)
            s = d["Close"].dropna()
            if len(s) > 100:
                return s
        except Exception:  # noqa: BLE001
            pass
    raise RuntimeError("S&P 500 지수를 불러오지 못했습니다")


def auto_params(ticker: str) -> dict:
    u = load_underlying(ticker)
    rr = load_us_rate_3m()
    r_simple = rr[0] if rr else 0.04
    hv1y = bsm.hist_vol(u["close"], 252)
    hv60 = bsm.hist_vol(u["close"], 60)
    sig = hv60 if hv60 == hv60 and hv60 > 0 else (hv1y if hv1y == hv1y else 0.3)
    return {"S": u["spot"], "r": math.log1p(r_simple), "r_simple": r_simple,
            "q": u["div12"] / u["spot"] if u["spot"] > 0 else 0.0,
            "sig_theo": sig, "close": u["close"], "asof": u["asof"]}


def _beta(close: pd.Series, spx: pd.Series, days: int = 252) -> float:
    a = pd.concat([close.pct_change(), spx.pct_change()], axis=1, join="inner").dropna().iloc[-days:]
    if len(a) < 30:
        return float("nan")
    c = np.cov(a.iloc[:, 0], a.iloc[:, 1], ddof=1)
    return float(c[0, 1] / c[1, 1])


# ----------------------------------------------------------------------
# 포지션 모으기
# ----------------------------------------------------------------------
def collect_positions(include_hedge: bool):
    """{티커: {"shares": 주식수, "opts": [{"exp","is_call","K","n"}]}}, 제외된 국내/기타 목록."""
    under, skipped = {}, []
    stocks = st.session_state.get("hedge_stocks_cur")
    opts = st.session_state.get("hedge_opts_cur")
    if stocks is not None:
        for r in stocks.to_dict("records"):
            code = str(r.get("종목코드") or "").strip().upper()
            sh = pd.to_numeric(r.get("보유 주식 수"), errors="coerce")
            if not code or code == "NAN" or not np.isfinite(sh):
                continue
            if not is_us_ticker(code):
                skipped.append(f"{r.get('종목명') or code}({code})")
                continue
            under.setdefault(code, {"shares": 0.0, "opts": []})["shares"] += float(sh)
    if opts is not None:
        for r in opts.to_dict("records"):
            code = str(r.get("기초자산") or "").strip().upper()
            k = pd.to_numeric(r.get("행사가"), errors="coerce")
            n = pd.to_numeric(r.get("수량(계약)"), errors="coerce")
            if not code or not is_us_ticker(code) or r.get("종류") not in ("콜", "풋") or not np.isfinite(k) \
                    or not np.isfinite(n) or not r.get("만기"):
                continue
            under.setdefault(code, {"shares": 0.0, "opts": []})["opts"].append(
                {"exp": str(r["만기"])[:10], "is_call": r["종류"] == "콜", "K": float(k), "n": float(n), "src": "보유"})
    if include_hedge:
        for code, trades in (st.session_state.get("hedge_trades") or {}).items():
            for t in trades:
                if not t["n"]:
                    continue
                u = under.setdefault(code, {"shares": 0.0, "opts": []})
                if t["kind"] == "stock":
                    u["shares"] += t["n"]
                else:
                    u["opts"].append({"exp": t["exp"], "is_call": t["is_call"], "K": t["K"], "n": t["n"], "src": "헷지"})
    return under, skipped


def price_positions(under: dict):
    """각 옵션에 시장가·IV·그릭스·잔존기간을 붙인다."""
    out, warns = {}, []
    for code, info in under.items():
        try:
            P = auto_params(code)
        except Exception as e:  # noqa: BLE001
            warns.append(f"{code}: 가격을 불러오지 못해 제외 ({e})")
            continue
        legs = []
        exps = None
        for o in info["opts"]:
            lab = f"{code} {o['exp']} {TYPE_KO[o['is_call']]} {o['K']:g}"
            try:
                exps = exps if exps is not None else load_expiries(code)
                if o["exp"] not in exps:
                    warns.append(f"{lab}: 상장된 만기가 아니라 제외")
                    continue
                T = years_to_expiry(o["exp"])
                row = _row_for(build_chain(load_chain_raw(code, o["exp"]), P, T, "IV"), o["is_call"], o["K"])
            except Exception as e:  # noqa: BLE001
                warns.append(f"{lab}: 가격 조회 실패로 제외 ({e})")
                continue
            if row is None or not np.isfinite(row["mkt"]):
                warns.append(f"{lab}: 시장가가 없어 제외")
                continue
            legs.append({**o, "T": T, "mkt": float(row["mkt"]), "sig": float(row["sig_g"]),
                         "delta": float(row["delta"]), "gamma": float(row["gamma"]), "vega": float(row["vega"]),
                         "theta": float(row["theta"]), "label": lab})
        out[code] = {"P": P, "shares": info["shares"], "legs": legs}
    return out, warns


# ----------------------------------------------------------------------
# 재평가
# ----------------------------------------------------------------------
def pnl_at(u: dict, S_new, h_years: float) -> np.ndarray:
    """기초자산 가격 S_new(배열), h_years 뒤의 손익($). 그 전에 만기가 지난 옵션은 내재가치로 정산."""
    P = u["P"]
    S_new = np.asarray(S_new, dtype=float)
    pnl = u["shares"] * (S_new - P["S"])
    for lg in u["legs"]:
        rem = lg["T"] - h_years
        if rem <= 1e-6:
            val = np.maximum(S_new - lg["K"], 0.0) if lg["is_call"] else np.maximum(lg["K"] - S_new, 0.0)
        else:
            val = bsm.price(S_new, lg["K"], rem, P["r"], P["q"], lg["sig"], lg["is_call"])
        pnl = pnl + lg["n"] * MULT * (val - lg["mkt"])
    return pnl


def simulate(book: dict, h_days: float, drifts: dict, n_sims: int = 20000, seed: int = 7, trading_day: bool = False):
    """상관을 반영한 로그정규 시나리오로 h_days(달력일) 뒤 전체 손익 분포.
    trading_day=True면 가격 변동은 거래일 1일(1/252년), 옵션 시간가치 감소는 달력일 기준으로 본다(1일 VaR용)."""
    codes = list(book)
    rets = pd.concat({c: np.log(book[c]["P"]["close"]).diff() for c in codes}, axis=1, join="inner").dropna().iloc[-252:]
    if len(rets) < 30:
        cov = np.diag([book[c]["P"]["sig_theo"] ** 2 for c in codes])
    else:
        cov = rets.cov().values * 252
    h = h_days / 365.0
    h_diff = h_days / 252.0 if trading_day else h
    rng = np.random.default_rng(seed)
    try:
        L = np.linalg.cholesky(cov * h_diff + np.eye(len(codes)) * 1e-12)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.clip(np.diag(cov), 1e-12, None) * h_diff))
    Z = rng.standard_normal((n_sims, len(codes))) @ L.T
    tot = np.zeros(n_sims)
    for i, c in enumerate(codes):
        var = cov[i, i]
        S1 = book[c]["P"]["S"] * np.exp((drifts.get(c, 0.0) - 0.5 * var) * h_diff + Z[:, i])
        tot += pnl_at(book[c], S1, h)
    return tot, cov


def _var_es(x: np.ndarray, a: float):
    q = np.quantile(x, 1 - a)
    return -q, -x[x <= q].mean()


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
def render_capm_opt() -> None:
    st.markdown("'델타·감마 헷지' 탭에 입력한 **보유 주식 + 옵션 포지션**(미국 종목)으로 손익 그래프와 옵션을 포함한 CAPM·위험 수치를 계산합니다. "
                "옵션은 각자의 내재변동성으로 재평가합니다.")
    c1, c2, c3 = st.columns([1.4, 1, 1])
    include_hedge = c1.toggle("헷지 탭에서 계산한 헷지 거래 포함", value=False, key="capm_inc_hedge")
    mrp = c2.number_input("시장 위험 프리미엄 E[Rm]−rf (%)", value=5.5, step=0.5, key="capm_mrp") / 100
    n_sims = c3.selectbox("시뮬레이션 횟수", [5000, 20000, 50000], index=1, key="capm_nsims")

    under, skipped = collect_positions(include_hedge)
    if skipped:
        st.caption("옵션 데이터가 없어 제외: " + ", ".join(skipped) + " (국내 종목의 CAPM은 '포트폴리오' 페이지 참고)")
    if not under:
        st.info("'델타·감마 헷지' 탭에서 미국 종목 주식이나 옵션 포지션을 입력하면 계산합니다.")
        return
    with st.spinner("포지션 가격·내재변동성을 계산하는 중... (종목과 만기가 많으면 오래 걸립니다)"):
        book, warns = price_positions(under)
    for w in warns:
        st.warning(w)
    book = {c: u for c, u in book.items() if u["shares"] or u["legs"]}
    if not book:
        st.info("계산할 수 있는 포지션이 없습니다.")
        return

    try:
        spx = load_spx()
    except Exception as e:  # noqa: BLE001
        spx = None
        st.warning(f"S&P 500을 불러오지 못해 베타를 1로 둡니다 ({e}).")
    rf = next(iter(book.values()))["P"]["r_simple"]

    # ---- 포지션별 CAPM 표
    rows, tot = [], {"value": 0.0, "dd": 0.0, "dbeta": 0.0, "gamma_d": 0.0, "vega": 0.0, "theta": 0.0}
    drifts = {}
    for code, u in book.items():
        P = u["P"]
        S = P["S"]
        beta_s = _beta(P["close"], spx) if spx is not None else float("nan")
        if not np.isfinite(beta_s):
            beta_s = 1.0
        u["beta"] = beta_s
        drifts[code] = rf + beta_s * mrp
        if u["shares"]:
            v = u["shares"] * S
            rows.append({"포지션": f"{code} 주식", "수량": f"{u['shares']:,.2f}주", "단가($)": S, "가치($)": v,
                         "달러 델타($)": v, "탄력성 Ω": 1.0, "베타": beta_s, "CAPM 기대수익률(연)": rf + beta_s * mrp})
            tot["value"] += v
            tot["dd"] += v
            tot["dbeta"] += v * beta_s
        for lg in u["legs"]:
            m = lg["n"] * MULT
            v = m * lg["mkt"]
            dd = m * lg["delta"] * S
            omega = lg["delta"] * S / lg["mkt"] if lg["mkt"] > 0 else float("nan")
            b = omega * beta_s
            rows.append({"포지션": f"{lg['label']} ({lg['src']})", "수량": f"{lg['n']:+g}계약", "단가($)": lg["mkt"], "가치($)": v,
                         "달러 델타($)": dd, "탄력성 Ω": omega, "베타": b, "CAPM 기대수익률(연)": rf + b * mrp})
            tot["value"] += v
            tot["dd"] += dd
            tot["dbeta"] += dd * beta_s
            tot["gamma_d"] += m * lg["gamma"] * S * S * 0.01  # 1% 움직일 때 달러 델타 변화
            tot["vega"] += m * lg["vega"]
            tot["theta"] += m * lg["theta"]

    st.subheader("포지션별 CAPM")
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"단가($)": "{:,.2f}", "가치($)": "{:,.0f}", "달러 델타($)": "{:,.0f}", "탄력성 Ω": "{:,.2f}",
                                  "베타": "{:,.2f}", "CAPM 기대수익률(연)": "{:.1%}"}, na_rep="-"),
                 use_container_width=True, hide_index=True)

    # ---- 포트폴리오 수치
    st.subheader("포트폴리오 수치")
    V = tot["value"]
    port_beta = tot["dbeta"] / V if V > 0 else float("nan")
    exp_dollar = V * rf + tot["dbeta"] * mrp
    mgrid([("순 포지션 가치", _usd(V)),
           ("포트폴리오 베타", f"{port_beta:.2f}" if np.isfinite(port_beta) else "정의 안 됨",
            "달러 베타 ÷ 순 포지션 가치. 순가치가 0 이하(옵션 순매도)면 정의되지 않습니다."),
           ("달러 베타", _usd(tot["dbeta"]), "S&P 500이 1% 오르면 약 이 금액의 1%만큼 손익이 납니다."),
           ("CAPM 기대수익률(연)", f"{exp_dollar / V:.1%}" if V > 0 else "-", f"기대 손익 연 {_usd(exp_dollar)}"),
           ("달러 델타", _usd(tot["dd"])),
           ("달러 감마(1% 변동)", _usd(tot["gamma_d"]), "기초자산이 1% 움직일 때 달러 델타가 변하는 양"),
           ("베가(변동성 1%p)", _usd(tot["vega"])),
           ("세타(1일)", _usd(tot["theta"]))])

    # ---- 위험·확률
    nearest = min([lg["T"] for u in book.values() for lg in u["legs"]], default=30 / 365) * 365
    max_days = max([lg["T"] for u in book.values() for lg in u["legs"]], default=30 / 365) * 365
    h_days = st.slider("손익을 볼 시점 (오늘부터 며칠 뒤)", min_value=0.0, max_value=float(max(1.0, round(max_days, 1))),
                       value=float(round(min(nearest, max(1.0, max_days)), 1)), step=0.5, key="capm_hdays")

    with st.spinner("몬테카를로 시뮬레이션 중..."):
        pnl1, _ = simulate(book, 1.0, {c: 0.0 for c in book}, n_sims, trading_day=True)
        pnlh, _ = simulate(book, max(h_days, 1e-3), drifts, n_sims) if h_days > 0 else (np.zeros(1), None)
    v95, e95 = _var_es(pnl1, 0.95)
    v99, e99 = _var_es(pnl1, 0.99)

    st.subheader("위험 수치 (1일)")
    mgrid([("VaR 95%", _usd(v95)), ("ES 95%", _usd(e95)), ("VaR 99%", _usd(v99)), ("ES 99%", _usd(e99))])
    st.caption(f"몬테카를로 {n_sims:,}회, 최근 1년 수익률 상관 반영, 옵션은 전체 재평가")

    if h_days > 0:
        st.subheader(f"{h_days:g}일 뒤 손익 분포 (CAPM 기대수익률을 추세로 사용)")
        mgrid([("기대 손익", _usd(pnlh.mean())), ("이익 확률", f"{(pnlh > 0).mean():.1%}"),
               ("하위 5% 손익", _usd(np.quantile(pnlh, 0.05))), ("상위 5% 손익", _usd(np.quantile(pnlh, 0.95)))])
        hist = go.Figure(go.Histogram(x=pnlh, nbinsx=80, marker_color=COLORS["horizon"], opacity=0.85,
                                      hovertemplate="손익 %{x:$,.0f}<br>%{y}회<extra></extra>"))
        hist.add_vline(x=0, line_dash="dot", line_color="#888")
        hist.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                           xaxis_title="손익($)", yaxis_title="횟수", bargap=0.02)
        st.plotly_chart(hist, use_container_width=True)

    # ---- 기초자산별 손익 그래프
    st.subheader("기초자산별 손익 그래프")
    rng_pct = st.slider("가격 범위 (현재가 대비 ±%)", 10, 80, 30, 5, key="capm_rng")
    for code, u in book.items():
        _payoff_chart(code, u, h_days, rng_pct / 100)


def _payoff_chart(code: str, u: dict, h_days: float, rng: float) -> None:
    P = u["P"]
    S0 = P["S"]
    xs = np.linspace(S0 * (1 - rng), S0 * (1 + rng), 401)
    first_T = min([lg["T"] for lg in u["legs"]], default=None)
    curves = {"오늘": (pnl_at(u, xs, 0.0), COLORS["today"], "dot")}
    if h_days > 0 and (first_T is None or abs(h_days / 365 - first_T) > 0.5 / 365):
        curves[f"{h_days:g}일 뒤"] = (pnl_at(u, xs, h_days / 365), COLORS["horizon"], "dash")
    if first_T is not None:
        exp_lbl = next(lg["exp"] for lg in u["legs"] if lg["T"] == first_T)
        curves[f"가장 가까운 만기 {exp_lbl}"] = (pnl_at(u, xs, first_T), COLORS["expiry"], "solid")
    key_name, (ky, _, _) = list(curves.items())[-1]

    fig = go.Figure()
    for name, (y, color, dash) in curves.items():
        fig.add_trace(go.Scatter(x=xs, y=y, name=name, mode="lines", line=dict(color=color, dash=dash, width=2),
                                 hovertemplate="기초자산 %{x:,.2f}<br>손익 %{y:$,.0f}<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line_color="#888", line_width=1)
    fig.add_vline(x=S0, line_dash="dot", line_color="#888", annotation_text=f"현재 {S0:,.2f}", annotation_position="top")
    sgn = np.sign(ky)
    be = [xs[i] - ky[i] * (xs[i + 1] - xs[i]) / (ky[i + 1] - ky[i]) for i in range(len(xs) - 1)
          if sgn[i] != 0 and sgn[i] != sgn[i + 1] and ky[i + 1] != ky[i]]
    for b in be:
        fig.add_trace(go.Scatter(x=[b], y=[0], mode="markers", marker=dict(size=9, color=COLORS["expiry"], symbol="diamond"),
                                 name="손익분기점", showlegend=False, hovertemplate=f"손익분기점 {b:,.2f}<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), font=dict(size=11), hovermode="x unified",
                      xaxis_title=f"{code} 가격($)", yaxis_title="손익($)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    st.markdown(f"**{code}** · 현재가 {S0:,.2f} · 주식 {u['shares']:,.2f}주 · 옵션 {len(u['legs'])}건 · 베타 {u.get('beta', float('nan')):.2f}")
    st.plotly_chart(fig, use_container_width=True)

    mx, mn = float(ky.max()), float(ky.min())
    up_slope = u["shares"] + sum(lg["n"] * MULT for lg in u["legs"] if lg["is_call"])  # 아주 높은 가격에서의 기울기(주)
    up_unlim, dn_unlim = up_slope > 1e-9, up_slope < -1e-9
    cost = sum(lg["n"] * MULT * lg["mkt"] for lg in u["legs"])
    info = pd.DataFrame([{
        "기준": key_name,
        "손익분기점": ", ".join(f"{b:,.2f}" for b in be) if be else "없음(범위 내)",
        "범위 내 최대 이익": f"{_usd(mx)}" + (" (가격이 오를수록 계속 증가 · 이론상 무한)" if up_unlim else ""),
        "범위 내 최대 손실": f"{_usd(mn)}" + (" (가격이 오를수록 손실 계속 증가 · 이론상 무한)" if dn_unlim else ""),
        "옵션 순지급(+)/수취(−) 프리미엄": f"{_usd(cost)}",
    }])
    st.dataframe(info, use_container_width=True, hide_index=True)
