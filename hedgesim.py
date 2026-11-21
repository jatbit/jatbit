"""헷지 고도화 + 옵션이 말하는 주가 분포
1) 재헷지 주기별 오차: 옵션 1계약 매도 후 일정 간격으로 델타 헷지 — 실제 변동성 ≠ 내재변동성, 거래비용 반영 몬테카를로
2) 델타·감마·베가 동시 헷지: 주식 + 옵션 2개로 세 그릭스를 0으로, 주가·변동성 충격 전후 비교
3) 스마일 반영 델타: SABR로 맞춘 스마일이 주가와 함께 움직일 때의 델타 vs BSM 델타
4) 위험중립 분포 (Breeden–Litzenberger): 체인 → SVI로 매끈한 스마일 → 콜 가격의 2차 미분 = 만기 주가 확률밀도
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bsm
import ui


# ----------------------------------------------------------------------
# 1) 이산 델타 헷지
# ----------------------------------------------------------------------
def sim_delta_hedge(S0, K, T_days, sig_imp, sig_real, r, is_call, every, cost_bp, n=3000, seed=11, mu=None):
    """옵션 1개 매도(BSM 가격 = 내재변동성) + every 거래일마다 델타 헷지 → 만기 손익 (1주 기준)."""
    rng = np.random.default_rng(seed)
    steps = int(T_days)
    dt = 1 / 252
    mu = r if mu is None else mu
    z = rng.standard_normal((n, steps))
    lnS = np.log(S0) + np.cumsum((mu - 0.5 * sig_real ** 2) * dt + sig_real * math.sqrt(dt) * z, axis=1)
    S = np.hstack([np.full((n, 1), S0), np.exp(lnS)])
    T = steps * dt
    prem = float(bsm.price(S0, K, T, r, 0.0, sig_imp, is_call))
    cash = np.full(n, prem)
    pos = np.zeros(n)
    turnover = np.zeros(n)
    for j in range(steps):
        if j % every == 0:
            Tl = T - j * dt
            dlt = bsm.greeks(S[:, j], K, Tl, r, 0.0, sig_imp, is_call)["delta"]
            trade = dlt - pos
            cash -= trade * S[:, j] + np.abs(trade) * S[:, j] * cost_bp / 1e4
            turnover += np.abs(trade)
            pos = dlt
        cash *= math.exp(r * dt)
    ST = S[:, -1]
    payoff = np.maximum(ST - K, 0) if is_call else np.maximum(K - ST, 0)
    return cash + pos * ST - payoff, turnover


def render_rehedge(ticker, S0, sig0, r) -> None:
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("매도한 옵션", ["콜", "풋"], key="hs_kind")
    days = c2.selectbox("만기", [21, 42, 63, 126], index=1, key="hs_days", format_func=lambda x: f"{x}거래일")
    sig_i = c3.number_input("내재변동성 (가격·델타에 씀, %)", 5.0, 150.0, float(round(sig0 * 100, 1)), 1.0, key="hs_si") / 100
    sig_r = c4.number_input("실제 변동성 (주가가 움직이는 폭, %)", 5.0, 150.0, float(round(sig0 * 100, 1)), 1.0, key="hs_sr") / 100
    c5, c6 = st.columns(2)
    cost = c5.number_input("거래비용 (거래금액의 bp)", 0.0, 50.0, 2.0, 0.5, key="hs_cost")
    mny = c6.number_input("행사가 (현재가 대비 %)", 70.0, 130.0, 100.0, 1.0, key="hs_mny") / 100
    K = S0 * mny
    freqs = [1, 2, 5, 10, 21]
    with st.spinner("경로 3,000개로 재헷지 주기별 손익을 계산하는 중..."):
        rows, dists = [], {}
        for f in freqs:
            pnl, to = sim_delta_hedge(S0, K, days, sig_i, sig_r, r, kind == "콜", f, cost)
            pnl100 = pnl * 100
            dists[f] = pnl100
            rows.append({"재헷지 주기": f"{f}거래일마다", "평균 손익": pnl100.mean(), "표준편차": pnl100.std(),
                         "하위 5%": np.quantile(pnl100, 0.05), "평균 거래량(주)": to.mean() * 100})
    df = pd.DataFrame(rows)
    prem = float(bsm.price(S0, K, days / 252, r, 0.0, sig_i, kind == "콜")) * 100
    vega = float(bsm.greeks(S0, K, days / 252, r, 0.0, sig_i, kind == "콜")["vega"]) * 100
    st.dataframe(df.style.format({"평균 손익": "{:,.1f}", "표준편차": "{:,.1f}", "하위 5%": "{:,.1f}", "평균 거래량(주)": "{:,.0f}"}),
                 hide_index=True, use_container_width=True)
    fig = go.Figure()
    for f, col in ((1, "#D4AF37"), (5, "#4C9BE8"), (21, "#888")):
        fig.add_trace(go.Histogram(x=dists[f], name=f"{f}일마다", opacity=0.55, nbinsx=70, marker_color=col))
    fig.update_layout(barmode="overlay", height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="만기 손익 (1계약 = 100주)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)
    exp_gain = vega * (sig_i - sig_r) * 100
    st.caption(f"옵션 1계약 매도 프리미엄 {prem:,.1f}. 내재 > 실제 변동성이면 평균적으로 이익(대략 베가 × 차이 ≈ {exp_gain:,.0f}), "
               "재헷지를 자주 할수록 표준편차(헷지 오차)가 줄지만 거래비용이 늘어납니다. 이론적으로 오차는 재헷지 횟수의 제곱근에 반비례합니다. "
               "주가는 위험중립 추세(무위험금리)로 움직인다고 가정했습니다.")


# ----------------------------------------------------------------------
# 2) 델타·감마·베가
# ----------------------------------------------------------------------
def solve_three(port, h1, h2):
    """옵션 h1, h2 수량으로 감마·베가를 0, 남은 델타는 주식으로. port/h = dict(delta, gamma, vega)."""
    A = np.array([[h1["gamma"], h2["gamma"]], [h1["vega"], h2["vega"]]], float)
    b = -np.array([port["gamma"], port["vega"]], float)
    if abs(np.linalg.det(A)) < 1e-14:
        return None
    n1, n2 = np.linalg.solve(A, b)
    ns = -(port["delta"] + n1 * h1["delta"] + n2 * h2["delta"])
    return float(ns), float(n1), float(n2)


def render_dgv(ticker, chains: dict, S, r, q) -> None:
    """지금 장부의 이 티커 옵션(없으면 등가격 스트래들 매도 예시)을 옵션 2개 + 주식으로 헷지."""
    ss = st.session_state
    book = ss.get("hedge_opts_cur")
    legs = []
    if book is not None and len(book):
        for rw in book.to_dict("records"):
            if str(rw.get("기초자산", "")).upper() == ticker and rw.get("종류") in ("콜", "풋") and str(rw.get("만기"))[:10] in chains:
                legs.append((str(rw["만기"])[:10], rw["종류"] == "콜", float(rw["행사가"]), float(rw["수량(계약)"])))
    exps = list(chains)
    if not legs:
        e0 = exps[min(1, len(exps) - 1)]
        d0 = chains[e0]
        k0 = float(d0.iloc[(d0["strike"] - S).abs().argsort().iloc[0]]["strike"])
        legs = [(e0, True, k0, -1.0), (e0, False, k0, -1.0)]
        st.caption(f"'내 포지션'에 {ticker} 옵션이 없어 예시로 {e0} 등가격 스트래들 1계약 매도를 헷지합니다.")

    def g_of(e, is_call, K):
        d = chains[e]
        m = d[(d["is_call"] == is_call) & np.isclose(d["strike"], K)]
        if not len(m):
            return None
        rw = m.iloc[0]
        return {"delta": float(rw["delta"]) * 100, "gamma": float(rw["gamma"]) * 100, "vega": float(rw["vega"]) * 100,
                "px": float(rw["mkt"]), "sig": float(rw["sig_g"]), "T": None}

    port = {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
    for e, c, K, n in legs:
        g = g_of(e, c, K)
        if g:
            for k in port:
                port[k] += n * g[k]
    st.caption(f"헷지할 포지션 — 델타 {port['delta']:,.1f}주 · 감마 {port['gamma']:,.3f} · 베가 {port['vega']:,.1f}/1%p")
    c1, c2, c3, c4 = st.columns(4)
    e1 = c1.selectbox("헷지 옵션 1 만기", exps, index=0, key="dgv_e1")
    e2 = c3.selectbox("헷지 옵션 2 만기", exps, index=min(len(exps) - 1, 3), key="dgv_e2")

    def ks(e):
        return sorted(chains[e]["strike"].unique().tolist())

    k1s, k2s = ks(e1), ks(e2)
    K1 = c2.selectbox("행사가 (콜)", k1s, index=int(np.argmin([abs(k - S) for k in k1s])), key=f"dgv_k1_{e1}", format_func=lambda x: f"{x:g}")
    K2 = c4.selectbox("행사가 (콜)", k2s, index=int(np.argmin([abs(k - S) for k in k2s])), key=f"dgv_k2_{e2}", format_func=lambda x: f"{x:g}")
    h1, h2 = g_of(e1, True, K1), g_of(e2, True, K2)
    if not h1 or not h2:
        st.warning("고른 옵션의 가격이 없습니다.")
        return
    sol = solve_three(port, h1, h2)
    if sol is None or (e1 == e2 and K1 == K2):
        st.warning("두 헷지 옵션의 감마·베가 비율이 같아 풀 수 없습니다. 만기가 다른 옵션을 고르세요 (단기는 감마, 장기는 베가가 큼).")
        return
    ns, n1, n2 = sol
    ui.cards([("주식", f"{ns:+,.0f}주"), (f"{e1} 콜 {K1:g}", f"{n1:+,.2f}계약"), (f"{e2} 콜 {K2:g}", f"{n2:+,.2f}계약")], cols=3)
    # 충격 전후
    def reval(dS, dv, with_hedge):
        tot = 0.0
        allp = [(e, c, K, n) for e, c, K, n in legs]
        if with_hedge:
            allp += [(e1, True, K1, n1), (e2, True, K2, n2)]
        for e, c, K, n in allp:
            d = chains[e]
            mm = d[(d["is_call"] == c) & np.isclose(d["strike"], K)]
            if not len(mm):
                continue  # 체인에 없는 행사가는 헷지 대상 계산(위)에서도 빠졌으므로 여기서도 뺀다
            m = mm.iloc[0]
            T = float(m["T"]) if "T" in m.index and np.isfinite(m["T"]) and m["T"] > 0 else \
                max((pd.Timestamp(e) - pd.Timestamp.now().normalize()).days, 1) / 365
            v0 = float(bsm.price(S, K, T, r, q, m["sig_g"], c))
            v1 = float(bsm.price(S * (1 + dS), K, T, r, q, m["sig_g"] + dv, c))
            tot += n * 100 * (v1 - v0)
        if with_hedge:
            tot += ns * S * dS
        return tot

    sc = [("주가 −5%", -0.05, 0.0), ("주가 +5%", 0.05, 0.0), ("변동성 +5%p", 0.0, 0.05), ("변동성 −5%p", 0.0, -0.05),
          ("주가 −10% & 변동성 +10%p", -0.10, 0.10)]
    tb = pd.DataFrame([{"충격": nm, "헷지 전 손익": reval(a, b, False), "헷지 후 손익": reval(a, b, True)} for nm, a, b in sc])
    st.dataframe(tb.style.format({"헷지 전 손익": "{:,.0f}", "헷지 후 손익": "{:,.0f}"}), hide_index=True, use_container_width=True)
    st.caption("단기 옵션은 감마가, 장기 옵션은 베가가 커서 만기가 다른 두 옵션을 섞어야 감마와 베가를 함께 없앨 수 있습니다. "
               "손익은 각 옵션의 IV를 그대로 두고 충격만 준 BSM 재평가입니다 (달러, 1계약 = 100주).")


# ----------------------------------------------------------------------
# 3) SABR 델타
# ----------------------------------------------------------------------
def render_sabr_delta(chains: dict, S, r, q) -> None:
    import volmodels as vm
    from optvol import _otm_quotes

    exps = list(chains)
    e = st.selectbox("만기", exps, index=min(1, len(exps) - 1), key="sd_e")
    d = chains[e]
    T = max((pd.Timestamp(e) - pd.Timestamp.now().normalize()).days, 1) / 365
    F, x = _otm_quotes(d, S, T, r, q)
    beta = st.select_slider("SABR β", [0.0, 0.5, 1.0], value=0.5, key="sd_beta", help="1에 가까울수록 로그정규, 0이면 정규")
    fit = vm.fit_sabr(F, x["strike"].values, x["iv"].values, T, beta)
    if not fit.get("ok"):
        st.warning("이 만기는 외가격 옵션이 부족해 SABR를 맞추지 못했습니다.")
        return
    a_, rho, nu = fit["alpha"], fit["rho"], fit["nu"]
    Ks = x["strike"].values
    h = F * 1e-3
    s0 = vm.sabr_vol(F, Ks, T, a_, beta, rho, nu)
    sp, sm = vm.sabr_vol(F + h, Ks, T, a_, beta, rho, nu), vm.sabr_vol(F - h, Ks, T, a_, beta, rho, nu)
    dsdF = (sp - sm) / (2 * h)
    rows = []
    for K, sig, ds in zip(Ks, s0, dsdF):
        is_call = K >= F
        g = bsm.greeks(S, K, T, r, q, sig, is_call)
        d_sabr = float(g["delta"]) + float(g["vega"]) * 100 * ds * math.exp((r - q) * T)
        rows.append({"행사가": K, "종류": "콜" if is_call else "풋", "SABR IV(%)": sig * 100, "BSM 델타": float(g["delta"]),
                     "SABR 델타": d_sabr, "차이 (1계약 주식 수)": (d_sabr - float(g["delta"])) * 100})
    tb = pd.DataFrame(rows)
    st.dataframe(tb.style.format({"행사가": "{:g}", "SABR IV(%)": "{:.1f}", "BSM 델타": "{:.3f}", "SABR 델타": "{:.3f}",
                                  "차이 (1계약 주식 수)": "{:+.1f}"}), hide_index=True, use_container_width=True, height=320)
    st.caption(f"SABR 적합: α {a_:.3f} · ρ {rho:.2f} · ν {nu:.2f} (오차 {fit['rmse'] * 100:.2f}%p). SABR 델타 = BSM 델타 + 베가 × ∂σ/∂F — "
               f"주가가 오를 때 스마일이 같이 움직이는 효과까지 넣은 델타입니다. ρ가 음수(주식에서 흔함)면 주가가 오를 때 IV가 내려가 "
               "콜의 실제 델타가 BSM보다 작습니다.")


# ----------------------------------------------------------------------
# 4) 위험중립 분포
# ----------------------------------------------------------------------
def rnd_from_chain(d: pd.DataFrame, S: float, T: float, r: float, q: float):
    """반환: dict(K, pdf, F, probs …) 또는 None. SVI로 매끈한 IV → BSM 콜 가격 → e^{rT} ∂²C/∂K²."""
    import volmodels as vm
    from optvol import _otm_quotes

    F, x = _otm_quotes(d, S, T, r, q)
    if len(x) < 6:
        return None
    fit = vm.fit_svi(x["k"].values, x["iv"].values, T)
    if not fit.get("ok"):
        return None
    p = (fit["a"], fit["b"], fit["rho"], fit["m"], fit["s"])
    kmin, kmax = float(x["k"].min()), float(x["k"].max())
    span = max(4 * math.sqrt(max(vm.svi_w(0.0, *p), 1e-6)), 0.3)
    kk = np.linspace(min(kmin, -span), max(kmax, span), 1201)
    Kg = F * np.exp(kk)
    iv = np.sqrt(np.maximum(vm.svi_w(kk, *p), 1e-10) / T)
    C = bsm.price(S, Kg, T, r, q, iv, True)
    dK = np.gradient(Kg)
    pdf = math.exp(r * T) * np.gradient(np.gradient(C, Kg), Kg)
    pdf = np.where(np.isfinite(pdf) & (pdf > 0), pdf, 0.0)
    mass = float((pdf * dK).sum())
    if mass <= 0:
        return None
    pdf /= mass
    cdf = np.cumsum(pdf * dK)

    def prob_above(level):
        return float(1 - np.interp(level, Kg, cdf))

    q05, q50, q95 = (float(np.interp(a, cdf, Kg)) for a in (0.05, 0.5, 0.95))
    atm_iv = float(np.sqrt(max(vm.svi_w(0.0, *p), 1e-10) / T))
    return dict(K=Kg, pdf=pdf, F=F, T=T, atm_iv=atm_iv, mass=mass, rmse=fit["rmse"], arb_free=fit["arb_free"],
                p_up=prob_above(S), p_up10=prob_above(S * 1.1), p_dn10=1 - prob_above(S * 0.9),
                q05=q05, q50=q50, q95=q95, mean=float((Kg * pdf * dK).sum()))


def render_rnd(ticker, chains: dict, S, r, q) -> None:
    exps = list(chains)
    today = pd.Timestamp.now().normalize()
    days = [(pd.Timestamp(e) - today).days for e in exps]
    default = int(np.argmin([abs(dd - 60) for dd in days]))
    e = st.selectbox("만기", exps, index=default, key="rnd_e", format_func=lambda x: f"{x} (D-{(pd.Timestamp(x) - today).days})")
    T = max((pd.Timestamp(e) - today).days, 1) / 365
    R = rnd_from_chain(chains[e], S, T, r, q)
    if R is None:
        st.warning("이 만기는 외가격 옵션 호가가 부족해 분포를 뽑지 못했습니다. 거래가 많은 만기를 고르세요.")
        return
    ui.cards([("오를 확률 (현재가 위)", f"{R['p_up'] * 100:.0f}%"), ("10% 넘게 오를 확률", f"{R['p_up10'] * 100:.0f}%"),
              ("10% 넘게 빠질 확률", f"{R['p_dn10'] * 100:.0f}%"), ("90% 구간", f"{R['q05']:,.0f} – {R['q95']:,.0f}"),
              ("중앙값", f"{R['q50']:,.1f}")], cols=5)
    sig = R["atm_iv"]
    mu = math.log(S) + (r - q - 0.5 * sig * sig) * T
    ln = np.exp(-0.5 * ((np.log(R["K"]) - mu) / (sig * math.sqrt(T))) ** 2) / (R["K"] * sig * math.sqrt(2 * math.pi * T))
    m = (R["K"] > S * 0.5) & (R["K"] < S * 1.6)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=R["K"][m], y=R["pdf"][m], name="옵션이 말하는 분포", fill="tozeroy", line=dict(color="#D4AF37")))
    fig.add_trace(go.Scatter(x=R["K"][m], y=ln[m], name=f"로그정규 (ATM IV {sig * 100:.1f}%)", line=dict(color="#888", dash="dash")))
    fig.add_vline(x=S, line_dash="dot", line_color="#888", annotation_text="현재가")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title=f"{e} 만기 주가", yaxis_title="확률밀도",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("체인의 외가격 옵션 IV를 SVI로 매끈하게 이은 뒤 콜 가격을 행사가로 두 번 미분해 얻은 위험중립 분포입니다 "
               "(Breeden–Litzenberger). 실제 확률이 아니라 위험 프리미엄이 섞인 확률이라 하락 쪽 꼬리가 실제보다 두껍게 나오는 편입니다. "
               + ("" if R["arb_free"] else "이 만기의 SVI 적합에 나비 차익이 있어 밀도 일부를 0으로 잘랐습니다."))


@st.cache_data(ttl=30 * 60, show_spinner=False)
def rnd_6m(ticker: str):
    """기업 리서치용: 6개월에 가장 가까운 만기(60일 이상)의 위험중립 분포 요약. 없으면 None."""
    from options_data import build_chain, load_chain_raw, load_expiries, market_params, years_to_expiry

    P = market_params(ticker, "rs_rnd", show=False)
    exps = load_expiries(ticker)
    today = pd.Timestamp.now().normalize()
    cand = [e for e in exps if (pd.Timestamp(e) - today).days >= 60]
    if not cand:
        return None
    e = min(cand, key=lambda x: abs((pd.Timestamp(x) - today).days - 182))
    T = years_to_expiry(e)
    R = rnd_from_chain(build_chain(load_chain_raw(ticker, e), P, T, "IV"), P["S"], T, P["r"], P["q"])
    if R is None:
        return None
    return {k: R[k] for k in ("p_up", "p_up10", "p_dn10", "q05", "q50", "q95", "atm_iv")} | {"expiry": e, "S": P["S"]}


# ----------------------------------------------------------------------
# 탭 진입점
# ----------------------------------------------------------------------
def _chains_for(ticker, n=6):
    from options_data import build_chain, load_chain_raw, load_expiries, market_params, years_to_expiry

    P = market_params(ticker, "hs", show=False)
    exps = load_expiries(ticker)
    today = pd.Timestamp.now().normalize()
    use = [e for e in exps if 3 <= (pd.Timestamp(e) - today).days <= 400][:n]
    chains = {}
    for e in use:
        try:
            d = build_chain(load_chain_raw(ticker, e), P, years_to_expiry(e), "IV")
            d["T"] = years_to_expiry(e)
            chains[e] = d
        except Exception:  # noqa: BLE001
            continue
    return P, chains


def render(ticker: str | None) -> None:
    """'내 포지션 → 헷지 시뮬레이션' 탭."""
    from options_data import is_us_ticker, load_underlying, load_us_rate_3m

    mode = st.radio("보기", ["재헷지 주기별 오차", "델타·감마·베가 헷지", "스마일 반영 델타"], horizontal=True, key="hs_mode",
                    label_visibility="collapsed")
    if mode == "재헷지 주기별 오차":
        S0, sig0 = 100.0, 0.3
        if ticker and is_us_ticker(ticker):
            try:
                u = load_underlying(ticker)
                S0 = float(u["spot"])
                sig0 = bsm.hist_vol(u["close"], 60)
                sig0 = sig0 if sig0 == sig0 else 0.3
            except Exception:  # noqa: BLE001
                pass
        rr = load_us_rate_3m()
        render_rehedge(ticker, S0, sig0, math.log1p(rr[0] if rr else 0.04))
        return
    if not ticker or not is_us_ticker(ticker):
        st.info("맨 위에서 미국 티커를 고르면 그 종목의 옵션 체인으로 계산합니다.")
        return
    with st.spinner("만기별 체인을 불러오는 중..."):
        P, chains = _chains_for(ticker)
    if len(chains) < 2:
        st.warning("옵션 체인을 충분히 불러오지 못했습니다.")
        return
    if mode == "델타·감마·베가 헷지":
        render_dgv(ticker, chains, P["S"], P["r"], P["q"])
    else:
        render_sabr_delta(chains, P["S"], P["r"], P["q"])


__all__ = [n for n in dir() if not n.startswith("__")]
