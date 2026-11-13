"""아메리칸(미국식) 옵션 — 옵션 조회 탭의 하위 탭
- 계산기: CRR 이항트리로 미국식 가격·그릭스, 같은 조건의 유럽식(BSM) 가격과 조기행사 프리미엄, 조기행사 경계
- 체인 비교: 실제 미국 주식옵션(미국식) 시장가로 유럽식 IV와 미국식 IV를 각각 구해 비교
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bsm
from options import OPT_PRESETS, is_us_ticker, load_expiries, load_underlying, load_us_rate_3m, mgrid, years_to_expiry


@st.cache_data(ttl=300, show_spinner=False)
def _am_iv_table(ticker: str, expiry: str, S: float, r: float, q: float, sig: float, band: float, steps: int) -> pd.DataFrame:
    from optvol import _chain_cached

    d = _chain_cached(ticker, expiry, S, r, q, sig)
    T = years_to_expiry(expiry)
    d = d[(d["strike"] >= S * (1 - band)) & (d["strike"] <= S * (1 + band)) & (d["mkt"] > 0)].copy()
    am_iv, prem = [], []
    for _, row in d.iterrows():
        v = bsm.american_iv(row["mkt"], S, row["strike"], T, r, q, bool(row["is_call"]), steps)
        am_iv.append(v)
        if np.isfinite(v):
            eu = float(bsm.price(S, row["strike"], T, r, q, v, bool(row["is_call"])))
            prem.append(row["mkt"] - eu)
        else:
            prem.append(np.nan)
    d["am_iv"] = am_iv
    d["early"] = prem
    return d


def render_american() -> None:
    st.markdown("##### 아메리칸 옵션 계산기")
    if hasattr(st, "pills"):
        st.pills("빠른 선택", OPT_PRESETS, key="am_pill",
                 on_change=lambda: st.session_state.__setitem__("am_ticker", st.session_state.get("am_pill")) if st.session_state.get("am_pill") else None)
    ticker = (st.text_input("미국 티커 (비우면 직접 입력한 값으로만 계산)", key="am_ticker", placeholder="AAPL") or "").strip().upper()
    S0, q0, r0, sig0 = 100.0, 0.0, 0.04, 0.3
    if ticker and is_us_ticker(ticker):
        try:
            u = load_underlying(ticker)
            S0 = float(u["spot"])
            q0 = u["div12"] / S0 if S0 else 0.0
            hv = bsm.hist_vol(u["close"], 60)
            sig0 = hv if hv == hv else 0.3
        except Exception as e:  # noqa: BLE001
            st.warning(f"{ticker} 가격을 불러오지 못해 기본값을 씁니다 ({e}).")
        rr = load_us_rate_3m()
        r0 = rr[0] if rr else 0.04
    k = f"am_{ticker or 'manual'}"
    c1, c2, c3, c4 = st.columns(4)
    S = c1.number_input("기초자산 가격", value=round(S0, 2), step=1.0, key=k + "_S")
    K = c2.number_input("행사가", value=float(round(S0)), step=1.0, key=k + "_K")
    days = c3.number_input("잔존일수", value=90, min_value=1, max_value=3650, step=5, key=k + "_d")
    kind = c4.radio("종류", ["풋", "콜"], horizontal=True, key=k + "_kind")
    c5, c6, c7, c8 = st.columns(4)
    sig = c5.number_input("변동성 (%)", value=round(sig0 * 100, 1), step=1.0, key=k + "_sig") / 100
    r = c6.number_input("무위험금리 (%)", value=round(r0 * 100, 2), step=0.1, key=k + "_r") / 100
    q = c7.number_input("배당수익률 (%)", value=round(q0 * 100, 2), step=0.1, key=k + "_q",
                        help="배당이 클수록 콜의 조기행사 가치가 커집니다") / 100
    steps = c8.select_slider("트리 단계 수", [50, 100, 200, 400, 800], value=200, key=k + "_n",
                             help="많을수록 정확하지만 느립니다")
    T = days / 365.0
    rc = math.log1p(r)
    is_call = kind == "콜"
    A = bsm.american_greeks(S, K, T, rc, q, sig, is_call, int(steps))
    eu = float(bsm.price(S, K, T, rc, q, sig, is_call))
    eg = bsm.greeks(S, K, T, rc, q, sig, is_call)
    prem = A["price"] - eu
    mgrid([("미국식 가격", f"{A['price']:,.4f}"), ("유럽식 가격", f"{eu:,.4f}"),
           ("조기행사 프리미엄", f"{prem:,.4f} ({prem / eu * 100:.2f}%)" if eu > 0 else f"{prem:,.4f}"),
           ("델타", f"{A['delta']:.4f}", f"유럽식 {float(eg['delta']):.4f}"), ("감마", f"{A['gamma']:.5f}", f"유럽식 {float(eg['gamma']):.5f}"),
           ("세타", f"{A['theta']:.4f}", f"하루, 유럽식 {float(eg['theta']):.4f}"), ("베가", f"{A['vega']:.4f}", f"유럽식 {float(eg['vega']):.4f}"),
           ("로", f"{A['rho']:.4f}", f"유럽식 {float(eg['rho']):.4f}")], cols=4)

    t1, t2, t3 = st.tabs(["조기행사 경계", "가격 곡선", "실제 옵션 체인 비교"])
    with t1:
        B = bsm.american_tree(S, K, T, rc, q, sig, is_call, max(int(steps), 200), boundary=True)["boundary"]
        b = pd.DataFrame(B, columns=["t", "S"]).dropna()
        if b.empty:
            st.info("이 조건에서는 만기 전에 행사하는 것이 유리한 구간이 없습니다 (배당 없는 콜은 조기행사하지 않는 것이 최선).")
        else:
            b["left"] = (T - b["t"]) * 365
            b = b[b["left"] > 0.5]
            f = go.Figure()
            f.add_trace(go.Scatter(x=b["left"], y=b["S"], mode="lines", name="조기행사 경계", line=dict(color="#E5534B", width=2),
                                   hovertemplate="만기까지 %{x:.0f}일 · 경계 %{y:,.2f}<extra></extra>"))
            f.add_hline(y=K, line_dash="dot", line_color="#888", annotation_text=f"행사가 {K:g}")
            f.add_hline(y=S, line_dash="dash", line_color="#4C9BE8", annotation_text=f"현재가 {S:g}")
            f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                            xaxis=dict(title="만기까지 남은 일수", autorange="reversed"), yaxis=dict(title="기초자산 가격"))
            st.plotly_chart(f, use_container_width=True)
            now_b = b.iloc[0]["S"] if len(b) else np.nan
            st.write(f"지금 바로 행사가 유리해지는 가격: **{now_b:,.2f}** "
                     + ("이하" if not is_call else "이상") + f" (현재가 {S:,.2f})")
    with t2:
        xs = np.linspace(S * 0.6, S * 1.4, 41)
        am = [bsm.american_price(x, K, T, rc, q, sig, is_call, 120) for x in xs]
        euv = bsm.price(xs, K, T, rc, q, sig, is_call)
        intr = np.maximum((xs - K) if is_call else (K - xs), 0)
        f = go.Figure()
        f.add_trace(go.Scatter(x=xs, y=am, name="미국식", line=dict(color="#E5534B", width=2)))
        f.add_trace(go.Scatter(x=xs, y=euv, name="유럽식", line=dict(color="#4C9BE8", width=1.5, dash="dash")))
        f.add_trace(go.Scatter(x=xs, y=intr, name="내재가치(바로 행사)", line=dict(color="#888", width=1, dash="dot")))
        f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                        xaxis_title="기초자산 가격", yaxis_title="옵션 가격", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(f, use_container_width=True)
    with t3:
        if not ticker or not is_us_ticker(ticker):
            st.info("위에 미국 티커를 넣으면 실제 옵션 체인으로 비교합니다.")
            return
        try:
            exps = load_expiries(ticker)
        except Exception as e:  # noqa: BLE001
            st.error(f"만기 목록을 불러오지 못했습니다: {e}")
            return
        e = st.selectbox("만기", exps, index=min(2, len(exps) - 1), key=k + "_exp")
        band = st.slider("현재가 ± 범위 (%)", 5, 40, 20, key=k + "_band") / 100
        with st.spinner("행사가마다 미국식 내재변동성을 푸는 중..."):
            try:
                d = _am_iv_table(ticker, e, float(S0), math.log1p(r0), float(q0), float(sig0), band, 100)
            except Exception as ex:  # noqa: BLE001
                st.error(f"옵션 체인을 불러오지 못했습니다: {ex}")
                return
        if d.empty:
            st.info("범위 안에 시장가가 있는 옵션이 없습니다.")
            return
        f = go.Figure()
        for c_, nm, col in ((True, "콜", "#4C9BE8"), (False, "풋", "#E5534B")):
            x = d[d["is_call"] == c_]
            f.add_trace(go.Scatter(x=x["strike"], y=x["iv"] * 100, mode="markers", name=f"{nm} 유럽식 IV", marker=dict(color=col, symbol="circle-open", size=8)))
            f.add_trace(go.Scatter(x=x["strike"], y=x["am_iv"] * 100, mode="markers", name=f"{nm} 미국식 IV", marker=dict(color=col, size=6)))
        f.add_vline(x=S0, line_dash="dot", line_color="#888")
        f.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), xaxis_title="행사가",
                        yaxis=dict(title="내재변동성(%)", ticksuffix="%"), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(f, use_container_width=True)
        show = pd.DataFrame({"구분": np.where(d["is_call"], "콜", "풋"), "행사가": d["strike"].map(lambda v: f"{v:g}"),
                             "시장가": d["mkt"].map(lambda v: f"{v:,.2f}"), "유럽식 IV": d["iv"].map(lambda v: f"{v * 100:.1f}%" if np.isfinite(v) else "-"),
                             "미국식 IV": d["am_iv"].map(lambda v: f"{v * 100:.1f}%" if np.isfinite(v) else "-"),
                             "IV 차이": [(f"{(a - b) * 100:+.2f}%p" if np.isfinite(a) and np.isfinite(b) else "-") for a, b in zip(d["iv"], d["am_iv"])],
                             "조기행사 프리미엄": d["early"].map(lambda v: f"{v:,.3f}" if np.isfinite(v) else "-")})
        from glossary import col_help

        st.dataframe(show, use_container_width=True, hide_index=True, column_config=col_help(show.columns))
