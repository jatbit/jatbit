"""옵션 변동성·수급 분석 (유료 옵션 분석 도구에서 흔히 보는 지표를 무료 데이터로 계산)
- ATM 내재변동성 기간구조, 변동성 스마일, 변동성 곡면(머니니스 × 만기)
- 25델타 스큐(리스크 리버설), 기대 변동폭(ATM 스트래들), 풋/콜 비율(거래량·미결제약정)
- 맥스페인과 행사가별 미결제약정, 딜러 감마 노출(GEX, 추정)
- IV/HV 비율과 HV 랭크(과거 IV 이력은 무료로 없어 역사적 변동성으로 대신)
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bsm
from options import (MULT, OPT_PRESETS, build_chain, is_us_ticker, load_chain_raw, load_expiries, load_underlying,
                     load_us_rate_3m, mgrid, years_to_expiry)


def _params(ticker: str) -> dict:
    u = load_underlying(ticker)
    rr = load_us_rate_3m()
    r = rr[0] if rr else 0.04
    hv60 = bsm.hist_vol(u["close"], 60)
    return {"S": u["spot"], "r": math.log1p(r), "q": u["div12"] / u["spot"] if u["spot"] else 0.0,
            "sig_theo": hv60 if hv60 == hv60 else 0.3, "close": u["close"], "asof": u["asof"]}


def _interp(x, xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if len(xs) < 2:
        return float("nan")
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if x < xs[0] or x > xs[-1]:
        return float("nan")
    return float(np.interp(x, xs, ys))


def max_pain(d: pd.DataFrame):
    c = d[d["is_call"]][["strike", "openInterest"]].fillna(0)
    p = d[~d["is_call"]][["strike", "openInterest"]].fillna(0)
    ks = np.sort(d["strike"].unique())
    pain = []
    for k in ks:
        pc = (c["openInterest"] * np.maximum(k - c["strike"], 0)).sum()
        pp = (p["openInterest"] * np.maximum(p["strike"] - k, 0)).sum()
        pain.append((pc + pp) * MULT)
    pain = np.array(pain)
    if not len(ks) or (c["openInterest"].sum() + p["openInterest"].sum()) <= 0:  # Yahoo가 미결제약정을 0으로 줄 때가 있음
        return float("nan"), ks, pain
    return float(ks[int(np.argmin(pain))]), ks, pain


def expiry_stats(d: pd.DataFrame, S: float, T: float) -> dict:
    calls, puts = d[d["is_call"]], d[~d["is_call"]]
    # ATM IV: 현재가 양옆 행사가의 콜·풋 IV를 보간해 평균
    iv_c = _interp(S, calls["strike"], calls["iv"])
    iv_p = _interp(S, puts["strike"], puts["iv"])
    atm = np.nanmean([iv_c, iv_p]) if np.isfinite([iv_c, iv_p]).any() else float("nan")
    # 25델타: 풋 델타 −0.25, 콜 델타 +0.25 위치의 IV
    p25 = _interp(-0.25, puts["delta"], puts["iv"])
    c25 = _interp(0.25, calls["delta"], calls["iv"])
    # 기대 변동폭: ATM 스트래들 (현재가와 가장 가까운 행사가의 콜+풋 시장가)
    k_atm = d.iloc[(d["strike"] - S).abs().argsort()]["strike"].iloc[0] if len(d) else float("nan")
    cs = calls[calls["strike"] == k_atm]["mkt"]
    ps = puts[puts["strike"] == k_atm]["mkt"]
    straddle = float(cs.iloc[0] + ps.iloc[0]) if len(cs) and len(ps) else float("nan")
    mp, _, _ = max_pain(d)
    cv, pv = calls["volume"].fillna(0).sum(), puts["volume"].fillna(0).sum()
    co, po = calls["openInterest"].fillna(0).sum(), puts["openInterest"].fillna(0).sum()
    return {"T": T, "atm_iv": atm, "p25": p25, "c25": c25, "rr25": p25 - c25 if np.isfinite(p25) and np.isfinite(c25) else float("nan"),
            "straddle": straddle, "move_pct": straddle / S if S else float("nan"),
            "move_iv": atm * math.sqrt(T) if np.isfinite(atm) else float("nan"),
            "max_pain": mp, "pc_vol": pv / cv if cv else float("nan"), "pc_oi": po / co if co and po else float("nan"),
            "call_vol": cv, "put_vol": pv, "call_oi": co, "put_oi": po}


def gex_by_strike(d: pd.DataFrame, S: float) -> pd.Series:
    """딜러가 고객 반대편(콜 매도·풋 매수 고객의 상대방)이라는 통상 가정: 콜 감마 +, 풋 감마 −. 1% 움직일 때 딜러 델타 변화($)."""
    g = d["gamma"].fillna(0) * d["openInterest"].fillna(0) * MULT * S * S * 0.01
    g = np.where(d["is_call"], g, -g)
    return pd.Series(g, index=d["strike"]).groupby(level=0).sum()


@st.cache_data(ttl=300, show_spinner=False)
def _chain_cached(ticker: str, expiry: str, S: float, r: float, q: float, sig: float) -> pd.DataFrame:
    P = {"S": S, "r": r, "q": q, "sig_theo": sig}
    return build_chain(load_chain_raw(ticker, expiry), P, years_to_expiry(expiry), "IV")


def render_vol() -> None:
    st.caption("Yahoo 옵션 체인(약 15분 지연)으로 직접 계산한 값입니다. 만기를 많이 고를수록 조회가 오래 걸립니다.")
    if hasattr(st, "pills"):
        st.pills("빠른 선택", OPT_PRESETS, key="ov_pill",
                 on_change=lambda: st.session_state.__setitem__("ov_ticker", st.session_state.get("ov_pill")) if st.session_state.get("ov_pill") else None)
    ticker = (st.text_input("미국 티커", key="ov_ticker", placeholder="AAPL") or "").strip().upper()
    if not ticker:
        st.info("티커를 입력하거나 위에서 고르세요.")
        return
    if not is_us_ticker(ticker):
        st.warning("미국 종목·ETF만 지원합니다.")
        return
    try:
        exps = load_expiries(ticker)
        P = _params(ticker)
    except Exception as e:  # noqa: BLE001
        st.error(f"데이터를 불러오지 못했습니다: {e}")
        return
    S = P["S"]
    today = pd.Timestamp.now().normalize()
    days = [(pd.Timestamp(e) - today).days for e in exps]
    c1, c2 = st.columns([2, 1])
    max_d = c1.select_slider("포함할 최대 만기", options=[30, 60, 90, 180, 365, 730], value=180, key="ov_maxd",
                             format_func=lambda x: f"{x}일")
    n_max = c2.number_input("최대 만기 개수", 3, 25, 12, key="ov_nmax")
    use = [e for e, dd in zip(exps, days) if 1 <= dd <= max_d][: int(n_max)]
    if not use:
        st.warning("조건에 맞는 만기가 없습니다.")
        return
    chains, stats = {}, []
    prog = st.progress(0.0, text="만기별 옵션 체인을 불러오는 중...")
    for i, e in enumerate(use):
        try:
            d = _chain_cached(ticker, e, S, P["r"], P["q"], P["sig_theo"])
            chains[e] = d
            stt = expiry_stats(d, S, years_to_expiry(e))
            stt["expiry"] = e
            stt["days"] = (pd.Timestamp(e) - today).days
            stats.append(stt)
        except Exception:  # noqa: BLE001
            pass
        prog.progress((i + 1) / len(use), text=f"만기별 옵션 체인을 불러오는 중... ({i + 1}/{len(use)})")
    prog.empty()
    if not stats:
        st.error("옵션 체인을 하나도 불러오지 못했습니다.")
        return
    ST = pd.DataFrame(stats).set_index("expiry")

    # ---- 요약
    close = P["close"]
    lr = np.log(close).diff().dropna()
    hv20 = lr.rolling(20).std() * math.sqrt(252)
    hv_now = float(hv20.iloc[-1])
    hv_y = hv20.iloc[-252:].dropna()
    hv_rank = (hv_now - hv_y.min()) / (hv_y.max() - hv_y.min()) if len(hv_y) and hv_y.max() > hv_y.min() else float("nan")
    hv_pct = float((hv_y < hv_now).mean()) if len(hv_y) else float("nan")
    near = ST[ST["days"] >= 7].iloc[0] if (ST["days"] >= 7).any() else ST.iloc[0]
    tot_cv, tot_pv = ST["call_vol"].sum(), ST["put_vol"].sum()
    oi_ok = (ST["call_oi"] > 0) & (ST["put_oi"] > 0)  # 콜·풋 미결제약정이 모두 있는 만기만 합산
    tot_co, tot_po = ST.loc[oi_ok, "call_oi"].sum(), ST.loc[oi_ok, "put_oi"].sum()
    st.markdown(f"##### {ticker} 요약 · 기초자산 {S:,.2f}")
    mgrid([("ATM IV (가까운 만기)", f"{near['atm_iv'] * 100:.1f}%" if np.isfinite(near["atm_iv"]) else "-", f"{near.name} 만기 기준"),
           ("HV 20일", f"{hv_now * 100:.1f}%"),
           ("IV / HV", f"{near['atm_iv'] / hv_now:.2f}배" if np.isfinite(near["atm_iv"]) and hv_now else "-",
            "1보다 크면 옵션이 최근 실제 변동성보다 비싸게 거래됨(변동성 위험 프리미엄)"),
           ("HV 랭크 (1년)", f"{hv_rank * 100:.0f}%" if np.isfinite(hv_rank) else "-",
            "최근 1년 HV20 범위에서 현재 위치. 과거 IV 이력은 무료로 받을 수 없어 HV로 대신함"),
           ("25델타 스큐", f"{near['rr25'] * 100:+.1f}%p" if np.isfinite(near["rr25"]) else "-", "25델타 풋 IV − 25델타 콜 IV. 클수록 하락 대비 수요가 큼"),
           ("기대 변동폭", f"±{near['straddle']:,.2f} (±{near['move_pct'] * 100:.1f}%)" if np.isfinite(near["straddle"]) else "-",
            f"{near.name} 만기까지 ATM 스트래들 가격 기준"),
           ("풋/콜 거래량 비율", f"{tot_pv / tot_cv:.2f}" if tot_cv else "-", "불러온 전체 만기 합계"),
           ("풋/콜 미결제 비율", f"{tot_po / tot_co:.2f}" if tot_co else "-", f"미결제약정이 있는 {int(oi_ok.sum())}개 만기 합계"),
           ("맥스페인 (가까운 만기)", f"{near['max_pain']:,.2f}" if np.isfinite(near["max_pain"]) else "-",
            "만기 때 옵션 매수자 전체 손실이 가장 커지는(매도자 이익이 최대인) 가격" if np.isfinite(near["max_pain"])
            else "Yahoo가 이 만기의 미결제약정을 0으로 제공해 계산할 수 없음"),
           ("HV 백분위 (1년)", f"{hv_pct * 100:.0f}%" if np.isfinite(hv_pct) else "-")], cols=5)

    sub = st.tabs(["기간구조·스큐", "스마일", "변동성 곡면", "미결제약정·맥스페인", "감마 노출(GEX)", "만기별 표", "변동성 모형 보정"])
    with sub[6]:
        render_calibration(ticker, chains, S, P["r"], P["q"])
    with sub[0]:
        f = go.Figure()
        f.add_trace(go.Scatter(x=ST["days"], y=ST["atm_iv"] * 100, name="ATM IV", mode="lines+markers", line=dict(color="#4C9BE8", width=2)))
        f.add_trace(go.Scatter(x=ST["days"], y=ST["p25"] * 100, name="25델타 풋 IV", mode="lines+markers", line=dict(color="#E5534B", dash="dot")))
        f.add_trace(go.Scatter(x=ST["days"], y=ST["c25"] * 100, name="25델타 콜 IV", mode="lines+markers", line=dict(color="#5BB56E", dash="dot")))
        f.add_hline(y=hv_now * 100, line_dash="dash", line_color="#888", annotation_text=f"HV20 {hv_now * 100:.1f}%")
        f.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                        xaxis=dict(title="잔존일수"), yaxis=dict(title="내재변동성(%)", ticksuffix="%"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(f, use_container_width=True)
    pick = None
    with sub[1]:
        pick = st.selectbox("만기", list(ST.index), key="ov_smile_exp",
                            format_func=lambda e: f"{e} (D-{int(ST.loc[e, 'days'])})")
        d = chains[pick]
        f = go.Figure()
        for is_call, nm, color in ((True, "콜 IV", "#5BB56E"), (False, "풋 IV", "#E5534B")):
            x = d[d["is_call"] == is_call]
            x = x[np.isfinite(x["iv"])]
            f.add_trace(go.Scatter(x=x["strike"], y=x["iv"] * 100, name=nm, mode="lines+markers", marker=dict(size=5),
                                   line=dict(color=color, width=1.5)))
        f.add_vline(x=S, line_dash="dot", line_color="#888", annotation_text=f"현재가 {S:,.2f}")
        lo, hi = S * 0.6, S * 1.4
        f.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                        xaxis=dict(title="행사가", range=[lo, hi]), yaxis=dict(title="내재변동성(%)", ticksuffix="%"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(f, use_container_width=True)
    with sub[2]:
        bins = np.arange(0.70, 1.31, 0.05)
        grid = []
        for e in ST.index:
            d = chains[e]
            otm = d[((~d["is_call"]) & (d["strike"] <= S)) | ((d["is_call"]) & (d["strike"] > S))]
            otm = otm[np.isfinite(otm["iv"])]
            m = otm["strike"] / S
            grid.append([_interp(b, m, otm["iv"]) * 100 for b in bins])
        z = np.array(grid)
        f = go.Figure(go.Heatmap(z=z, x=[f"{b:.2f}" for b in bins], y=[f"{e} (D-{int(ST.loc[e, 'days'])})" for e in ST.index],
                                 colorscale="RdYlBu_r", colorbar=dict(title="IV%"),
                                 hovertemplate="머니니스 %{x}<br>%{y}<br>IV %{z:.1f}%<extra></extra>"))
        f.update_layout(height=max(300, 26 * len(ST) + 80), margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                        xaxis=dict(title="머니니스 (행사가 ÷ 현재가)"), yaxis=dict(autorange="reversed"))
        st.plotly_chart(f, use_container_width=True)
        st.caption("변동성 곡면: 가로는 행사가/현재가, 세로는 만기. 현재가 아래는 외가격 풋, 위는 외가격 콜의 IV(유동성이 더 좋은 쪽)를 썼습니다.")
    with sub[3]:
        e2 = st.selectbox("만기", list(ST.index), key="ov_oi_exp", format_func=lambda e: f"{e} (D-{int(ST.loc[e, 'days'])})")
        d = chains[e2]
        mp, ks, pain = max_pain(d)
        if not np.isfinite(mp):
            st.info("Yahoo가 이 만기의 미결제약정을 0으로 제공하고 있어(장중·신규 만기에 자주 발생) 맥스페인과 미결제약정 분포를 계산할 수 없습니다. "
                    "다른 만기를 고르거나 다음 거래일에 다시 보세요.")
        c = d[d["is_call"]].groupby("strike")["openInterest"].sum()
        p = d[~d["is_call"]].groupby("strike")["openInterest"].sum()
        rng = (S * 0.7, S * 1.3)
        f = go.Figure()
        f.add_trace(go.Bar(x=c.index, y=c.values, name="콜 미결제약정", marker_color="#5BB56E"))
        f.add_trace(go.Bar(x=p.index, y=-p.values, name="풋 미결제약정(아래)", marker_color="#E5534B"))
        if np.isfinite(mp):
            f.add_vline(x=mp, line_color="#9B6FD6", line_dash="dash", annotation_text=f"맥스페인 {mp:,.2f}")
        f.add_vline(x=S, line_color="#888", line_dash="dot", annotation_text="현재가", annotation_position="bottom right")
        f.update_layout(barmode="relative", height=330, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                        xaxis=dict(title="행사가", range=list(rng)), yaxis=dict(title="계약 수"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(f, use_container_width=True)
        f2 = go.Figure(go.Scatter(x=ks, y=pain / 1e6, mode="lines", line=dict(color="#9B6FD6"),
                                  hovertemplate="만기가 %{x:,.2f}일 때 옵션 매수자 총가치 %{y:,.1f}백만$<extra></extra>"))
        if np.isfinite(mp):
            f2.add_vline(x=mp, line_color="#9B6FD6", line_dash="dash")
        f2.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                         xaxis=dict(title="만기 시 주가", range=list(rng)), yaxis=dict(title="옵션 내재가치 합(백만$)"))
        st.plotly_chart(f2, use_container_width=True)
        top_c = c.sort_values(ascending=False).head(3)
        top_p = p.sort_values(ascending=False).head(3)
        st.caption(f"콜 미결제 최다 행사가(저항으로 자주 해석): {', '.join(f'{k:g}' for k in top_c.index)} · "
                   f"풋 미결제 최다 행사가(지지로 자주 해석): {', '.join(f'{k:g}' for k in top_p.index)}. "
                   "맥스페인은 만기 직전 주가가 끌려가는 경향이 있다는 주장이 있지만 근거가 강하지는 않으니 참고용으로만 보세요.")
    with sub[4]:
        e3 = st.selectbox("만기 범위", ["불러온 전체 만기 합계"] + list(ST.index), key="ov_gex_exp")
        if e3.startswith("불러온"):
            g = pd.concat([gex_by_strike(chains[e], S) for e in ST.index]).groupby(level=0).sum()
        else:
            g = gex_by_strike(chains[e3], S)
        g = g[(g.index > S * 0.7) & (g.index < S * 1.3)]
        if not (g != 0).any():
            st.info("선택한 만기의 미결제약정이 Yahoo에서 0으로 와서 감마 노출을 계산할 수 없습니다. '불러온 전체 만기 합계'나 다른 만기를 고르세요.")
        tot = float(g.sum())
        flip = None
        cum = g.sort_index().cumsum()
        sgn = np.sign(cum.values)
        for i in range(1, len(sgn)):
            if sgn[i] != sgn[i - 1] and sgn[i] != 0:
                flip = cum.index[i]
                break
        f = go.Figure(go.Bar(x=g.index, y=g.values / 1e6, marker_color=np.where(g.values >= 0, "#5BB56E", "#E5534B"),
                             hovertemplate="행사가 %{x:,.2f}<br>GEX %{y:,.2f}백만$ /1%<extra></extra>"))
        f.add_vline(x=S, line_color="#888", line_dash="dot", annotation_text="현재가")
        f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                        xaxis=dict(title="행사가"), yaxis=dict(title="감마 노출(백만$, 1% 변동당)"))
        st.plotly_chart(f, use_container_width=True)
        mgrid([("총 GEX (1% 변동당)", f"{tot / 1e6:,.1f}백만$"),
               ("해석", "딜러 순매수 감마 → 변동 완화" if tot > 0 else "딜러 순매도 감마 → 변동 증폭"),
               ("누적 GEX 부호 전환 행사가", f"{flip:,.2f}" if flip is not None else "-")], cols=3)
        st.caption("딜러 포지션은 공개되지 않아 업계 통상 가정(고객이 콜 매수·풋 매수)으로 추정한 값입니다.")
    with sub[5]:
        tbl = ST[["days", "atm_iv", "p25", "c25", "rr25", "straddle", "move_pct", "move_iv", "max_pain", "pc_vol", "pc_oi",
                  "call_vol", "put_vol", "call_oi", "put_oi"]].copy()
        tbl.columns = ["잔존일", "ATM IV", "25Δ 풋 IV", "25Δ 콜 IV", "25Δ 스큐", "스트래들($)", "기대 변동폭(%)", "IV 기준 1σ(%)",
                       "맥스페인", "풋/콜 거래량", "풋/콜 미결제", "콜 거래량", "풋 거래량", "콜 미결제", "풋 미결제"]
        fmt = {"잔존일": "{:.0f}", "ATM IV": "{:.1%}", "25Δ 풋 IV": "{:.1%}", "25Δ 콜 IV": "{:.1%}", "25Δ 스큐": "{:+.1%}",
               "스트래들($)": "{:,.2f}", "기대 변동폭(%)": "{:.1%}", "IV 기준 1σ(%)": "{:.1%}", "맥스페인": "{:,.2f}",
               "풋/콜 거래량": "{:.2f}", "풋/콜 미결제": "{:.2f}", "콜 거래량": "{:,.0f}", "풋 거래량": "{:,.0f}",
               "콜 미결제": "{:,.0f}", "풋 미결제": "{:,.0f}"}
        show = tbl.copy().astype(object)
        for c_, f_ in fmt.items():
            show[c_] = [f_.format(v) if v is not None and np.isfinite(v) else "-" for v in tbl[c_].astype(float)]
        from glossary import col_help
        st.dataframe(show, use_container_width=True, column_config=col_help(show.columns))
        st.caption("Yahoo가 미결제약정을 0으로 준 만기는 맥스페인·풋/콜 미결제가 '-'로 표시됩니다.")


# ----------------------------------------------------------------------
# 변동성 모형 보정 (SVI · SABR · 헤스턴)
# ----------------------------------------------------------------------
def _otm_quotes(d: pd.DataFrame, S: float, T: float, r: float, q: float, kmax: float = 0.5):
    """외가격 옵션(행사가 < 선도가는 풋, 이상은 콜)의 IV. 호가가 있는 것만, 로그 머니니스 |k| ≤ kmax."""
    F = S * math.exp((r - q) * T)
    x = d[np.isfinite(d["iv"]) & (d["iv"] > 0.02) & (d["iv"] < 3) & (d["mkt"].fillna(0) > 0)].copy()  # 장 마감 뒤엔 호가가 0이라 체결가도 씀
    x = x[(x["is_call"] & (x["strike"] >= F)) | (~x["is_call"] & (x["strike"] < F))]
    x["k"] = np.log(x["strike"] / F)
    x = x[x["k"].abs() <= kmax * max(1.0, math.sqrt(T / 0.25))]
    return F, x.sort_values("strike")


@st.cache_data(ttl=600, show_spinner=False)
def _fit_all(ticker: str, keyset: tuple, data: tuple, S: float, r: float, q: float, beta: float, n_hes: int):
    import volmodels as vm

    out = {"svi": {}, "sabr": {}}
    hes_in = []
    for e, T, F, ks, Ks, ivs in data:
        ks, Ks, ivs = np.array(ks), np.array(Ks), np.array(ivs)
        out["svi"][e] = vm.fit_svi(ks, ivs, T)
        out["sabr"][e] = vm.fit_sabr(F, Ks, ivs, T, beta)
        if T >= 10 / 365 and len(Ks) >= 4:
            sel = np.linspace(0, len(Ks) - 1, min(15, len(Ks))).round().astype(int)
            hes_in.append((T, Ks[sel], ivs[sel]))
    hes_in = hes_in[:n_hes]
    out["heston"] = vm.fit_heston(S, r, q, hes_in) if hes_in else {"ok": False, "msg": "10일 이상 만기가 없습니다"}
    return out


def render_calibration(ticker: str, chains: dict, S: float, r: float, q: float) -> None:
    import volmodels as vm

    if vm.least_squares is None:
        st.error("이 기능에는 scipy 패키지가 필요합니다 (requirements.txt에 scipy 추가).")
        return
    c1, c2 = st.columns(2)
    beta = c1.selectbox("SABR β", [0.5, 1.0, 0.0, 0.7], index=0, key="cal_beta",
                        help="1 = 로그정규, 0 = 정규, 주식은 보통 0.5~1을 고정하고 나머지만 맞춥니다")
    n_hes = c2.slider("헤스턴에 쓸 만기 수", 2, 8, 5, key="cal_nh", help="많을수록 정확하지만 오래 걸립니다")
    data = []
    for e, d in chains.items():
        T = years_to_expiry(e)
        if T < 3 / 365:
            continue
        F, x = _otm_quotes(d, S, T, r, q)
        if len(x) >= 5:
            data.append((e, T, F, tuple(x["k"].round(6)), tuple(x["strike"]), tuple(x["iv"].round(6))))
    if not data:
        st.info("보정에 쓸 외가격 옵션 호가가 부족합니다.")
        return
    with st.spinner("SVI·SABR·헤스턴 모형을 시장 IV에 맞추는 중..."):
        Fit = _fit_all(ticker, tuple(e for e, *_ in data), tuple(data), float(S), float(r), float(q), float(beta), int(n_hes))

    # ---- 요약
    H = Fit["heston"]
    svi_r = [v["rmse"] for v in Fit["svi"].values() if v.get("ok")]
    sab_r = [v["rmse"] for v in Fit["sabr"].values() if v.get("ok")]
    st.markdown("###### 모형별 평균 오차")
    mgrid([("SVI RMSE", f"{np.mean(svi_r) * 100:.2f}%p" if svi_r else "-", "만기별 스마일 맞춤"),
           ("SABR RMSE", f"{np.mean(sab_r) * 100:.2f}%p" if sab_r else "-", f"β = {beta:g}"),
           ("헤스턴 RMSE", f"{H['rmse'] * 100:.2f}%p" if H.get("ok") else "-", f"{len(H.get('fits', []))}개 만기를 한 번에 맞춤" if H.get("ok") else H.get("msg", "")),
           ("나비 차익 없는 SVI 만기", f"{sum(v.get('arb_free', False) for v in Fit['svi'].values())}/{len(Fit['svi'])}")], cols=4)
    if H.get("ok"):
        st.markdown("###### 헤스턴 파라미터 (곡면 전체)")
        mgrid([("v0", f"{H['v0']:.4f}", f"현재 변동성 {math.sqrt(H['v0']) * 100:.1f}%"), ("θ", f"{H['theta']:.4f}", f"장기 변동성 {math.sqrt(H['theta']) * 100:.1f}%"),
               ("κ", f"{H['kappa']:.2f}", f"반감기 {math.log(2) / H['kappa'] * 12:.1f}개월"), ("σv", f"{H['sigma']:.2f}"),
               ("ρ", f"{H['rho']:+.2f}"), ("펠러 조건", "충족" if H["feller"] else "위반")], cols=6)

    # ---- 만기별 스마일
    exps = [e for e, *_ in data]
    e = st.selectbox("스마일을 볼 만기", exps, key="cal_exp")
    row = next(x for x in data if x[0] == e)
    _, T, F, ks, Ks, ivs = row
    ks, Ks, ivs = np.array(ks), np.array(Ks), np.array(ivs)
    grid = np.linspace(Ks.min(), Ks.max(), 80)
    kg = np.log(grid / F)
    f = go.Figure()
    f.add_trace(go.Scatter(x=Ks, y=ivs * 100, mode="markers", name="시장 IV (외가격)", marker=dict(color="#888", size=7)))
    sv = Fit["svi"].get(e, {})
    if sv.get("ok"):
        f.add_trace(go.Scatter(x=grid, y=np.sqrt(np.maximum(vm.svi_w(kg, sv["a"], sv["b"], sv["rho"], sv["m"], sv["s"]), 1e-12) / T) * 100,
                               name=f"SVI (오차 {sv['rmse'] * 100:.2f}%p)", line=dict(color="#4C9BE8", width=2)))
    sb = Fit["sabr"].get(e, {})
    if sb.get("ok"):
        f.add_trace(go.Scatter(x=grid, y=vm.sabr_vol(F, grid, T, sb["alpha"], sb["beta"], sb["rho"], sb["nu"]) * 100,
                               name=f"SABR (오차 {sb['rmse'] * 100:.2f}%p)", line=dict(color="#F2A93B", width=2, dash="dash")))
    if H.get("ok"):
        hv = vm.heston_iv(S, grid, T, r, q, (H["v0"], H["kappa"], H["theta"], H["sigma"], H["rho"]))
        f.add_trace(go.Scatter(x=grid, y=hv * 100, name="헤스턴", line=dict(color="#E5534B", width=2, dash="dot")))
    f.add_vline(x=F, line_dash="dot", line_color="#888", annotation_text="선도가")
    f.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                    xaxis_title="행사가", yaxis=dict(title="내재변동성(%)", ticksuffix="%"), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(f, use_container_width=True)

    rows = []
    for e_, T_, *_r in data:
        sv, sb = Fit["svi"].get(e_, {}), Fit["sabr"].get(e_, {})
        rows.append({"만기": e_, "잔존일": int(round(T_ * 365)),
                     "SVI a": f"{sv['a']:.4f}" if sv.get("ok") else "-", "SVI b": f"{sv['b']:.4f}" if sv.get("ok") else "-",
                     "SVI ρ": f"{sv['rho']:+.3f}" if sv.get("ok") else "-", "SVI m": f"{sv['m']:+.3f}" if sv.get("ok") else "-",
                     "SVI σ": f"{sv['s']:.3f}" if sv.get("ok") else "-", "SVI 오차": f"{sv['rmse'] * 100:.2f}%p" if sv.get("ok") else "-",
                     "나비 차익": ("없음" if sv.get("arb_free") else "있음") if sv.get("ok") else "-",
                     "SABR α": f"{sb['alpha']:.4f}" if sb.get("ok") else "-", "SABR ρ": f"{sb['rho']:+.3f}" if sb.get("ok") else "-",
                     "SABR ν": f"{sb['nu']:.3f}" if sb.get("ok") else "-", "SABR 오차": f"{sb['rmse'] * 100:.2f}%p" if sb.get("ok") else "-"})
    from glossary import col_help

    tb = pd.DataFrame(rows)
    st.dataframe(tb, use_container_width=True, hide_index=True, column_config=col_help(tb.columns))
