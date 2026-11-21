"""옵션 전략 — 체인·전략 탭의 '전략 만들기'
- 템플릿(불 콜 스프레드, 스트랭글, 아이언 콘도르 …)으로 다리(leg)를 만들고 표에서 고칠 수 있다
- 손익: 만기 손익과 선택한 날의 손익(BSM 재평가), 손익분기점, 최대 이익·손실, 이익 확률(위험중립 로그정규), 그릭스 합계
- 합성 백테스트: 과거 옵션 시세가 무료로 없어 과거 주가 + 역사적 변동성으로 BSM 가격을 다시 만들어 매달 같은 전략을 돌린다
  (손절 룰 유무 비교). 스큐·호가 차이·유동성은 반영되지 않으므로 방향성 참고용.
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bsm

# (종류, 간격 배수, 수량): 종류 C/P/S(주식), 간격 배수 × 간격(%)만큼 현재가에서 떨어진 행사가
TEMPLATES = {
    "불 콜 스프레드": [("C", 0, 1), ("C", 1, -1)],
    "베어 풋 스프레드": [("P", 0, 1), ("P", -1, -1)],
    "불 풋 스프레드 (매도)": [("P", 0, -1), ("P", -1, 1)],
    "롱 스트래들": [("C", 0, 1), ("P", 0, 1)],
    "숏 스트래들": [("C", 0, -1), ("P", 0, -1)],
    "롱 스트랭글": [("C", 1, 1), ("P", -1, 1)],
    "숏 스트랭글": [("C", 1, -1), ("P", -1, -1)],
    "아이언 콘도르": [("P", -2, 1), ("P", -1, -1), ("C", 1, -1), ("C", 2, 1)],
    "콜 버터플라이": [("C", -1, 1), ("C", 0, -2), ("C", 1, 1)],
    "커버드 콜": [("S", 0, 1), ("C", 1, -1)],
    "프로텍티브 풋": [("S", 0, 1), ("P", -1, 1)],
    "칼라": [("S", 0, 1), ("P", -1, 1), ("C", 1, -1)],
    "콜 매수": [("C", 0, 1)],
    "풋 매수": [("P", 0, 1)],
    "콜 매도": [("C", 1, -1)],
    "풋 매도": [("P", -1, -1)],
}
KIND_KO = {"C": "콜", "P": "풋", "S": "주식"}
KO_KIND = {v: k for k, v in KIND_KO.items()}


def nice_step(S: float) -> float:
    """체인이 없을 때 행사가를 반올림할 간격 (가격대별)."""
    if S <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(S))
    for m in (0.025, 0.05, 0.1):
        if S / (m * e) <= 40:
            return m * e
    return 0.1 * e


def pick_strike(target: float, strikes) -> float:
    if strikes is not None and len(strikes):
        a = np.asarray(sorted(strikes), float)
        return float(a[np.argmin(np.abs(a - target))])
    st_ = nice_step(target)
    return round(target / st_) * st_


def build_legs(template: str, S: float, width: float, qty: int, strikes=None) -> pd.DataFrame:
    rows = []
    atm = pick_strike(S, strikes)
    for kind, off, n in TEMPLATES.get(template, []):
        if kind == "S":
            rows.append({"종류": "주식", "행사가": np.nan, "수량(계약)": n * qty})
            continue
        K = pick_strike(S * (1 + off * width), strikes) if off else atm
        rows.append({"종류": KIND_KO[kind], "행사가": K, "수량(계약)": n * qty})
    return pd.DataFrame(rows, columns=["종류", "행사가", "수량(계약)"])


def price_legs(legs: pd.DataFrame, ctx: dict) -> list:
    """표 → 계산용 다리 목록. 체인이 있으면 시장가·IV, 없으면 ctx['sig']로 BSM 이론가."""
    S, T, r, q = ctx["S"], ctx["T"], ctx["r"], ctx["q"]
    chain = ctx.get("chain")
    out = []
    for rw in legs.to_dict("records"):
        kind = KO_KIND.get(str(rw.get("종류")))
        n = pd.to_numeric(rw.get("수량(계약)"), errors="coerce")
        if kind is None or not np.isfinite(n) or n == 0:
            continue
        if kind == "S":
            out.append({"kind": "S", "K": np.nan, "n": float(n), "px": S, "iv": np.nan})
            continue
        K = pd.to_numeric(rw.get("행사가"), errors="coerce")
        if not np.isfinite(K) or K <= 0:
            continue
        is_call = kind == "C"
        px, iv = np.nan, np.nan
        if chain is not None and len(chain):
            m = chain[(chain["is_call"] == is_call) & (np.isclose(chain["strike"], K))]
            if len(m):
                px, iv = float(m["mkt"].iloc[0]), float(m["sig_g"].iloc[0])
        man_px = pd.to_numeric(rw.get("가격(직접)"), errors="coerce")
        if np.isfinite(man_px) and man_px > 0:
            px = float(man_px)
            iv = float(bsm.implied_vol(np.array([px]), S, np.array([K]), T, r, q, np.array([is_call]))[0])
        if not np.isfinite(iv) or iv <= 0:
            iv = ctx["sig"]
        if not np.isfinite(px) or px <= 0:
            px = float(bsm.price(S, K, T, r, q, iv, is_call))
        out.append({"kind": kind, "K": float(K), "n": float(n), "px": px, "iv": iv})
    return out


def value_at(legs: list, S_grid, T_left: float, ctx: dict, mult: float) -> np.ndarray:
    """기초자산 가격별 포지션 손익 (진입가 대비). T_left = 남은 기간(년), 0이면 만기."""
    S_grid = np.asarray(S_grid, float)
    pnl = np.zeros_like(S_grid)
    r, q = ctx["r"], ctx["q"]
    for lg in legs:
        if lg["kind"] == "S":
            pnl += lg["n"] * mult * (S_grid - lg["px"])
            continue
        c = lg["kind"] == "C"
        if T_left <= 1e-6:
            v = np.maximum(S_grid - lg["K"], 0) if c else np.maximum(lg["K"] - S_grid, 0)
        else:
            v = bsm.price(S_grid, lg["K"], T_left, r, q, lg["iv"], c)
        pnl += lg["n"] * mult * (v - lg["px"])
    return pnl


def stats(legs: list, ctx: dict, mult: float) -> dict:
    S, T, r, q = ctx["S"], ctx["T"], ctx["r"], ctx["q"]
    Ks = [lg["K"] for lg in legs if lg["kind"] != "S"] or [S]
    hi = max(max(Ks), S) * 3.0
    g = np.linspace(hi / 4000, hi, 4000)  # 0 근처부터 (주가가 0으로 가도 손익은 유한)
    pe = value_at(legs, g, 0.0, ctx, mult)
    slope_hi = (pe[-1] - pe[-50]) / (g[-1] - g[-50])  # 주가가 끝없이 오를 때의 기울기 → 무제한 여부
    max_p = np.inf if slope_hi > 1e-9 else float(pe.max())
    max_l = -np.inf if slope_hi < -1e-9 else float(pe.min())
    sgn = np.sign(pe)
    be = [float(g[i] - pe[i] * (g[i + 1] - g[i]) / (pe[i + 1] - pe[i])) for i in range(len(g) - 1)
          if sgn[i] != 0 and sgn[i] != sgn[i + 1] and sgn[i + 1] != 0]
    ivs = [lg["iv"] for lg in legs if lg["kind"] != "S" and np.isfinite(lg["iv"])]
    sig = float(np.mean(ivs)) if ivs else ctx["sig"]
    # 위험중립 로그정규로 만기 손익 > 0 확률
    if T > 0 and sig > 0:
        mu = math.log(S) + (r - q - 0.5 * sig * sig) * T
        sd = sig * math.sqrt(T)
        z = (np.log(g) - mu) / sd
        dens = np.exp(-0.5 * z * z) / (g * sd * math.sqrt(2 * math.pi))
        w = dens * np.gradient(g)
        pop = float(w[pe > 0].sum() / w.sum())
        ev = float((w * pe).sum() / w.sum())
    else:
        pop, ev = np.nan, np.nan
    prem = -sum(lg["n"] * mult * lg["px"] for lg in legs if lg["kind"] != "S")
    gk = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    for lg in legs:
        if lg["kind"] == "S":
            gk["delta"] += lg["n"] * mult
            continue
        gg = bsm.greeks(S, lg["K"], T, r, q, lg["iv"], lg["kind"] == "C")
        for k in gk:
            gk[k] += lg["n"] * mult * float(gg[k])
    return dict(max_p=max_p, max_l=max_l, be=be, pop=pop, ev=ev, prem=prem, sig=sig, **gk)


def _m(x, cur):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if np.isinf(x):
        return "무제한"
    if cur == "$":
        return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"
    return f"{x:,.0f}{cur}"


def render_builder(ctx: dict, key: str, on_add=None) -> None:
    """ctx: S, T(년), r(연속), q, sig(체인 없을 때 변동성), strikes(list|None), chain(DataFrame|None),
    mult(1계약 주식 수 또는 지수 승수), cur('$' 또는 '원'), label(예: 'AAPL 2026-12-19'), hist(백테스트용 종가 Series|None)."""
    import ui

    S, mult, cur = ctx["S"], ctx["mult"], ctx["cur"]
    c1, c2, c3 = st.columns([2, 1.2, 1])
    tmpl = c1.selectbox("전략", list(TEMPLATES) + ["직접 구성"], key=f"{key}_tmpl")
    width = c2.number_input("행사가 간격 (현재가 대비 %)", 1.0, 30.0, 5.0, 1.0, key=f"{key}_w") / 100
    qty = int(c3.number_input("세트 수", 1, 1000, 1, key=f"{key}_q"))
    base = build_legs(tmpl, S, width, qty, ctx.get("strikes")) if tmpl != "직접 구성" else \
        pd.DataFrame([{"종류": "콜", "행사가": pick_strike(S, ctx.get("strikes")), "수량(계약)": 1}])
    base["가격(직접)"] = np.nan
    legs_df = st.data_editor(
        base, key=f"{key}_ed_{tmpl}_{width}_{qty}_{ctx.get('label')}", num_rows="dynamic", hide_index=True,
        use_container_width=True,
        column_config={"종류": st.column_config.SelectboxColumn(options=["콜", "풋", "주식"], required=True),
                       "행사가": st.column_config.NumberColumn(format="%.2f"),
                       "수량(계약)": st.column_config.NumberColumn(step=1, help="매수 +, 매도 − (주식은 계약 단위, 1 = 1계약분 주식)"),
                       "가격(직접)": st.column_config.NumberColumn(format="%.2f", help="비우면 체인 시장가(없으면 이론가)")})
    legs = price_legs(legs_df, ctx)
    if not legs:
        st.info("다리를 하나 이상 넣어 주세요.")
        return
    s = stats(legs, ctx, mult)
    be_txt = ", ".join(f"{b:,.2f}" for b in s["be"][:4]) or "없음"
    ui.cards([("순 프리미엄", _m(s["prem"], cur), "받으면 +, 내면 −"),
              ("최대 이익 (만기)", _m(s["max_p"], cur)), ("최대 손실 (만기)", _m(s["max_l"], cur)),
              ("손익분기점", be_txt), ("이익 확률", f"{s['pop'] * 100:.0f}%" if np.isfinite(s["pop"]) else "-",
                                    f"위험중립 로그정규, 변동성 {s['sig'] * 100:.1f}%")], cols=5)
    days = max(1, int(round(ctx["T"] * 365)))
    h = st.slider("손익을 볼 날 (오늘부터 며칠 뒤)", 0, days, 0, key=f"{key}_h")
    rng = st.slider("가격 범위 (현재가 대비 ±%)", 5, 60, 20, 5, key=f"{key}_rng") / 100
    g = np.linspace(S * (1 - rng), S * (1 + rng), 301)
    pe = value_at(legs, g, 0.0, ctx, mult)
    ph = value_at(legs, g, max(ctx["T"] - h / 365, 0.0), ctx, mult)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g, y=pe, name="만기", line=dict(color="#D4AF37", width=2.4),
                             hovertemplate="%{x:,.2f} → %{y:,.0f}<extra>만기</extra>"))
    if h < days:
        fig.add_trace(go.Scatter(x=g, y=ph, name=f"{h}일 뒤", line=dict(color="#4C9BE8", dash="dash"),
                                 hovertemplate="%{x:,.2f} → %{y:,.0f}<extra></extra>"))
    fig.add_hline(y=0, line_color="#888", line_dash="dot")
    fig.add_vline(x=S, line_color="#888", line_dash="dot", annotation_text="현재가")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified",
                      xaxis_title="기초자산 가격", yaxis_title=f"손익({cur})",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"그릭스 합계 — 델타 {s['delta']:,.1f}주 · 감마 {s['gamma']:,.3f} · 베가 {_m(s['vega'], cur)}/1%p · "
               f"세타 {_m(s['theta'], cur)}/일. 1계약 = {mult:g}{'주' if mult < 1000 else ' (지수 승수)'}.")
    if on_add is not None:
        st.button("이 전략을 내 포지션에 추가", key=f"{key}_add", on_click=on_add, args=(legs,), type="primary")
        msg = st.session_state.pop(f"{key}_addmsg", None)
        if msg:
            st.success(msg)
    if ctx.get("hist") is not None:
        exp, op = ui.more("이 전략을 과거에 돌려보기 (합성 백테스트)", key=f"{key}_bt")
        if op:
            with exp:
                render_backtest(tmpl if tmpl != "직접 구성" else None, legs_df, width, ctx, key)


# ----------------------------------------------------------------------
# 합성 백테스트
# ----------------------------------------------------------------------
def _offsets(tmpl, legs_df, S, width):
    """템플릿이면 그 정의, 직접 구성이면 표의 행사가를 현재가 대비 비율로 바꿔 매달 같은 모양으로 쓴다."""
    if tmpl:
        return [(k, 1 + off * width, n) for k, off, n in TEMPLATES[tmpl]]
    out = []
    for rw in legs_df.to_dict("records"):
        kind = KO_KIND.get(str(rw.get("종류")))
        n = pd.to_numeric(rw.get("수량(계약)"), errors="coerce")
        K = pd.to_numeric(rw.get("행사가"), errors="coerce")
        if kind and np.isfinite(n) and n:
            out.append((kind, (K / S) if kind != "S" and np.isfinite(K) else 1.0, float(n)))
    return out


def backtest(close: pd.Series, shape: list, hold_days: int, r: float, q: float, vol_win: int, vol_mult: float,
             stop: float | None, take: float | None, cost: float, mult: float) -> pd.DataFrame:
    """매 hold_days 거래일마다 진입, 만기 = 진입 후 hold_days 거래일. 하루씩 BSM으로 재평가하며 손절·익절."""
    close = close.dropna()
    lr = np.log(close).diff()
    hv = (lr.rolling(vol_win).std() * math.sqrt(252)).bfill()
    idx = close.index
    trades = []
    i = vol_win + 1
    while i + hold_days < len(idx):
        S0 = float(close.iloc[i])
        sig0 = max(float(hv.iloc[i]) * vol_mult, 0.05)
        T0 = hold_days / 252
        legs = []
        for kind, km, n in shape:
            if kind == "S":
                legs.append(("S", np.nan, n, S0))
            else:
                K = S0 * km
                legs.append((kind, K, n, float(bsm.price(S0, K, T0, r, q, sig0, kind == "C"))))
        prem = sum(n * px for k, K, n, px in legs if k != "S")
        # 최대 손실(만기, 손절 기준) 추정
        g = np.linspace(S0 * 0.2, S0 * 3, 600)
        pe = np.zeros_like(g)
        for k, K, n, px in legs:
            pe += n * ((g - px) if k == "S" else ((np.maximum(g - K, 0) if k == "C" else np.maximum(K - g, 0)) - px))
        max_l = float(-pe.min()) if pe.min() < 0 else 0.0
        risk = max_l if max_l > 0 and np.isfinite(max_l) else abs(prem)
        entry_cost = cost * sum(abs(n) * px for k, K, n, px in legs if k != "S")
        exit_i, reason, pnl = i + hold_days, "만기", 0.0
        for j in range(i + 1, i + hold_days + 1):
            S = float(close.iloc[j])
            Tl = max((i + hold_days - j) / 252, 0.0)
            sig = max(float(hv.iloc[j]) * vol_mult, 0.05)
            v = 0.0
            for k, K, n, px in legs:
                if k == "S":
                    v += n * (S - px)
                else:
                    val = (max(S - K, 0) if k == "C" else max(K - S, 0)) if Tl <= 0 else float(bsm.price(S, K, Tl, r, q, sig, k == "C"))
                    v += n * (val - px)
            pnl = v - entry_cost
            if j < i + hold_days and risk > 0:
                if stop is not None and pnl <= -stop * risk:
                    exit_i, reason = j, "손절"
                    break
                if take is not None and pnl >= take * risk:
                    exit_i, reason = j, "익절"
                    break
        trades.append({"진입일": idx[i], "청산일": idx[exit_i], "진입 주가": S0, "변동성": sig0, "순 프리미엄": -prem * mult,
                       "손익": pnl * mult, "사유": reason, "위험(최대손실)": risk * mult})
        i += hold_days
    return pd.DataFrame(trades)


def _summ(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    eq = df["손익"].cumsum()
    dd = (eq - eq.cummax()).min()
    return {"거래 수": len(df), "승률": (df["손익"] > 0).mean(), "평균 손익": df["손익"].mean(), "누적 손익": eq.iloc[-1],
            "최대 낙폭": dd, "최악 거래": df["손익"].min(), "손절 횟수": int((df["사유"] == "손절").sum())}


def render_backtest(tmpl, legs_df, width, ctx, key) -> None:
    import ui

    close = ctx["hist"]
    c1, c2, c3, c4 = st.columns(4)
    yrs = c1.selectbox("기간", [3, 5, 10], index=1, key=f"{key}_bty", format_func=lambda x: f"최근 {x}년")
    hold = c2.selectbox("보유 기간(만기)", [10, 21, 42, 63], index=1, key=f"{key}_bth", format_func=lambda x: f"{x}거래일")
    vwin = c3.selectbox("변동성 창", [20, 60, 120], index=0, key=f"{key}_btv", format_func=lambda x: f"HV {x}일")
    iv_ratio = ctx.get("iv_hv") or 1.0
    vm = c4.number_input("IV ÷ HV 배수", 0.5, 2.5, float(round(iv_ratio, 2)), 0.05, key=f"{key}_btm",
                         help="옵션 가격에 쓸 변동성 = 그때의 HV × 이 배수. 기본값은 지금 시장의 IV ÷ HV")
    c5, c6, c7 = st.columns(3)
    stop = c5.selectbox("손절 (최대손실 대비)", [None, 0.5, 0.75, 1.0], index=2, key=f"{key}_bts",
                        format_func=lambda x: "없음" if x is None else f"{x * 100:.0f}%")
    take = c6.selectbox("익절 (최대손실 대비)", [None, 0.5, 1.0], index=0, key=f"{key}_btt",
                        format_func=lambda x: "없음" if x is None else f"{x * 100:.0f}%")
    cost = c7.number_input("거래 비용 (프리미엄 대비 %)", 0.0, 20.0, 3.0, 0.5, key=f"{key}_btc") / 100
    cl = close[close.index >= close.index[-1] - pd.Timedelta(days=int(365.25 * yrs) + 200)]
    shape = _offsets(tmpl, legs_df, ctx["S"], width)
    if not shape:
        return
    with st.spinner("과거 구간마다 옵션 가격을 다시 만들어 돌리는 중..."):
        res = {}
        for lab, sl in (("손절 없음", None), ("선택한 룰", stop)):
            res[lab] = backtest(cl, shape, hold, ctx["r"], ctx["q"], vwin, vm, sl, take, cost, ctx["mult"])
            if stop is None:
                res["선택한 룰"] = res["손절 없음"]
                break
    base = res["선택한 룰"]
    if base.empty:
        st.info("기간이 짧아 거래가 없습니다.")
        return
    cur = ctx["cur"]
    s1, s0 = _summ(base), _summ(res["손절 없음"])
    ui.cards([("거래 수", f"{s1['거래 수']}"), ("승률", f"{s1['승률'] * 100:.0f}%", f"손절 없음 {s0['승률'] * 100:.0f}%"),
              ("누적 손익", _m(s1["누적 손익"], cur), f"손절 없음 {_m(s0['누적 손익'], cur)}"),
              ("최대 낙폭", _m(s1["최대 낙폭"], cur), f"손절 없음 {_m(s0['최대 낙폭'], cur)}"),
              ("최악 거래", _m(s1["최악 거래"], cur), f"손절 {s1['손절 횟수']}회")], cols=5)
    fig = go.Figure()
    for lab, df in res.items():
        if lab == "손절 없음" or stop is not None:
            fig.add_trace(go.Scatter(x=df["청산일"], y=df["손익"].cumsum(), name=lab, mode="lines",
                                     line=dict(color="#D4AF37" if lab == "선택한 룰" else "#888", width=2 if lab == "선택한 룰" else 1.3)))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=f"누적 손익({cur}, 1세트)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("거래 내역"):
        show = base.copy()
        show["진입일"] = show["진입일"].dt.strftime("%Y-%m-%d")
        show["청산일"] = show["청산일"].dt.strftime("%Y-%m-%d")
        show["변동성"] = (show["변동성"] * 100).round(1)
        st.dataframe(show.round(2), hide_index=True, use_container_width=True, height=300)
    st.caption("과거 옵션 시세가 무료로 없어 그때의 주가와 역사적 변동성(× 배수)으로 BSM 가격을 다시 만든 결과입니다. "
               "스큐(행사가별 IV 차이), 호가 차이, 거래 불가 구간은 반영되지 않습니다. 행사가는 진입일 주가 대비 같은 비율로 잡았습니다.")


__all__ = [n for n in dir() if not n.startswith("__")]
