"""전체 위험 — 주식(미국·국내) · 미국 옵션 · 국내 옵션 · 채권(원화·달러)을 원화로 합쳐
그릭스·DV01·환율 노출, 몬테카를로/과거 시나리오 VaR·ES(자산군별 기여), 스트레스 시나리오를 계산한다.
위험 요인: 기초자산별 주가, 미 국채 10년 금리, 국고채 10년 금리, 원/달러 환율 (최근 일별 변화).
채권은 해당 통화 10년 금리 변화만큼 곡선 전체가 평행 이동한다고 보고 듀레이션·볼록성으로 재평가한다.
요인마다 GARCH(1,1)로 지금 변동성을 예측하고 하루씩 이어 붙여 기간 손익을 만든다 (몬테카를로: 상관 t분포, 과거 시나리오: 필터드 히스토리컬).
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bsm
import ui

CLASSES = ["미국 주식·옵션", "국내 주식·옵션", "달러 채권", "원화 채권"]

STRESS = [  # 이름, 주가 %, 변동성 %p, 미 금리 bp, 국내 금리 bp, 원/달러 %
    ("2008 금융위기형", -35, 25, -150, -150, 25),
    ("2020 코로나형", -30, 35, -100, -60, 6),
    ("2022 금리 급등형", -20, 8, 200, 200, 12),
    ("원화 강세 (원/달러 −10%)", 0, 0, 0, 0, -10),
    ("금리만 +100bp", 0, 0, 100, 100, 0),
    ("주가 +10% · 변동성 −3%p", 10, -3, 0, 0, 0),
]


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_fx_hist() -> pd.Series | None:
    import FinanceDataReader as fdr

    try:
        s = fdr.DataReader("USD/KRW", (datetime.now() - timedelta(days=800)).strftime("%Y-%m-%d"))["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        if len(s) > 100:
            return s[s > 0]
    except Exception:  # noqa: BLE001
        pass
    try:  # FRED 원/달러 (DEXKOUS)
        from bonds import _fred_one

        s = _fred_one("DEXKOUS")
        return s[s.index >= pd.Timestamp.now() - pd.Timedelta(days=800)]
    except Exception:  # noqa: BLE001
        return None


def _kr_stocks() -> dict:
    from options_data import is_us_ticker

    out = {}
    s = st.session_state.get("hedge_stocks_cur")
    if s is None:
        return out
    for r in s.to_dict("records"):
        code = str(r.get("종목코드") or "").strip()
        sh = pd.to_numeric(r.get("보유 주식 수"), errors="coerce")
        if not code or code.lower() == "nan" or not np.isfinite(sh) or is_us_ticker(code.upper()):
            continue
        out[code] = out.get(code, 0.0) + float(sh)
    return out


def gather():
    """반환: dict(us, kr, bonds, fx, warns)."""
    from bondbook import _book, value_book
    from capm_opt import collect_positions, price_positions
    from curve import curves as get_curves
    from options_data import load_usdkrw

    import kropt

    warns = []
    fxv = load_usdkrw()
    fx = fxv[0] if fxv else 1400.0
    if not fxv:
        warns.append("원/달러 환율을 받지 못해 1,400원으로 계산했습니다.")
    under, _ = collect_positions(False)
    us, w = price_positions(under) if under else ({}, [])
    warns += w
    # 국내: 옵션 장부 + 주식
    kb = kropt.kr_book()
    kr, w2 = ({}, [])
    if kb is not None and len(kb.dropna(how="all")):
        kr, w2, _ = kropt.price_kr_book(kb)
        warns += w2
    for code, sh in _kr_stocks().items():
        if code not in kr:
            try:
                cl = kropt.kr_close(code)
            except Exception as e:  # noqa: BLE001
                warns.append(f"{code}: 가격을 불러오지 못해 제외 ({e})")
                continue
            kr[code] = {"S": float(cl.iloc[-1]), "close": cl, "name": kropt.name_of(code), "mult": kropt.mult_of(code), "legs": [],
                        "r": 0.03, "q": 0.0}
        kr[code]["shares"] = kr[code].get("shares", 0.0) + sh
    for u in kr.values():
        u.setdefault("shares", 0.0)
    # 채권
    bonds = {}
    b = _book()
    if b is not None and len(b.dropna(how="all")):
        try:
            from bonds import load_treasury

            ts = load_treasury()
        except Exception:  # noqa: BLE001
            ts = None
        bonds = value_book(b, get_curves(ts))
    return {"us": us, "kr": kr, "bonds": bonds, "fx": fx, "warns": warns}


def factor_levels(G) -> pd.DataFrame:
    """요인 수준 (주가, 금리 소수, 환율)."""
    cols = {}
    for c, u in G["us"].items():
        cols[f"EQ:{c}"] = u["P"]["close"]
    for c, u in G["kr"].items():
        cols[f"EQ:{c}"] = u["close"]
    if "USD" in G["bonds"]:
        try:
            from bonds import load_treasury

            cols["RATE:USD"] = load_treasury()["DGS10"].dropna()
        except Exception:  # noqa: BLE001
            pass
    if "KRW" in G["bonds"]:
        try:
            from curve import load_kr_curve

            k = load_kr_curve()
            if k is not None and "10Y" in k:
                cols["RATE:KRW"] = k["10Y"].dropna()
        except Exception:  # noqa: BLE001
            pass
    fxh = load_fx_hist()
    if fxh is not None:
        cols["FX"] = fxh
    if not cols:
        return pd.DataFrame()
    def _naive(s):
        idx = pd.to_datetime(s.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return pd.Series(s.values, index=idx.normalize()).groupby(level=0).last()

    df = pd.concat({k: _naive(v) for k, v in cols.items()}, axis=1).sort_index()
    df = df[df.index >= df.index.max() - pd.Timedelta(days=760)].ffill().dropna()
    return df


def changes(lv: pd.DataFrame, h: int) -> pd.DataFrame:
    out = {}
    for c in lv.columns:
        out[c] = lv[c].diff(h) if c.startswith("RATE") else np.log(lv[c]).diff(h)
    return pd.DataFrame(out).dropna()


def _fname(c: str) -> str:
    if c == "FX":
        return "원/달러 환율"
    if c == "RATE:USD":
        return "미 국채 10년 금리"
    if c == "RATE:KRW":
        return "국고채 10년 금리"
    return c.split(":", 1)[1] + " 주가"


def garch_sims(d1: pd.DataFrame, h: int, method: str = "mc", n: int = 20000, seed: int = 17):
    """요인별 GARCH(1,1)로 h거래일 누적 변화를 시뮬레이션. 반환 ({요인: 배열}, 요인 표).
    mc: 표준화 잔차의 상관 + 공통 자유도 t분포 / fhs: 과거 표준화 잔차 벡터를 날짜 단위로 다시 뽑음."""
    import garch

    d1 = d1.dropna()
    cols = list(d1.columns)
    fits, Z, rows = {}, {}, []
    for c in cols:
        f = None
        try:
            f = garch.fit(d1[c], years=None)
        except Exception:  # noqa: BLE001
            f = None
        fits[c] = f
        x = d1[c]
        Z[c] = f["z"] if f else (x - x.mean()) / (x.std() or 1.0)
        rate = c.startswith("RATE")
        unit = (lambda v: f"{v * 1e4:,.0f}bp") if rate else (lambda v: f"{v * 100:.1f}%")
        sd_now = np.sqrt((f["next"] if f else x.var()) * 252)
        sd_lr = np.sqrt((f["VL"] if f else x.var()) * 252)
        rows.append({"요인": _fname(c), "지금 변동성": unit(sd_now), "장기 평균": unit(sd_lr),
                     "지금 ÷ 장기": f"{sd_now / sd_lr:.2f}배" if sd_lr > 0 else "-",
                     "지속성 α+β": f"{f['persist']:.3f}" if f else "-",
                     "반감기(일)": f"{f['half_life']:.0f}" if f and np.isfinite(f["half_life"]) else "-"})
    zdf = pd.DataFrame(Z).dropna()
    zv = zdf[cols].values
    zv = (zv - zv.mean(0)) / np.where(zv.std(0) > 0, zv.std(0), 1.0)
    rng = np.random.default_rng(seed)
    k = len(cols)
    if method == "mc":
        R = np.corrcoef(zv.T) if k > 1 else np.ones((1, 1))
        R = np.atleast_2d(R)
        try:
            Lc = np.linalg.cholesky(R + np.eye(k) * 1e-10)
        except np.linalg.LinAlgError:
            Lc = np.eye(k)
        nus = [fits[c]["nu"] for c in cols if fits[c]]
        nu = float(np.clip(np.median(nus), 4.0, 30.0)) if nus else 8.0

        def draw():
            g = rng.standard_normal((n, k)) @ Lc.T
            w = rng.chisquare(nu, size=(n, 1))
            return g * np.sqrt((nu - 2) / w)
    else:
        def draw():
            return zv[rng.integers(0, len(zv), size=n)]

    s2 = np.array([fits[c]["next"] if fits[c] else d1[c].var() for c in cols], float)
    mu = np.array([fits[c]["mu"] if fits[c] else d1[c].mean() for c in cols], float)
    om = np.array([fits[c]["omega"] if fits[c] else 0.0 for c in cols], float)
    al = np.array([fits[c]["alpha"] if fits[c] else 0.0 for c in cols], float)
    be = np.array([fits[c]["beta"] if fits[c] else 1.0 for c in cols], float)
    S2 = np.tile(s2, (n, 1))
    tot = np.zeros((n, k))
    for _ in range(max(int(h), 1)):
        e = np.sqrt(S2) * draw()
        tot += mu + e
        S2 = om + al * e * e + be * S2
    return {c: tot[:, i] for i, c in enumerate(cols)}, pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 재평가 (원화 손익)
# ----------------------------------------------------------------------
def _opt_val(S1, lg, T_left, r, q, dvol=0.0):
    if T_left <= 1e-6:
        return np.maximum(S1 - lg["K"], 0.0) if lg["is_call"] else np.maximum(lg["K"] - S1, 0.0)
    return bsm.price(S1, lg["K"], T_left, r, q, max(lg["sig"] + dvol, 0.01), lg["is_call"])


def reval(G, X: dict, h_years: float, dvol: float = 0.0) -> dict:
    """X: {요인: 변화 배열}. 반환 {자산군: 원화 손익 배열}."""
    from capm_opt import MULT

    n = len(next(iter(X.values()))) if X else 1
    fx0 = G["fx"]
    fx1 = fx0 * np.exp(X.get("FX", np.zeros(n)))
    out = {k: np.zeros(n) for k in CLASSES}
    for c, u in G["us"].items():
        P = u["P"]
        S1 = P["S"] * np.exp(X.get(f"EQ:{c}", np.zeros(n)))
        pnl = u["shares"] * (S1 - P["S"])
        v0 = u["shares"] * P["S"]
        for lg in u["legs"]:
            pnl = pnl + lg["n"] * MULT * (_opt_val(S1, lg, lg["T"] - h_years, P["r"], P["q"], dvol) - lg["mkt"])
            v0 += lg["n"] * MULT * lg["mkt"]
        out["미국 주식·옵션"] += pnl * fx1 + v0 * (fx1 - fx0)
    for c, u in G["kr"].items():
        S1 = u["S"] * np.exp(X.get(f"EQ:{c}", np.zeros(n)))
        pnl = u["shares"] * (S1 - u["S"])
        for lg in u["legs"]:
            pnl = pnl + lg["n"] * u["mult"] * (_opt_val(S1, lg, lg["T"] - h_years, u["r"], 0.0, dvol) - lg["mkt"])
        out["국내 주식·옵션"] += pnl
    for cur, V in G["bonds"].items():
        dy = X.get(f"RATE:{cur}", np.zeros(n))
        dv = V["tot"] * (-V["dur"] * dy + 0.5 * V["conv"] * dy * dy)
        if cur == "USD":
            out["달러 채권"] += dv * fx1 + V["tot"] * (fx1 - fx0)
        else:
            out["원화 채권"] += dv
    return out


def totals(G) -> dict:
    from capm_opt import MULT

    fx = G["fx"]
    v = {k: 0.0 for k in CLASSES}
    gk = {"delta1": 0.0, "gamma5": 0.0, "vega": 0.0, "usd": 0.0, "dv01_usd": 0.0, "dv01_krw": 0.0}
    for u in G["us"].values():
        P = u["P"]
        val = u["shares"] * P["S"] + sum(lg["n"] * MULT * lg["mkt"] for lg in u["legs"])
        v["미국 주식·옵션"] += val * fx
        gk["usd"] += val
        d = u["shares"] + sum(lg["n"] * MULT * lg["delta"] for lg in u["legs"])
        g = sum(lg["n"] * MULT * lg["gamma"] for lg in u["legs"])
        gk["delta1"] += d * P["S"] * 0.01 * fx
        gk["gamma5"] += 0.5 * g * (0.05 * P["S"]) ** 2 * fx
        gk["vega"] += sum(lg["n"] * MULT * lg["vega"] for lg in u["legs"]) * fx
    for u in G["kr"].values():
        m = u["mult"]
        v["국내 주식·옵션"] += u["shares"] * u["S"] + sum(lg["n"] * m * lg["mkt"] for lg in u["legs"])
        d = u["shares"] + sum(lg["n"] * m * lg["delta"] for lg in u["legs"])
        g = sum(lg["n"] * m * lg["gamma"] for lg in u["legs"])
        gk["delta1"] += d * u["S"] * 0.01
        gk["gamma5"] += 0.5 * g * (0.05 * u["S"]) ** 2
        gk["vega"] += sum(lg["n"] * m * lg["vega"] for lg in u["legs"])
    for cur, V in G["bonds"].items():
        if cur == "USD":
            v["달러 채권"] += V["tot"] * fx
            gk["usd"] += V["tot"]
            gk["dv01_usd"] += V["dv01"] * fx
        else:
            v["원화 채권"] += V["tot"]
            gk["dv01_krw"] += V["dv01"]
    gk["fx1"] = gk["usd"] * fx * 0.01
    return {"value": v, "greeks": gk}


def _won(x):
    if not np.isfinite(x):
        return "-"
    return "0원" if abs(x) < 0.5 else f"{x:,.0f}원"


def render(key: str = "ra") -> None:
    with st.spinner("보유 자산을 모아 위험 요인을 불러오는 중..."):
        G = gather()
    n_pos = len(G["us"]) + len(G["kr"]) + len(G["bonds"])
    if not n_pos:
        st.info("옵션 페이지 '내 포지션'에 주식·옵션을, 채권 페이지 '내 채권'에 채권을 넣으면 전부 합쳐 계산합니다.")
        return
    for w in G["warns"][:6]:
        st.warning(w)
    T = totals(G)
    tv = sum(T["value"].values())
    gk = T["greeks"]
    has_eq = bool(G["us"] or G["kr"])
    has_opt = any(u["legs"] for u in list(G["us"].values()) + list(G["kr"].values()))
    cards = [("전체 평가액", _won(tv), f"원/달러 {G['fx']:,.1f}")]
    if has_eq:
        cards.append(("주가 1% 상승 시", _won(gk["delta1"]), "모든 주식·기초자산이 같이 1%"))
    if has_opt:
        cards += [("주가 ±5% 시 감마 효과", _won(gk["gamma5"]), "델타로 설명 안 되는 부분"), ("변동성 1%p 상승 시", _won(gk["vega"]))]
    if "USD" in G["bonds"]:
        cards.append(("미 금리 1bp 상승 시", _won(-gk["dv01_usd"]), "달러 채권"))
    if "KRW" in G["bonds"]:
        cards.append(("국내 금리 1bp 상승 시", _won(-gk["dv01_krw"]), "원화 채권"))
    if gk["usd"]:
        cards.append(("원/달러 1% 상승 시", _won(gk["fx1"]), f"달러 자산 ${gk['usd']:,.0f}"))
    cards.append(("자산 구성", " · ".join(f"{k} {v / tv * 100:.0f}%" for k, v in T["value"].items() if abs(v) > 0) if tv else "-"))
    ui.cards(cards[:4], cols=4)
    if len(cards) > 4:
        ui.cards(cards[4:], cols=4)

    c1, c2, c3 = st.columns(3)
    h = c1.selectbox("기간", [1, 5, 10, 20], index=0, key=f"{key}_h", format_func=lambda x: f"{x}거래일")
    meth = c2.radio("방법", ["몬테카를로", "과거 시나리오"], horizontal=True, key=f"{key}_m",
                    help="둘 다 요인마다 GARCH(1,1)로 지금 변동성을 예측해 하루씩 이어 붙입니다. 몬테카를로: 상관을 반영한 t분포 충격 2만 개 · "
                         "과거 시나리오: 최근 2년 실제로 있었던 하루치 충격(표준화)을 다시 뽑아 씀 (필터드 히스토리컬)")
    conf = c3.selectbox("신뢰수준", [0.95, 0.99], index=1, key=f"{key}_c", format_func=lambda x: f"{x * 100:.0f}%")
    lv = factor_levels(G)
    if lv.empty or len(lv) < 60:
        st.warning("위험 요인 데이터를 충분히 받지 못했습니다.")
        return
    for cur in G["bonds"]:
        if f"RATE:{cur}" not in lv:
            st.warning(f"{'미 국채' if cur == 'USD' else '국고채'} 10년 금리 이력을 받지 못해 VaR에서 {'달러' if cur == 'USD' else '원화'} 채권의 금리 위험이 빠졌습니다 (스트레스에는 반영).")
    with st.spinner("위험 요인마다 GARCH(1,1) 변동성을 추정하는 중..."):
        X, ftab = garch_sims(changes(lv, 1), h, "mc" if meth == "몬테카를로" else "fhs")
    P = reval(G, X, h / 252)
    tot = sum(P.values())
    q = np.quantile(tot, 1 - conf)
    tail = tot <= q
    var, es = -q, -tot[tail].mean()
    ui.cards([(f"VaR {conf * 100:.0f}% ({h}일)", _won(var), f"{(1 - conf) * 100:.0f}% 확률로 이보다 더 잃음"),
              ("기대 손실 ES", _won(es), "그 나쁜 경우들의 평균 손실"),
              ("VaR ÷ 평가액", f"{var / tv * 100:.2f}%" if tv else "-"),
              ("시나리오 수", f"{len(tot):,}")], cols=4)
    rows = []
    for k in CLASSES:
        x = P[k]
        if not np.any(x):
            continue
        sv = -np.quantile(x, 1 - conf)
        rows.append({"자산군": k, "평가액": T["value"][k], "단독 VaR": sv, "ES 기여": -x[tail].mean(),
                     "ES 기여 비중(%)": (-x[tail].mean()) / es * 100 if es else np.nan})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df.style.format({"평가액": "{:,.0f}", "단독 VaR": "{:,.0f}", "ES 기여": "{:,.0f}", "ES 기여 비중(%)": "{:.0f}"}),
                     hide_index=True, use_container_width=True)
        div = df["단독 VaR"].sum() - var
        if len(df) > 1:
            st.caption(f"단독 VaR 합계 {df['단독 VaR'].sum():,.0f}원 − 전체 VaR {var:,.0f}원 = 분산 효과 {div:,.0f}원. "
                       "ES 기여는 전체가 크게 잃은 경우들에서 각 자산군이 평균적으로 잃은 금액이라 다 더하면 ES와 같습니다.")
    exp_f, op_f = ui.more("위험 요인별 변동성 (GARCH(1,1))", key=f"{key}_more_garch")
    if op_f:
        with exp_f:
            st.dataframe(ftab, hide_index=True, use_container_width=True)
            st.caption("지금 변동성 = 내일 하루 변동성 예측(연환산), 장기 평균 = 시간이 지나면 돌아갈 수준. 지속성(α+β)이 1에 가까울수록 "
                       "충격이 오래 남고, 반감기는 지금과 장기 평균의 차이가 절반으로 줄어드는 데 걸리는 거래일입니다. "
                       "추정이 안 된 요인은 최근 2년 표본 변동성을 그대로 씁니다.")
    fig = go.Figure(go.Histogram(x=tot, nbinsx=80, marker_color="#4C9BE8", opacity=0.8))
    fig.add_vline(x=q, line_color="#E5534B", line_dash="dash", annotation_text=f"VaR {conf * 100:.0f}%")
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), xaxis_title=f"{h}거래일 손익(원)",
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 스트레스 시나리오")
    srows = []
    for nm, eq, vol, us_bp, kr_bp, fxp in STRESS:
        Xs = {f"EQ:{c}": np.array([np.log1p(eq / 100)]) for c in list(G["us"]) + list(G["kr"])}
        Xs["RATE:USD"] = np.array([us_bp / 1e4])
        Xs["RATE:KRW"] = np.array([kr_bp / 1e4])
        Xs["FX"] = np.array([np.log1p(fxp / 100)])
        Ps = reval(G, Xs, 0.0, dvol=vol / 100)
        r = {"시나리오": nm, "조건": f"주가 {eq:+d}% · 변동성 {vol:+d}%p · 미 금리 {us_bp:+d}bp · 국내 금리 {kr_bp:+d}bp · 원/달러 {fxp:+d}%"}
        for k in CLASSES:
            if T["value"][k] or np.any(Ps[k]):
                r[k] = float(Ps[k][0])
        r["합계"] = float(sum(v[0] for v in Ps.values()))
        srows.append(r)
    sd = pd.DataFrame(srows)
    num = [c for c in sd.columns if c not in ("시나리오", "조건")]
    st.dataframe(sd.style.format({c: "{:,.0f}" for c in num}), hide_index=True, use_container_width=True,
                 column_config={"조건": st.column_config.TextColumn(width="medium")})
    st.caption("스트레스는 즉시(오늘) 충격이 왔을 때의 원화 손익입니다. 주가는 모든 종목이 같은 비율로 움직인다고 가정했고, "
               "옵션은 변동성 충격까지 넣어 다시 계산했습니다. 채권은 각 통화 10년 금리 변화만큼 곡선 전체가 평행 이동한다고 보고 "
               "듀레이션·볼록성으로 계산했습니다. 과거 위기 숫자는 대표적인 크기를 단순화한 값입니다.")


__all__ = [n for n in dir() if not n.startswith("__")]
