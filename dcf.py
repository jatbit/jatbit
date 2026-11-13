"""DCF(현금흐름할인) 가치평가 — 기업 리서치 탭의 하위 탭
- 기준 FCF: SEC 10-K(영업현금흐름 − 설비투자) 또는 Yahoo 현금흐름표
- 2단계 성장(고성장 → 영구성장률로 선형 수렴) + 영구가치(고든 성장 모형 또는 출구 배수)
- WACC: CAPM 자기자본비용(FRED 10년 국채 + 베타 × 시장위험프리미엄) + 세후 타인자본비용
- 결과: 주당 내재가치·상승 여력, 연도별 추정표, 현재가치 구성, WACC × 영구성장률 민감도, 역DCF(현재가에 내재된 성장률)
"""
import io
import math
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from options import mgrid

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None


# ----------------------------------------------------------------------
# 계산 (화면과 분리 — 테스트 가능)
# ----------------------------------------------------------------------
def growth_path(g1: float, n1: int, n_fade: int, gt: float) -> list:
    """1~n1년차는 g1, 이후 n_fade년 동안 g1 → gt로 선형 수렴."""
    path = [g1] * n1
    for i in range(1, n_fade + 1):
        path.append(g1 + (gt - g1) * i / (n_fade + 1))
    return path


def dcf_value(fcf0: float, g1: float, n1: int, n_fade: int, gt: float, wacc: float, mid_year: bool = True,
              exit_mult: float | None = None) -> dict:
    """기업가치(EV)와 구성 요소. exit_mult가 있으면 마지막 해 FCF × 배수로 영구가치를 계산."""
    gs = growth_path(g1, n1, n_fade, gt)
    fcf, f = [], fcf0
    for g in gs:
        f = f * (1 + g)
        fcf.append(f)
    n = len(fcf)
    t = np.arange(1, n + 1) - (0.5 if mid_year else 0.0)
    disc = 1 / (1 + wacc) ** t
    pv = np.array(fcf) * disc
    if exit_mult is not None:
        tv = fcf[-1] * exit_mult
    else:
        if wacc <= gt:
            return {"ok": False, "msg": "WACC가 영구성장률보다 커야 합니다"}
        tv = fcf[-1] * (1 + gt) / (wacc - gt)
    pv_tv = tv / (1 + wacc) ** n  # 영구가치는 마지막 해 말 시점
    ev = float(pv.sum() + pv_tv)
    return {"ok": True, "years": list(range(1, n + 1)), "growth": gs, "fcf": fcf, "disc": disc.tolist(), "pv": pv.tolist(),
            "tv": tv, "pv_tv": pv_tv, "ev": ev, "tv_share": pv_tv / ev if ev else float("nan"),
            "implied_mult": tv / fcf[-1] if fcf[-1] else float("nan")}


def per_share(ev: float, debt: float, cash: float, shares: float) -> float:
    return (ev - debt + cash) / shares if shares else float("nan")


def wacc_calc(rf: float, beta: float, mrp: float, rd: float, tax: float, E: float, D: float) -> dict:
    re = rf + beta * mrp
    w_e = E / (E + D) if (E + D) > 0 else 1.0
    return {"re": re, "rd_after": rd * (1 - tax), "w_e": w_e, "w_d": 1 - w_e, "wacc": w_e * re + (1 - w_e) * rd * (1 - tax)}


def implied_growth(target_ps: float, fcf0, n1, n_fade, gt, wacc, mid, debt, cash, shares, exit_mult=None):
    """역DCF: 주당가치가 현재가와 같아지는 고성장 구간 성장률 g1 (이분법)."""
    def f(g):
        r = dcf_value(fcf0, g, n1, n_fade, gt, wacc, mid, exit_mult)
        return per_share(r["ev"], debt, cash, shares) - target_ps if r["ok"] else float("nan")
    lo, hi = -0.5, 1.5
    flo, fhi = f(lo), f(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
        return float("nan")
    for _ in range(80):
        mid_g = (lo + hi) / 2
        fm = f(mid_g)
        if flo * fm <= 0:
            hi, fhi = mid_g, fm
        else:
            lo, flo = mid_g, fm
    return (lo + hi) / 2


def cagr(s: pd.Series, years: int):
    s = s.dropna()
    s = s.iloc[-(years + 1):]
    if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return float("nan")
    return (s.iloc[-1] / s.iloc[0]) ** (1 / (len(s) - 1)) - 1


# ----------------------------------------------------------------------
# 데이터
# ----------------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_us10y():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                d = pd.read_csv(io.StringIO(resp.read().decode("utf-8")), na_values=["."]).dropna()
            return float(d.iloc[-1, 1]) / 100, str(d.iloc[-1, 0])
        except Exception:  # noqa: BLE001
            pass
    return None


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_fx(frm: str, to: str):
    if yf is None or frm == to:
        return 1.0
    try:
        s = yf.Ticker(f"{frm}{to}=X").history(period="10d")["Close"].dropna()
        return float(s.iloc[-1]) if len(s) else None
    except Exception:  # noqa: BLE001
        return None


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:  # noqa: BLE001
        return None


def _yahoo_hist(R: dict) -> pd.DataFrame:
    cf, inc = R.get("cashflow"), R.get("income")
    out = {}
    for nm, df, keys in (("잉여현금흐름", cf, ["Free Cash Flow"]), ("영업현금흐름", cf, ["Operating Cash Flow"]),
                         ("매출", inc, ["Total Revenue", "Operating Revenue"]), ("이자비용", inc, ["Interest Expense"])):
        if isinstance(df, pd.DataFrame) and not df.empty:
            for k in keys:
                if k in df.index:
                    out[nm] = pd.to_numeric(df.loc[k], errors="coerce")
                    break
    if not out:
        return pd.DataFrame()
    h = pd.DataFrame(out).T
    return h[sorted(h.columns)]


def _big(x):
    import money

    return money.big(x)


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
def render_dcf(ticker: str, R: dict, sec: pd.DataFrame | None, src_name: str = "SEC 10-K", domestic: bool = False) -> None:
    import money

    info = R.get("info") or {}
    px = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    shares = _num(info.get("sharesOutstanding")) or _num(info.get("impliedSharesOutstanding"))
    if not px or not shares:
        st.warning("현재가나 발행주식수가 없어 DCF를 계산할 수 없습니다 (ETF·펀드는 지원하지 않습니다).")
        return
    given = sec is not None and "잉여현금흐름" in sec.index
    src = src_name if given else "Yahoo"
    hist = sec if given else _yahoo_hist(R)
    if hist is None or hist.empty or "잉여현금흐름" not in hist.index or hist.loc["잉여현금흐름"].dropna().empty:
        st.warning("잉여현금흐름 데이터를 찾지 못했습니다. 아래 '기준 FCF'를 직접 입력하면 계산할 수 있습니다.")
        hist = pd.DataFrame()
    # 재무제표 통화와 주가 통화가 다르면(예: 대만·일본 기업 ADR) 환율로 맞춤
    fin_cur, px_cur = info.get("financialCurrency") or "USD", info.get("currency") or "USD"
    fx = 1.0
    if fin_cur != px_cur and not given:
        fx = load_fx(fin_cur, px_cur) or float("nan")
        st.caption(f"재무제표 통화({fin_cur})를 주가 통화({px_cur})로 환산했습니다: 1 {fin_cur} = {fx:.4f} {px_cur}")
    fcf_h = hist.loc["잉여현금흐름"].dropna() * fx if not hist.empty else pd.Series(dtype=float)
    rev_h = hist.loc["매출"].dropna() * fx if not hist.empty and "매출" in hist.index else pd.Series(dtype=float)

    # ---- 과거 FCF
    st.markdown("##### 과거 잉여현금흐름")
    if len(fcf_h):
        f = go.Figure()
        xs = [pd.Timestamp(c).strftime("%Y-%m") for c in fcf_h.index]
        f.add_trace(go.Bar(x=xs, y=fcf_h.values, name="잉여현금흐름", marker_color=np.where(fcf_h.values >= 0, "#4C9BE8", "#E5534B"),
                           hovertemplate="%{x} FCF " + money.hv() + "<extra></extra>"))
        if len(rev_h):
            m = (fcf_h / rev_h.reindex(fcf_h.index) * 100)
            f.add_trace(go.Scatter(x=xs, y=m.values, name="FCF 마진(%)", yaxis="y2", mode="lines+markers", line=dict(color="#F2A93B"),
                                   hovertemplate="%{x} FCF 마진 %{y:.1f}%<extra></extra>"))
        f.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                        yaxis=dict(title="FCF"), yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(f, use_container_width=True)
    g_rev5, g_fcf5 = cagr(rev_h, 5), cagr(fcf_h, 5)
    st.caption(f"출처: {src}")

    # ---- 입력
    st.markdown("##### 가정 (모두 직접 고칠 수 있습니다)")
    fcf_last = float(fcf_h.iloc[-1]) if len(fcf_h) else None
    fcf_avg3 = float(fcf_h.iloc[-3:].mean()) if len(fcf_h) else None
    fcf_ttm = (_num(info.get("freeCashflow")) or 0) * ((load_fx(fin_cur, px_cur) or 1) if fin_cur != px_cur else 1) or None
    opts = {}
    if fcf_last is not None:
        opts[f"최근 회계연도 FCF ({_big(fcf_last)})"] = fcf_last
    if fcf_avg3 is not None and len(fcf_h) >= 3:
        opts[f"최근 3년 평균 FCF ({_big(fcf_avg3)})"] = fcf_avg3
    if fcf_ttm:
        opts[f"Yahoo 최근 12개월 FCF ({_big(fcf_ttm)})"] = fcf_ttm
    opts["직접 입력"] = None
    k = f"dcf_{ticker}"
    c1, c2 = st.columns([2, 1])
    base_pick = c1.selectbox("기준 FCF", list(opts), key=k + "_base")
    if opts[base_pick] is None:
        sc, un = money.unit()
        fcf0 = c2.number_input(f"기준 FCF ({un})", value=round((fcf_last or 1000 * sc) / sc, 1), step=100.0, key=k + "_basem") * sc
    else:
        fcf0 = opts[base_pick]
        c2.metric("기준 FCF", _big(fcf0))
    if fcf0 <= 0:
        st.warning("기준 FCF가 0 이하라 DCF 결과가 의미 없습니다. 3년 평균이나 직접 입력을 쓰세요.")

    g_def = g_rev5 if np.isfinite(g_rev5) else (_num(info.get("revenueGrowth")) or 0.05)
    g_def = float(np.clip(g_def, -0.05, 0.30))
    if domestic:
        from kr import kr_rate

        kk = kr_rate("10Y")
        rr = (kk[0], kk[1]) if kk else None
        rf_name = kk[2] if kk else "국고채 10년"
    else:
        rr = load_us10y()
        rf_name = "미 국채 10년"
    rf_def = rr[0] if rr else (0.03 if domestic else 0.042)
    beta_def = _num(info.get("beta")) or 1.0
    D = _num(info.get("totalDebt")) or 0.0
    cash = _num(info.get("totalCash")) or 0.0
    E = _num(info.get("marketCap")) or px * shares
    int_exp = None
    if not hist.empty and "이자비용" in hist.index and hist.loc["이자비용"].dropna().size:
        int_exp = abs(float(hist.loc["이자비용"].dropna().iloc[-1])) * fx
    rd_def = float(np.clip(int_exp / D, 0.01, 0.15)) if int_exp and D > 0 else 0.05

    a1, a2, a3, a4 = st.columns(4)
    g1 = a1.number_input("고성장 구간 성장률 (%)", value=round(g_def * 100, 1), step=0.5, key=k + "_g1",
                         help="기본값: 매출 5년 연평균 성장률(−5%~30%로 제한)") / 100
    n1 = a2.number_input("고성장 기간 (년)", 1, 15, 5, key=k + "_n1")
    n_fade = a3.number_input("수렴 기간 (년)", 0, 15, 5, key=k + "_nf", help="고성장률에서 영구성장률로 선형으로 줄어드는 기간")
    gt = a4.number_input("영구성장률 (%)", value=2.5, step=0.25, key=k + "_gt", help="장기 명목 GDP 성장률 수준(2~3%)이 보통") / 100
    b1, b2, b3, b4 = st.columns(4)
    rf = b1.number_input("무위험금리 (%)", value=round(rf_def * 100, 2), step=0.05, key=k + "_rf",
                         help=f"{rf_name} {rr[0] * 100:.2f}% ({rr[1]})" if rr else "금리 조회 실패로 기본값 사용") / 100
    beta = b2.number_input("베타", value=round(beta_def, 2), step=0.05, key=k + "_beta", help="Yahoo 베타(5년 월간)")
    mrp = b3.number_input("시장위험프리미엄 (%)", value=5.5 if domestic else 5.0, step=0.25, key=k + "_mrp",
                          help="주식이 무위험금리보다 더 벌 것으로 기대되는 몫. 4~6%가 흔히 쓰임") / 100
    tax = b4.number_input("법인세율 (%)", value=27.5 if domestic else 21.0, step=1.0, key=k + "_tax",
                          help="국내: 최고세율 25% + 지방소득세 2.5%" if domestic else "미국 연방 법인세율") / 100
    c1, c2, c3, c4 = st.columns(4)
    rd = c1.number_input("세전 타인자본비용 (%)", value=round(rd_def * 100, 2), step=0.25, key=k + "_rd",
                         help="기본값: 최근 이자비용 ÷ 총차입금" if int_exp and D > 0 else "이자비용 자료가 없어 5% 기본값") / 100
    term = c2.radio("영구가치 방식", ["고든 성장 모형", "출구 배수(EV/FCF)"], key=k + "_term", horizontal=False)
    exit_mult = c3.number_input("출구 배수 (배)", value=20.0, step=1.0, key=k + "_em",
                                disabled=term.startswith("고든"))
    mid = c4.toggle("연중 현금흐름 가정", value=True, key=k + "_mid", help="현금이 연말이 아니라 1년 동안 고르게 들어온다고 보고 반년씩 덜 할인")
    em = exit_mult if term.startswith("출구") else None

    W = wacc_calc(rf, beta, mrp, rd, tax, E, D)
    wacc = W["wacc"]
    with st.expander("WACC 직접 지정 (선택)"):
        manual = st.checkbox("계산된 WACC 대신 직접 입력한 값 사용", key=k + "_wman")
        if manual:
            wacc = st.number_input("WACC (%)", value=round(W["wacc"] * 100, 2), step=0.25, key=k + "_wv") / 100
    res = dcf_value(fcf0, g1, int(n1), int(n_fade), gt, wacc, mid, em)
    if not res["ok"]:
        st.error(res["msg"])
        return
    ps = per_share(res["ev"], D, cash, shares)
    up = ps / px - 1

    # ---- 결과
    st.markdown("##### 결과")
    mgrid([("주당 내재가치", money.price(ps), f"현재가 {money.price(px)}"),
           ("상승 여력", f"{up * 100:+.1f}%", "내재가치 ÷ 현재가 − 1"),
           ("WACC", f"{wacc * 100:.2f}%", f"자기자본 {W['w_e'] * 100:.0f}% × {W['re'] * 100:.2f}% + 타인자본 {W['w_d'] * 100:.0f}% × 세후 {W['rd_after'] * 100:.2f}%"),
           ("자기자본비용", f"{W['re'] * 100:.2f}%", f"{rf * 100:.2f}% + {beta:.2f} × {mrp * 100:.2f}%"),
           ("기업가치(EV)", _big(res["ev"])),
           ("− 순차입금", _big(D - cash), f"총차입금 {_big(D)} − 현금 {_big(cash)}"),
           ("= 주주가치", _big(res["ev"] - D + cash), f"÷ 발행주식수 {shares / 1e6:,.1f}백만주"),
           ("영구가치 비중", f"{res['tv_share'] * 100:.0f}%", "EV 중 영구가치(추정 기간 이후) 현재가치의 비중. 높을수록 가정에 민감"),
           ("내재 출구 배수", f"{res['implied_mult']:.1f}배", "영구가치 ÷ 마지막 해 FCF"),
           ("현재 시장 EV/FCF", f"{(E + D - cash) / fcf0:.1f}배" if fcf0 > 0 else "-", "(시가총액 + 순차입금) ÷ 기준 FCF")], cols=5)

    f = go.Figure()
    yrs = [f"{y}년차" for y in res["years"]]
    f.add_trace(go.Bar(x=yrs, y=res["fcf"], name="추정 FCF", marker_color="rgba(76,155,232,0.35)",
                       hovertemplate="%{x} FCF " + money.hv() + "<extra></extra>"))
    f.add_trace(go.Bar(x=yrs, y=res["pv"], name="현재가치", marker_color="#4C9BE8",
                       hovertemplate="%{x} 현재가치 " + money.hv() + "<extra></extra>"))
    f.add_trace(go.Bar(x=["영구가치"], y=[res["pv_tv"]], name="영구가치 현재가치", marker_color="#9B6FD6",
                       hovertemplate="영구가치 현재가치 " + money.hv() + "<extra></extra>"))
    f.update_layout(barmode="overlay", height=300, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), yaxis=dict(title=money.sym()))
    st.plotly_chart(f, use_container_width=True)
    tbl = pd.DataFrame([[f"{v * 100:.1f}%" for v in res["growth"]], [_big(v) for v in res["fcf"]],
                        [f"{v:.4f}" for v in res["disc"]], [_big(v) for v in res["pv"]]],
                       index=["성장률", "FCF", "할인계수", "현재가치"], columns=yrs)
    st.dataframe(tbl, use_container_width=True)

    # ---- 민감도
    st.markdown("##### 민감도 (주당 내재가치)")
    ws = [wacc + d for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
    if em is None:
        cols = [gt + d for d in (-0.01, -0.005, 0.0, 0.005, 0.01)]
        grid = [[(lambda r: per_share(r["ev"], D, cash, shares) if r["ok"] else np.nan)(dcf_value(fcf0, g1, int(n1), int(n_fade), g, w, mid))
                 for g in cols] for w in ws]
        cl = [f"영구 {g * 100:.1f}%" for g in cols]
    else:
        cols = [em * m for m in (0.7, 0.85, 1.0, 1.15, 1.3)]
        grid = [[per_share(dcf_value(fcf0, g1, int(n1), int(n_fade), gt, w, mid, m)["ev"], D, cash, shares) for m in cols] for w in ws]
        cl = [f"배수 {m:.1f}" for m in cols]
    sens = pd.DataFrame(grid, index=[f"WACC {w * 100:.2f}%" for w in ws], columns=cl)
    up_grid = sens / px - 1
    z = up_grid.values * 100
    hm = go.Figure(go.Heatmap(z=z, x=cl, y=list(sens.index), colorscale="RdYlGn", zmid=0,
                              text=[[f"{money.price(v)}<br>{u:+.0f}%" if np.isfinite(v) else "-" for v, u in zip(r1, r2)] for r1, r2 in zip(sens.values, z)],
                              texttemplate="%{text}", hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>", showscale=False))
    hm.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), yaxis=dict(autorange="reversed"))
    st.plotly_chart(hm, use_container_width=True)

    # ---- 역DCF
    st.markdown("##### 역DCF (현재 주가에 반영된 성장률)")
    ig = implied_growth(px, fcf0, int(n1), int(n_fade), gt, wacc, mid, D, cash, shares, em) if fcf0 > 0 else float("nan")
    mgrid([("시장이 가정한 고성장률", f"{ig * 100:.1f}%" if np.isfinite(ig) else "-",
            f"고성장 {int(n1)}년 + 수렴 {int(n_fade)}년, 나머지 가정은 위와 동일"),
           ("내가 넣은 고성장률", f"{g1 * 100:.1f}%"),
           ("과거 매출 성장률(5년)", f"{g_rev5 * 100:.1f}%" if np.isfinite(g_rev5) else "-"),
           ("과거 FCF 성장률(5년)", f"{g_fcf5 * 100:.1f}%" if np.isfinite(g_fcf5) else "-")], cols=4)
