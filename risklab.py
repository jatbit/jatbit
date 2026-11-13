"""백테스트 탭의 위험 분석
- 스트레스 테스트: 과거 위기 구간을 지금 비중으로 다시 겪으면 얼마나 잃는지 (그때 데이터가 없는 종목은 벤치마크 × 베타로 대신),
  가상의 시장 충격, 백테스트 기간 중 최악의 1·5·20일
- VaR 검증(백테스트): 매일 전날까지의 자료로 VaR를 예측하고 실제 손실과 비교 — 쿠픽(횟수)·크리스토퍼슨(몰림) 검정, 바젤 신호등
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from glossary import table_html
from options import mgrid

SCENARIOS = {  # 이름: (시작, 끝) — 미국·글로벌 주가 고점→저점 기준
    "글로벌 금융위기": ("2007-10-09", "2009-03-09"),
    "유럽 재정위기·미국 신용등급 강등": ("2011-07-07", "2011-10-03"),
    "중국 위안화 쇼크": ("2015-08-10", "2016-02-11"),
    "2018년 4분기 급락": ("2018-09-20", "2018-12-24"),
    "코로나19 폭락": ("2020-02-19", "2020-03-23"),
    "2022년 금리 인상 약세장": ("2022-01-03", "2022-10-12"),
    "2024년 8월 엔 캐리 청산": ("2024-07-10", "2024-08-05"),
}
Z = {0.95: 1.6448536269514722, 0.99: 2.3263478740408408}


# ----------------------------------------------------------------------
# 계산
# ----------------------------------------------------------------------
def chi2_sf(x: float, df: int) -> float:
    if not np.isfinite(x) or x < 0:
        return float("nan")
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    if df == 2:
        return math.exp(-x / 2)
    raise ValueError("df")


def _xlogy(x, y):
    return 0.0 if x == 0 else x * math.log(y)


def kupiec(n: int, x: int, p: float) -> tuple:
    """예외 횟수가 기대 비율 p와 맞는지 (우도비, p값)."""
    if n <= 0:
        return np.nan, np.nan
    ph = x / n
    lr = -2 * (_xlogy(n - x, 1 - p) + _xlogy(x, p) - _xlogy(n - x, 1 - ph) - _xlogy(x, ph))
    return lr, chi2_sf(max(lr, 0.0), 1)


def christoffersen(hits: np.ndarray) -> tuple:
    """예외가 연달아 몰리는지 (독립성 우도비, p값)."""
    h = np.asarray(hits, int)
    if len(h) < 3:
        return np.nan, np.nan
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    l0 = _xlogy(n00 + n10, 1 - p) + _xlogy(n01 + n11, p)
    l1 = _xlogy(n00, 1 - p01) + _xlogy(n01, p01) + _xlogy(n10, 1 - p11) + _xlogy(n11, p11)
    lr = -2 * (l0 - l1)
    return lr, chi2_sf(max(lr, 0.0), 1)


def var_forecasts(r: pd.Series, conf: float, window: int, method: str, lam: float = 0.94) -> pd.Series:
    """t일의 VaR(손실을 양수로)를 t−1일까지의 자료로 예측."""
    r = r.dropna()
    a = 1 - conf
    if method == "역사적 시뮬레이션":
        v = -r.rolling(window).quantile(a).shift(1)
    elif method == "정규분포(분산-공분산)":
        v = -(r.rolling(window).mean() - Z[conf] * r.rolling(window).std()).shift(1)
    else:  # EWMA (리스크메트릭스)
        s2 = np.empty(len(r))
        x = r.values
        s2[0] = np.var(x[:window]) if len(x) >= window else np.var(x)
        for t in range(1, len(x)):
            s2[t] = lam * s2[t - 1] + (1 - lam) * x[t - 1] ** 2
        v = pd.Series(Z[conf] * np.sqrt(s2), index=r.index)
        v.iloc[:window] = np.nan
    return v


def var_backtest(r: pd.Series, conf: float, window: int, method: str) -> dict:
    v = var_forecasts(r, conf, window, method)
    d = pd.DataFrame({"r": r, "var": v}).dropna()
    hits = (d["r"] < -d["var"]).astype(int).values
    n, x = len(d), int(hits.sum())
    lr_uc, p_uc = kupiec(n, x, 1 - conf)
    lr_ind, p_ind = christoffersen(hits)
    lr_cc = lr_uc + lr_ind if np.isfinite(lr_uc) and np.isfinite(lr_ind) else np.nan
    last = d.iloc[-250:]
    v99 = var_forecasts(r, 0.99, window, method).reindex(last.index)
    basel = int((last["r"] < -v99).sum())
    return {"d": d, "hits": hits, "n": n, "x": x, "exp": n * (1 - conf), "rate": x / n if n else np.nan,
            "p_uc": p_uc, "p_ind": p_ind, "p_cc": chi2_sf(lr_cc, 2) if np.isfinite(lr_cc) else np.nan, "basel": basel,
            "basel_n": len(last)}


def basel_zone(x: int) -> str:
    return "초록 (정상)" if x <= 4 else ("노랑 (주의)" if x <= 9 else "빨강 (모형 부적합)")


def beta_to(r_a: pd.Series, r_b: pd.Series) -> float:
    j = pd.concat([r_a, r_b], axis=1, join="inner").dropna()
    if len(j) < 60 or j.iloc[:, 1].var() <= 0:
        return 1.0
    return float(np.cov(j.iloc[:, 0], j.iloc[:, 1], ddof=1)[0, 1] / j.iloc[:, 1].var(ddof=1))


def scenario_return(series: dict, weights: dict, bench: pd.Series | None, betas: dict, s: str, e: str) -> dict:
    """구간 [s, e] 매수 후 보유 손익. series: 이름 → 그 구간 종가(없으면 빈 시리즈)."""
    s, e = pd.Timestamp(s), pd.Timestamp(e)
    b_ret, b_path = np.nan, None
    if bench is not None and len(bench):
        bb = bench[(bench.index >= s - pd.Timedelta(days=5)) & (bench.index <= e + pd.Timedelta(days=1))].dropna()
        bb = bb[bb.index >= bb.index[bb.index <= s].max()] if (bb.index <= s).any() else bb
        if len(bb) >= 2:
            b_ret = float(bb.iloc[-1] / bb.iloc[0] - 1)
            b_path = bb / bb.iloc[0]
    paths, proxy = {}, []
    for nm, w in weights.items():
        x = series.get(nm)
        ok = x is not None and len(x) >= 2 and x.index.min() <= s + pd.Timedelta(days=7) and x.index.max() >= e - pd.Timedelta(days=7)
        if ok:
            x = x[(x.index >= s - pd.Timedelta(days=5)) & (x.index <= e + pd.Timedelta(days=1))]
            x = x[x.index >= x.index[x.index <= s].max()] if (x.index <= s).any() else x
            paths[nm] = x / x.iloc[0]
        elif b_path is not None:
            paths[nm] = 1 + betas.get(nm, 1.0) * (b_path - 1)
            proxy.append(nm)
        else:
            return {"ok": False}
    P = pd.DataFrame(paths).sort_index().ffill().dropna()
    if len(P) < 2:
        return {"ok": False}
    v = sum(P[nm] * w for nm, w in weights.items())
    dd = float((v / v.cummax() - 1).min())
    return {"ok": True, "ret": float(v.iloc[-1] - 1), "mdd": dd, "bench": b_ret, "proxy": proxy, "path": v,
            "contrib": {nm: float(weights[nm] * (P[nm].iloc[-1] - 1)) for nm in weights}}


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
def _p(x, sign=True, nd=1):
    return "-" if x is None or not np.isfinite(x) else (f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%")


def render_stress(items: list, names: list, w: list, P: pd.DataFrame, bench: pd.Series | None, bench_code, bench_name: str,
                  load_close, value_now: float, key: str) -> None:
    """items: 종목 정보, w: 비중, P: 백테스트 기간 종가, load_close(code, start, end) → 종가 시리즈."""
    weights = dict(zip(names, np.asarray(w, float) / np.sum(w)))
    r = P.pct_change().dropna()
    port_r = (r * pd.Series(weights)).sum(axis=1)
    betas = {}
    br = None
    if bench is not None and len(bench):
        br = bench.pct_change().dropna()
        for nm in names:
            betas[nm] = beta_to(r[nm], br)
    beta_p = beta_to(port_r, br) if br is not None else np.nan

    # ---- 최악의 구간 (백테스트 기간)
    st.markdown("###### 백테스트 기간 중 최악의 구간 (같은 비중, 매일 리밸런싱 가정)")
    v = (1 + port_r).cumprod()
    cells = []
    for nday in (1, 5, 20, 60):
        rr = (v / v.shift(nday) - 1).dropna()
        if len(rr):
            d = rr.idxmin()
            cells.append((f"최악의 {nday}일", _p(float(rr.min())), f"{(d - pd.tseries.offsets.BDay(nday)):%Y-%m-%d} ~ {d:%Y-%m-%d}"))
    mgrid(cells, cols=4)

    # ---- 가상 충격
    st.markdown("###### 가상 시장 충격")
    if np.isfinite(beta_p):
        rows = []
        for sh in (-0.05, -0.10, -0.20, -0.30, -0.40):
            loss = beta_p * sh
            rows.append([f"{bench_name} {sh * 100:.0f}%", _p(loss), f"{loss * value_now:+,.0f}"])
        st.markdown(table_html(rows, ["시나리오", "포트폴리오 손익", "평가액 변화"]), unsafe_allow_html=True)
        mgrid([("포트폴리오 베타", f"{beta_p:.2f}", f"{bench_name} 대비, 백테스트 기간")], cols=3)
    else:
        st.caption("벤치마크를 고르면 시장 충격 시나리오를 계산합니다.")

    # ---- 과거 위기 재현
    st.markdown("###### 과거 위기 재현 (지금 비중 그대로)")
    pick = st.multiselect("시나리오", list(SCENARIOS), default=list(SCENARIOS), key=key + "_sc")
    run = st.button("과거 위기 시나리오 계산", key=key + "_run", help="종목마다 위기 구간 가격을 새로 불러와 시간이 걸릴 수 있습니다")
    sig = (tuple(names), tuple(np.round(w, 4)), bench_code, tuple(pick))
    if run:
        out = []
        prog = st.progress(0.0, text="위기 구간 가격을 불러오는 중...")
        for i, sc in enumerate(pick):
            s, e = SCENARIOS[sc]
            ser = {}
            for it in items:
                try:
                    x = load_close(it["code"], (pd.Timestamp(s) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                                   (pd.Timestamp(e) + pd.Timedelta(days=3)).strftime("%Y-%m-%d"))
                    x.index = pd.to_datetime(x.index)
                    x.index = x.index.tz_localize(None) if x.index.tz is not None else x.index
                    ser[it["name"]] = x.dropna()
                except Exception:  # noqa: BLE001
                    ser[it["name"]] = pd.Series(dtype=float)
            bser = None
            if bench_code:
                try:
                    bser = load_close(bench_code, (pd.Timestamp(s) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                                      (pd.Timestamp(e) + pd.Timedelta(days=3)).strftime("%Y-%m-%d"))
                    bser.index = pd.to_datetime(bser.index)
                    bser.index = bser.index.tz_localize(None) if bser.index.tz is not None else bser.index
                except Exception:  # noqa: BLE001
                    bser = None
            res = scenario_return(ser, weights, bser, betas, s, e)
            out.append((sc, s, e, res))
            prog.progress((i + 1) / max(len(pick), 1))
        prog.empty()
        st.session_state[key + "_res"] = (sig, out)
    saved = st.session_state.get(key + "_res")
    if not saved or saved[0] != sig:
        st.info("'과거 위기 시나리오 계산'을 누르면 계산합니다.")
        return
    out = saved[1]
    rows = []
    for sc, s, e, res in out:
        if not res.get("ok"):
            rows.append([sc, f"{s} ~ {e}", "-", "-", "-", "-", "자료 없음"])
            continue
        rows.append([sc, f"{s} ~ {e}", _p(res["ret"]), _p(res["mdd"]), _p(res["bench"]), f"{res['ret'] * value_now:+,.0f}",
                     ", ".join(res["proxy"]) if res["proxy"] else "없음"])
    st.markdown(table_html(rows, ["시나리오", "기간", "포트폴리오 손익", "최대 낙폭", "벤치마크 손익", "평가액 변화", "베타로 대신 계산한 종목"]),
                unsafe_allow_html=True)
    ok = [(sc, res) for sc, _s, _e, res in out if res.get("ok")]
    if ok:
        f = go.Figure()
        for sc, res in ok:
            pth = res["path"]
            f.add_trace(go.Scatter(x=list(range(len(pth))), y=(pth.values - 1) * 100, name=sc, mode="lines",
                                   hovertemplate=sc + " %{x}거래일째 %{y:+.1f}%<extra></extra>"))
        f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), xaxis_title="위기 시작 후 거래일",
                        yaxis=dict(title="누적 손익(%)", ticksuffix="%"), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(f, use_container_width=True)
        sc0 = st.selectbox("종목별 기여도 보기", [sc for sc, _ in ok], key=key + "_ct")
        ct = dict(ok)[sc0]["contrib"]
        f2 = go.Figure(go.Bar(x=list(ct), y=[v * 100 for v in ct.values()], marker_color=["#5BB56E" if v >= 0 else "#E5534B" for v in ct.values()],
                              hovertemplate="%{x} %{y:+.2f}%p<extra></extra>"))
        f2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), yaxis=dict(title="손익 기여(%p)", ticksuffix="%p"))
        st.plotly_chart(f2, use_container_width=True)


def render_var_test(curves: dict, key: str) -> None:
    c1, c2, c3 = st.columns(3)
    k = c1.selectbox("전략", list(curves), key=key + "_k")
    conf = c2.radio("신뢰수준", [0.95, 0.99], format_func=lambda x: f"{x * 100:.0f}%", horizontal=True, key=key + "_c")
    window = c3.select_slider("추정 기간 (거래일)", [60, 125, 250, 500], value=250, key=key + "_w")
    r = curves[k].pct_change().dropna()
    if len(r) < window + 60:
        st.info(f"VaR 검증에는 추정 기간({window}일) + 60거래일 이상의 자료가 필요합니다. 백테스트 기간을 늘리세요.")
        return
    methods = ["역사적 시뮬레이션", "정규분포(분산-공분산)", "EWMA (λ=0.94)"]
    res = {m: var_backtest(r, conf, window, m) for m in methods}
    rows = []
    for m, R in res.items():
        verdict = "적합" if (R["p_uc"] > 0.05 and (not np.isfinite(R["p_ind"]) or R["p_ind"] > 0.05)) else "부적합"
        rows.append([m, f"{R['x']}회", f"{R['exp']:.1f}회", _p(R["rate"], False, 2), f"{R['p_uc']:.3f}",
                     f"{R['p_ind']:.3f}" if np.isfinite(R["p_ind"]) else "-", f"{R['p_cc']:.3f}" if np.isfinite(R["p_cc"]) else "-",
                     f"{R['basel']}회 · {basel_zone(R['basel'])}", verdict])
    st.markdown(table_html(rows, ["방법", "예외 횟수", "기대 예외 횟수", "예외율", "쿠픽 p값", "독립성 p값", "종합 p값", "바젤 신호등", "판정"]),
                unsafe_allow_html=True)
    m = st.radio("그래프로 볼 방법", methods, horizontal=True, key=key + "_m")
    R = res[m]
    d = R["d"]
    mgrid([("예외 횟수", f"{R['x']}회 / {R['n']}일"), ("기대 예외 횟수", f"{R['exp']:.1f}회"), ("쿠픽 p값", f"{R['p_uc']:.3f}"),
           ("바젤 신호등", basel_zone(R["basel"]), f"최근 {R['basel_n']}일 99% VaR 예외 {R['basel']}회")], cols=4)
    f = go.Figure()
    f.add_trace(go.Bar(x=d.index, y=d["r"] * 100, name="일별 손익", marker_color="rgba(128,128,128,0.45)",
                       hovertemplate="%{x|%Y-%m-%d} %{y:+.2f}%<extra></extra>"))
    f.add_trace(go.Scatter(x=d.index, y=-d["var"] * 100, name=f"−VaR {conf * 100:.0f}%", line=dict(color="#4C9BE8", width=1.3),
                           hovertemplate="%{x|%Y-%m-%d} VaR %{y:.2f}%<extra></extra>"))
    ex = d[R["hits"].astype(bool)]
    f.add_trace(go.Scatter(x=ex.index, y=ex["r"] * 100, mode="markers", name="예외(VaR 초과 손실)", marker=dict(color="#E5534B", size=7),
                           hovertemplate="%{x|%Y-%m-%d} %{y:+.2f}%<extra>예외</extra>"))
    f.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                    yaxis=dict(title="일별 손익(%)", ticksuffix="%"), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(f, use_container_width=True)
