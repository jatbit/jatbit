"""포트폴리오 백테스트 탭
- '포트폴리오' 페이지에서 구한 비중(최대 샤프·최소 분산·동일 비중) 또는 직접 정한 비중을 과거 가격에 적용
- 리밸런싱 주기·거래비용 반영, 벤치마크 비교
- 지표: 누적 수익률, 연평균 수익률(CAGR), 변동성, 샤프·소르티노·칼마, 최대 낙폭(MDD), 월간 승률, 베타·알파, 정보비율, 회전율
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

from options import mgrid

TD = 252


def _month_end(v: pd.Series) -> pd.Series:
    try:
        return v.resample("ME").last()
    except Exception:  # noqa: BLE001  (pandas 2.2 미만)
        return v.resample("M").last()
REBAL = {"리밸런싱 안 함 (매수 후 보유)": None, "매월": "M", "분기마다": "Q", "반기마다": "H", "매년": "Y"}
COLORS = ["#E5534B", "#4C9BE8", "#5BB56E", "#F2A93B", "#9B6FD6", "#3BB5B5"]


# ----------------------------------------------------------------------
# 계산
# ----------------------------------------------------------------------
def _period_key(idx: pd.DatetimeIndex, freq: str) -> np.ndarray:
    if freq == "M":
        return idx.year * 100 + idx.month
    if freq == "Q":
        return idx.year * 10 + (idx.month - 1) // 3
    if freq == "H":
        return idx.year * 10 + (idx.month - 1) // 6
    return idx.year.values


def simulate(P: pd.DataFrame, w, freq: str | None, cost_bp: float = 0.0, init: float = 1.0) -> dict:
    """주식 수 기준 시뮬레이션. 리밸런싱 날(새 기간 첫 거래일) 종가에 목표 비중으로 맞추고, 거래금액 × 비용을 뺀다."""
    w = np.asarray(w, float)
    w = w / w.sum()
    px = P.values
    n_days = len(P)
    c = cost_bp / 1e4
    reb = np.zeros(n_days, bool)
    if freq:
        k = _period_key(P.index, freq)
        reb[1:] = k[1:] != k[:-1]
    V = np.empty(n_days)
    Wt = np.empty_like(px)
    cost_tot, turn_tot = init * c, 1.0  # 첫 매수
    v = init * (1 - c)
    units = v * w / px[0]
    V[0] = v
    Wt[0] = units * px[0] / v
    for t in range(1, n_days):
        hold = units * px[t]
        v = hold.sum()
        if reb[t]:
            # 목표 비중으로 맞출 때 거래금액 T = Σ|v'w − hold|, v' = v − cT (고정점 반복)
            v_new = v
            for _ in range(5):
                trade = np.abs(v_new * w - hold).sum()
                v_new = v - c * trade
            trade = np.abs(v_new * w - hold).sum()
            cost_tot += c * trade
            turn_tot += trade / v / 2 if v else 0
            v = v_new
            units = v * w / px[t]
            hold = units * px[t]
        V[t] = v
        Wt[t] = hold / v if v else w
    vals = pd.Series(V, P.index)
    return {"value": vals, "weights": pd.DataFrame(Wt, P.index, P.columns), "cost": cost_tot,
            "turnover_y": (turn_tot - 1) / max((P.index[-1] - P.index[0]).days / 365.25, 1e-9), "n_reb": int(reb.sum())}


def metrics(v: pd.Series, rf: float, bench: pd.Series | None = None) -> dict:
    r = v.pct_change().dropna()
    yrs = (v.index[-1] - v.index[0]).days / 365.25
    tot = v.iloc[-1] / v.iloc[0] - 1
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(TD)
    ex = r - rf / TD
    sharpe = ex.mean() * TD / vol if vol > 0 else np.nan
    dd_std = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(TD)
    sortino = ex.mean() * TD / dd_std if dd_std > 0 else np.nan
    dd = v / v.cummax() - 1
    mdd = dd.min()
    trough = dd.idxmin()
    peak = v[:trough].idxmax()
    rec = v[trough:][v[trough:] >= v[peak]]
    rec_d = rec.index[0] if len(rec) else None
    m = _month_end(v).pct_change().dropna()
    out = {"tot": tot, "cagr": cagr, "vol": vol, "sharpe": sharpe, "sortino": sortino, "mdd": mdd,
           "calmar": cagr / abs(mdd) if mdd < 0 else np.nan, "peak": peak, "trough": trough, "recover": rec_d,
           "win_m": (m > 0).mean() if len(m) else np.nan, "best_m": m.max() if len(m) else np.nan, "worst_m": m.min() if len(m) else np.nan,
           "var95": -np.percentile(r, 5) if len(r) else np.nan}
    if bench is not None:
        rb = bench.pct_change().reindex(r.index).dropna()
        ra = r.reindex(rb.index)
        if len(rb) > 20 and rb.var() > 0:
            beta = np.cov(ra, rb, ddof=1)[0, 1] / rb.var(ddof=1)
            alpha = ((ra - rf / TD).mean() - beta * (rb - rf / TD).mean()) * TD
            te = (ra - rb).std(ddof=1) * np.sqrt(TD)
            out.update({"beta": beta, "alpha": alpha, "corr": ra.corr(rb), "te": te,
                        "ir": (ra - rb).mean() * TD / te if te > 0 else np.nan})
    return out


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
def _pct(x, sign=False, nd=1):
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%"


def _r(x, nd=2):
    return f"{x:.{nd}f}" if x is not None and np.isfinite(x) else "-"


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _load_close(_load_prices, code: str, start: str, end: str) -> pd.Series:
    d = _load_prices(code, start, end)
    return d["Close"].dropna() if d is not None and "Close" in d else pd.Series(dtype=float)


def render_backtest(load_prices, find_candidates, listing, risk_free_for) -> None:
    st.caption("과거 가격에 비중을 적용해 '그때 이렇게 투자했다면'을 계산합니다. 국내 종목은 가격 수익률 기준, 환율 변화는 반영하지 않습니다.")
    res = st.session_state.get("pf_opt_result") or {}
    sets = dict(res.get("weight_sets") or {})
    if res.get("weights") and not sets:
        sets = {"최대 샤프 (최적)": res["weights"]}
    items = list(res.get("items") or [])

    src_opts = (["포트폴리오 페이지 결과"] if items else []) + ["직접 종목 입력"]
    src = st.radio("종목 가져오기", src_opts, horizontal=True, key="bt_src")
    if src == "직접 종목 입력":
        raw = st.text_input("종목 (쉼표로 구분)", key="bt_codes", placeholder="예: 삼성전자, SK하이닉스, AAPL, SPY, 금 선물")
        items = []
        for tok in [t.strip() for t in (raw or "").split(",") if t.strip()]:
            c = find_candidates(tok, listing)
            if c:
                items.append({"code": c[0]["code"], "name": c[0]["name"], "domestic": c[0]["domestic"]})
            else:
                st.warning(f"'{tok}' 종목을 찾지 못했습니다.")
        sets = {"동일 비중": [1 / len(items)] * len(items)} if items else {}
    else:
        st.caption(f"최적화 기준일 {res.get('asof', '-')} · 최적화 구간 {res.get('start', '-')} ~ {res.get('asof', '-')}. "
                   "같은 구간을 백테스트하면 미래 정보를 쓴 결과(사후 최적화)라 실제보다 좋게 나옵니다. 최적화 구간 이후를 따로 보면 더 공정합니다.")
    if len(items) < 1:
        st.info("'포트폴리오' 페이지에서 최적화를 먼저 하거나, 위에서 종목을 직접 입력하세요.")
        return
    names = [it["name"] for it in items]

    # ---- 비중
    with st.expander("직접 비중 정하기", expanded=src == "직접 종목 입력"):
        base = sets.get("최대 샤프 (최적)") or [1 / len(items)] * len(items)
        ed = st.data_editor(pd.DataFrame({"종목": names, "비중(%)": [round(x * 100, 1) for x in base]}), hide_index=True,
                            disabled=["종목"], key=f"bt_w_{'|'.join(names)}", use_container_width=True)
        wv = pd.to_numeric(ed["비중(%)"], errors="coerce").fillna(0).clip(lower=0).values
        if wv.sum() > 0:
            sets["직접 입력"] = list(wv / wv.sum())
            st.caption(f"합계 {wv.sum():.1f}% → 100%로 맞춰서 사용합니다.")
    pick = st.multiselect("비교할 비중", list(sets), default=list(sets)[:3], key="bt_sets")
    if not pick:
        st.info("비교할 비중을 하나 이상 고르세요.")
        return

    # ---- 조건
    today = date.today()
    try:
        c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.3, 1.2, 1], vertical_alignment="bottom")
    except TypeError:
        c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.3, 1.2, 1])
    start = c1.date_input("시작일", value=today - timedelta(days=365 * 5), min_value=date(1990, 1, 1), max_value=today, key="bt_start")
    end = c2.date_input("종료일", value=today, min_value=date(1990, 1, 2), max_value=today, key="bt_end")
    init = c3.number_input("초기 투자금", value=10_000_000, step=1_000_000, key="bt_init", format="%d")
    reb_name = c4.selectbox("리밸런싱", list(REBAL), index=2, key="bt_reb")
    domestic_all = all(it["domestic"] for it in items)
    b_opts = {"S&P 500": "US500", "코스피 지수": "KS11", "코스닥 지수": "KQ11", "나스닥 종합": "IXIC", "없음": None}
    with c5:
        with ui.popover("세부 조건", icon=":material/tune:"):
            cost_bp = st.number_input("거래비용 (bp, 매매금액 대비)", value=10.0, step=5.0, key="bt_cost",
                                      help="수수료+세금+슬리피지. 10bp = 0.1%")
            bench_name = st.selectbox("벤치마크", list(b_opts), index=1 if domestic_all else 0, key="bt_bench")
            also_bh = st.checkbox("같은 비중의 '리밸런싱 안 함'도 함께 비교", value=REBAL[reb_name] is not None, key="bt_bh")
    if start >= end:
        st.warning("시작일이 종료일보다 앞이어야 합니다.")
        return

    # ---- 가격
    series = {}
    with st.spinner("가격을 불러오는 중..."):
        for it in items:
            try:
                s = _load_close(load_prices, it["code"], start.isoformat(), end.isoformat())
            except Exception as e:  # noqa: BLE001
                st.error(f"{it['name']}: 가격을 불러오지 못했습니다 ({e})")
                return
            if s.empty:
                st.error(f"{it['name']}: 선택한 기간에 가격이 없습니다.")
                return
            series[it["name"]] = s
        bench = None
        if b_opts[bench_name]:
            try:
                bench = _load_close(load_prices, b_opts[bench_name], start.isoformat(), end.isoformat())
            except Exception:  # noqa: BLE001
                st.caption("벤치마크를 불러오지 못해 빼고 계산합니다.")
    P = pd.concat(series, axis=1).sort_index()
    P.index = pd.to_datetime(P.index).tz_localize(None) if getattr(P.index, "tz", None) is not None else pd.to_datetime(P.index)
    first = P.apply(lambda s: s.first_valid_index()).max()
    P = P[P.index >= first].ffill().dropna()  # 국내·해외 휴장일이 달라 직전 종가로 채움
    if len(P) < 30:
        st.warning("모든 종목의 가격이 겹치는 기간이 너무 짧습니다.")
        return
    if first.date() > start + timedelta(days=7):
        late = [nm for nm in names if series[nm].index.min() >= first - pd.Timedelta(days=3)]
        st.info(f"{', '.join(late)}의 가격이 {first:%Y-%m-%d}부터 있어 이날부터 계산합니다.")
    if bench is not None and len(bench):
        bench.index = pd.to_datetime(bench.index).tz_localize(None) if getattr(bench.index, "tz", None) is not None else pd.to_datetime(bench.index)
        bench = bench.reindex(P.index.union(bench.index)).ffill().reindex(P.index).dropna()
        bench = bench / bench.iloc[0] * init if len(bench) else None
    rf, rf_label = risk_free_for(domestic_all, P.index[-1].date())

    runs = {}
    for nm in pick:
        runs[f"{nm}"] = simulate(P[names], sets[nm], REBAL[reb_name], cost_bp, init)
        if also_bh and REBAL[reb_name] is not None:
            runs[f"{nm} · 보유만"] = simulate(P[names], sets[nm], None, cost_bp, init)
    M = {k: metrics(r["value"], rf, bench) for k, r in runs.items()}
    if bench is not None and len(bench):
        M[f"벤치마크 ({bench_name})"] = metrics(bench, rf, None)

    # ---- 요약
    ui.section("백테스트 결과", f"{P.index[0]:%Y-%m-%d} ~ {P.index[-1]:%Y-%m-%d} ({(P.index[-1] - P.index[0]).days / 365.25:.1f}년)")
    k0 = pick[0]
    m0, r0 = M[k0], runs[k0]
    bm = M.get(f"벤치마크 ({bench_name})")
    ui.cards([("최종 평가액", f"{r0['value'].iloc[-1]:,.0f}", f"초기 {init:,.0f} · {k0}", m0["tot"]),
              ("누적 수익률", _pct(m0["tot"], True), f"벤치마크 {_pct(bm['tot'], True)}" if bm else "", m0["tot"]),
              ("연평균 수익률(CAGR)", _pct(m0["cagr"], True), f"벤치마크 {_pct(bm['cagr'], True)}" if bm else "", m0["cagr"]),
              ("최대 낙폭(MDD)", _pct(m0["mdd"]), f"{m0['peak']:%Y-%m-%d} → {m0['trough']:%Y-%m-%d}"),
              ("연 변동성", _pct(m0["vol"]), f"벤치마크 {_pct(bm['vol'])}" if bm else ""),
              ("샤프 비율", _r(m0["sharpe"]), f"무위험 {rf * 100:.2f}%"),
              ("월간 승률", _pct(m0["win_m"], nd=0)),
              ("총 거래비용", f"{r0['cost']:,.0f}", f"리밸런싱 {r0['n_reb']}회 · 연 회전율 {r0['turnover_y'] * 100:.0f}%")], cols=4)

    # ---- 누적 가치 (바로 보여줌)
    curves = {k: r["value"] for k, r in runs.items()}
    if bench is not None and len(bench):
        curves[f"벤치마크 ({bench_name})"] = bench
    logy = st.toggle("로그 눈금", value=False, key="bt_log")
    f = go.Figure()
    for i, (k, v) in enumerate(curves.items()):
        dash = "dot" if "보유만" in k else ("dash" if k.startswith("벤치마크") else None)
        color = "#888" if k.startswith("벤치마크") else COLORS[(pick.index(k.split(" · ")[0]) if k.split(" · ")[0] in pick else i) % len(COLORS)]
        f.add_trace(go.Scatter(x=v.index, y=v.values, name=k, line=dict(color=color, width=1.6, dash=dash),
                               hovertemplate="%{x|%Y-%m-%d} " + k + " %{y:,.0f}<extra></extra>"))
    f.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                    yaxis=dict(type="log" if logy else "linear", title="평가액"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(f, use_container_width=True)

    rows = []
    for k, m in M.items():
        rows.append({"전략": k, "누적": _pct(m["tot"], True), "CAGR": _pct(m["cagr"], True), "변동성": _pct(m["vol"]),
                     "샤프": _r(m["sharpe"]), "소르티노": _r(m["sortino"]), "MDD": _pct(m["mdd"]), "칼마": _r(m["calmar"]),
                     "월 승률": _pct(m["win_m"], nd=0), "최고 월": _pct(m["best_m"], True), "최악 월": _pct(m["worst_m"], True),
                     "1일 VaR95": _pct(m["var95"]), "베타": _r(m.get("beta", np.nan)), "알파(연)": _pct(m.get("alpha", np.nan), True),
                     "추적오차": _pct(m.get("te", np.nan)), "정보비율": _r(m.get("ir", np.nan))})
    from glossary import col_help

    exp, op = ui.more("전략별 지표 전체 더보기 · 소르티노 · 칼마 · 베타 · 알파 · 정보비율", key="bt_more_tbl")
    if op:
        with exp:
            tb = pd.DataFrame(rows)
            st.dataframe(tb, use_container_width=True, hide_index=True, column_config=col_help(tb.columns))
            st.caption(f"무위험수익률 {rf * 100:.2f}% ({rf_label}). MDD 회복일: "
                       + (f"{m0['recover']:%Y-%m-%d}" if m0["recover"] is not None else "아직 회복 전"))

    # ---- 그래프
    ui.section("자세히 보기")
    (t2, t2_open), (t3, t3_open), (t4, t4_open), (t5, t5_open), (t6, t6_open), (t7, t7_open) = ui.lazy_tabs(
        ["낙폭", "연도별 수익률", "월별 수익률", "비중 변화", "스트레스 테스트", "VaR 검증"], key="bt_tabs")
    if t2_open:
        with t2:
            f = go.Figure()
            for i, (k, v) in enumerate(curves.items()):
                dd = (v / v.cummax() - 1) * 100
                f.add_trace(go.Scatter(x=dd.index, y=dd.values, name=k, fill="tozeroy" if i == 0 else None,
                                       line=dict(width=1.2, color="#888" if k.startswith("벤치마크") else COLORS[i % len(COLORS)]),
                                       hovertemplate="%{x|%Y-%m-%d} %{y:.1f}%<extra>" + k + "</extra>"))
            f.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                            yaxis=dict(title="고점 대비(%)", ticksuffix="%"), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(f, use_container_width=True)
    if t3_open:
        with t3:
            f = go.Figure()
            for i, (k, v) in enumerate(curves.items()):
                y = v.groupby(v.index.year).last()
                prev = pd.concat([pd.Series([v.iloc[0]], index=[y.index[0] - 1]), y.iloc[:-1]])
                yr = (y.values / prev.values - 1) * 100
                f.add_trace(go.Bar(x=[str(a) for a in y.index], y=yr, name=k,
                                   marker_color="#888" if k.startswith("벤치마크") else COLORS[i % len(COLORS)],
                                   hovertemplate="%{x} %{y:+.1f}%<extra>" + k + "</extra>"))
            f.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                            yaxis=dict(title="수익률(%)", ticksuffix="%"), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(f, use_container_width=True)
            st.caption("첫 해와 마지막 해는 일부 기간만 포함합니다.")
    if t4_open:
        with t4:
            k = st.selectbox("전략", list(curves), key="bt_mh")
            v = curves[k]
            mr = _month_end(v)
            mr = pd.concat([pd.Series([v.iloc[0]], index=[v.index[0]]), mr]).pct_change().dropna()
            tbl = pd.DataFrame({"y": mr.index.year, "m": mr.index.month, "r": mr.values * 100}).pivot(index="y", columns="m", values="r")
            tbl = tbl.reindex(columns=range(1, 13))
            z = tbl.values
            hm = go.Figure(go.Heatmap(z=z, x=[f"{m}월" for m in range(1, 13)], y=[str(y) for y in tbl.index], colorscale="RdYlGn", zmid=0,
                                      text=[[f"{x:+.1f}" if np.isfinite(x) else "" for x in row] for row in z], texttemplate="%{text}",
                                      hovertemplate="%{y}년 %{x} %{z:+.2f}%<extra></extra>", showscale=False))
            hm.update_layout(height=max(220, 28 * len(tbl) + 60), margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                             yaxis=dict(autorange="reversed", type="category"))
            st.plotly_chart(hm, use_container_width=True)
    if t5_open:
        with t5:
            k = st.selectbox("전략", list(runs), key="bt_wk")
            W = runs[k]["weights"]
            f = go.Figure()
            for i, c in enumerate(W.columns):
                f.add_trace(go.Scatter(x=W.index, y=W[c] * 100, name=c, stackgroup="one", mode="lines", line=dict(width=0.5),
                                       hovertemplate="%{x|%Y-%m-%d} " + c + " %{y:.1f}%<extra></extra>"))
            f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                            yaxis=dict(title="비중(%)", ticksuffix="%", range=[0, 100]), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(f, use_container_width=True)
            st.caption("리밸런싱 사이에는 가격 변화에 따라 비중이 흘러가고(드리프트), 리밸런싱 날 목표 비중으로 돌아옵니다.")
    from risklab import render_stress, render_var_test

    if t6_open:
        with t6:
            k_s = st.selectbox("비중", pick, key="bt_st_k")
            render_stress(items, names, sets[k_s], P[names], bench, b_opts[bench_name], bench_name,
                          lambda c, a, b: _load_close(load_prices, c, a, b), float(runs[k_s]["value"].iloc[-1]), "bt_st")
    if t7_open:
        with t7:
            render_var_test(curves, "bt_var")
