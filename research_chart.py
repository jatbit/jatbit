"""기업 리서치: 재무비율·주가·밸류에이션을 골라 겹쳐 그리는 그래프 (일별 PER 등)."""
import html
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

from options import OPT_PRESETS, is_us_ticker, mgrid

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None

from research_data import *  # noqa: F401,F403


def _combo_chart(sec, R: dict, ticker: str, px, n_eps, src: str = "SEC 10-K") -> None:
    """연간 재무비율(SEC)과 일별 주가·PER 지표를 골라서 한 그래프에 겹쳐 그린다."""
    rt = compute_ratios(sec, R.get("hist"), R.get("splits")) if sec is not None else None
    avail = [k for k in RATIOS if rt is not None and k in rt.index and rt.loc[k].notna().sum() >= 2]
    per = per_series(R.get("hist"), R.get("edates"))
    hist = R.get("hist")
    d_avail = ["주가"] if isinstance(hist, pd.Series) and len(hist) else []
    if per is not None:
        d_avail += ["12M 트레일링 PER", "12M 선행 PER (사후 실현 EPS)", "트레일링 PER 평균 ±1σ 밴드"]
    ann = annual_per_series(hist, sec, R.get("splits"))
    if ann is not None:
        d_avail.insert(1 if d_avail[:1] == ["주가"] else 0, "PER (직전 사업연도 EPS)")
    if px and n_eps and n_eps > 0:
        d_avail.append("현재 12M 선행 PER (컨센서스)")
    dv, dv_err = None, None
    if isinstance(hist, pd.Series) and len(hist) and _VAL:
        try:
            from valuation import daily_valuation, load_quarterly
            if src == "SEC 10-K":
                dv = daily_valuation(load_quarterly(ticker), hist, R.get("splits"))
            elif sec is not None:  # 국내·SEC 미등록 종목: 연간 재무제표를 공시 시점부터 반영 (다음 연간 공시까지 유지)
                from kr import annual_quarterly_dict
                info = R.get("info") or {}
                dv = daily_valuation(annual_quarterly_dict(sec, _num(info.get("sharesOutstanding"))), hist, R.get("splits"), stale=470)
            else:
                raise RuntimeError("재무제표 없음")
            d_avail += [k for k in _VAL if k in dv and dv[k].notna().sum() > 20]
        except Exception as e:  # noqa: BLE001
            dv, dv_err = None, str(e)

    # 일별로 계산되는 지표가 있으면 같은 이름의 연간 밸류에이션 비율(PER·PBR·PSR)은 빼고, 시가총액 기준 PER은
    # EPS 기준 12M 트레일링 PER과 겹치므로 EPS 기준이 있을 때는 뺀다.
    if (per is not None or ann is not None) and "PER (시가총액 ÷ 순이익 TTM)" in d_avail:
        d_avail.remove("PER (시가총액 ÷ 순이익 TTM)")
    daily_names = set(d_avail) | ({"PER"} if per is not None else set())
    a_avail = [k for k in avail if k not in daily_names]
    val_daily = [k for k in d_avail if k != "주가"]
    options = (["주가"] if "주가" in d_avail else []) + val_daily + a_avail

    def _label(k):
        if k in a_avail:
            return f"{k} · 연간 ({RATIOS[k][1]}, {RATIOS[k][0]})"
        if k in _PER_LABEL:
            return _PER_LABEL[k]
        if k in _VAL:
            return f"{k} · 일별 ({DAILY[k]})"
        return _PER_LABEL.get(k) or (k if k == "주가" else f"{k} · 일별")

    chosen = st.multiselect("그래프에 그릴 항목 (주가·밸류에이션은 일별, 나머지 재무비율은 연간)", options,
                            default=(["주가"] if "주가" in options else [])
                            + [k for k in ("PER (직전 사업연도 EPS)", "12M 트레일링 PER", "PER (시가총액 ÷ 순이익 TTM)") if k in options][:1],
                            key=f"rs_combo_{ticker}",
                            format_func=_label)
    sel = [k for k in chosen if k in a_avail]
    dsel = [k for k in chosen if k in d_avail]
    if dv is None and dv_err:
        st.caption(f"일별 밸류에이션을 계산하지 못해 연간 값으로 보여줍니다 ({dv_err}).")
    elif sec is None:
        st.caption("연간 재무제표가 없어 재무비율을 그릴 수 없습니다.")
    mean_lines = st.checkbox("일별 지표에 기간 평균선 표시", value=False, key=f"rs_daily_mean_{ticker}") if dv is not None else False
    if not sel and not dsel:
        st.info("그릴 항목을 하나 이상 고르세요.")
        return

    # 단위별 축: % → 왼쪽(y), 배·회 → 오른쪽(y2), 주가 → 오른쪽 바깥(y3)
    units = []
    for k in sel:
        units.append("%" if RATIOS[k][0] == "%" else "x")
    for k in dsel:
        units.append({"$": "$", "%": "%"}.get(DAILY[k], "x"))
    order = [u for u in ("%", "x", "$") if u in units]
    axis_of = {u: ("y" if i == 0 else f"y{i + 1}") for i, u in enumerate(order)}
    import money
    title_of = {"%": "%", "x": "배·회", "$": f"주가({money.sym()})"}

    # 일별 지표 시작일을 재무비율 기간에 맞춘다
    start = None
    if rt is not None and sel:
        start = pd.Timestamp(rt.columns[0]) - pd.DateOffset(months=6)
    palette = ["#4C9BE8", "#F2A93B", "#E5534B", "#5BB56E", "#9B6FD6", "#2BB3B1", "#D97AB0", "#8C8C8C", "#C9A227", "#3D7DCA"]
    fig = go.Figure()
    ci = 0
    for k in sel:
        unit = RATIOS[k][0]
        ax = axis_of["%" if unit == "%" else "x"]
        fig.add_trace(go.Scatter(x=list(rt.columns), y=rt.loc[k].values, name=f"{k} ({unit})", mode="lines+markers", yaxis=ax,
                                 line=dict(color=palette[ci % len(palette)], width=2), marker=dict(size=8),
                                 hovertemplate="%{x|%Y-%m} " + k + (" %{y:.1f}%" if unit == "%" else " %{y:.2f}" + unit) + "<extra></extra>"))
        ci += 1

    def _clip(sr):
        sr = sr.copy()
        sr.index = pd.to_datetime(sr.index)
        if sr.index.tz is not None:
            sr.index = sr.index.tz_localize(None)
        return sr[sr.index >= start] if start is not None else sr

    trail = fwd_r = None
    if per is not None:
        trail, fwd_r, _e = per
        trail, fwd_r = _clip(trail), _clip(fwd_r)
    x_vals = []
    for k in dsel:
        ax = axis_of[{"$": "$", "%": "%"}.get(DAILY[k], "x")]
        color = palette[ci % len(palette)]
        if k in _VAL and dv is not None:
            sr = _clip(dv[k]).dropna()
            unit = DAILY[k]
            fig.add_trace(go.Scatter(x=sr.index, y=sr.values, name=_PER_TRACE.get(k, f"{k} (일별)"), mode="lines", yaxis=ax, line=dict(color=color, width=1.3),
                                     hovertemplate="%{x|%Y-%m-%d} " + k + (" %{y:.2f}%" if unit == "%" else " %{y:.2f}배") + "<extra></extra>"))
            if unit != "%":
                x_vals.append(sr.values)
            if mean_lines and len(sr):
                m = float(sr.mean())
                fig.add_trace(go.Scatter(x=[sr.index[0], sr.index[-1]], y=[m, m], name=f"{k} 평균 {m:.2f}", mode="lines", yaxis=ax,
                                         line=dict(color=color, width=1, dash="dash"), hoverinfo="skip"))
        elif k == "주가":
            h = _clip(hist)
            fig.add_trace(go.Scatter(x=h.index, y=h.values, name="주가", mode="lines", yaxis=ax, line=dict(color=color, width=1.4),
                                     hovertemplate="%{x|%Y-%m-%d} 주가 %{y:,.2f}<extra></extra>"))
        elif k == "12M 트레일링 PER":
            fig.add_trace(go.Scatter(x=trail.index, y=trail.values, name="12M PER", mode="lines", yaxis=ax, line=dict(color=color, width=1.4),
                                     hovertemplate="%{x|%Y-%m-%d} 12M PER %{y:.1f}배<extra></extra>"))
        elif k == "12M 선행 PER (사후 실현 EPS)":
            fig.add_trace(go.Scatter(x=fwd_r.index, y=fwd_r.values, name="선행 PER(사후)", mode="lines", yaxis=ax,
                                     line=dict(color=color, width=1.4, dash="dash"),
                                     hovertemplate="%{x|%Y-%m-%d} 선행 PER(사후) %{y:.1f}배<extra></extra>"))
        elif k == "트레일링 PER 평균 ±1σ 밴드":
            tv = trail.dropna()
            if len(tv):
                m, sd = float(tv.mean()), float(tv.std())
                x0, x1 = tv.index[0], tv.index[-1]
                for yv, nm, dash in ((m + sd, f"평균+1σ {m + sd:.1f}배", "dot"), (m, f"PER 평균 {m:.1f}배", "dash"),
                                     (m - sd, f"평균−1σ {m - sd:.1f}배", "dot")):
                    fig.add_trace(go.Scatter(x=[x0, x1], y=[yv, yv], name=nm, mode="lines", yaxis=ax,
                                             line=dict(color="#888", width=1, dash=dash), hoverinfo="skip"))
        elif k == "PER (직전 사업연도 EPS)":
            sr = _clip(ann).dropna()
            fig.add_trace(go.Scatter(x=sr.index, y=sr.values, name="PER", mode="lines", yaxis=ax, line=dict(color=color, width=1.4),
                                     hovertemplate="%{x|%Y-%m-%d} PER %{y:.1f}배<extra></extra>"))
            x_vals.append(sr.values)
        elif k == "현재 12M 선행 PER (컨센서스)":
            now = pd.Timestamp(hist.index[-1]).tz_localize(None) if getattr(hist.index, "tz", None) is not None else pd.Timestamp(hist.index[-1])
            fig.add_trace(go.Scatter(x=[now], y=[px / n_eps], name=f"오늘 선행 PER {px / n_eps:.1f}배", mode="markers", yaxis=ax,
                                     marker=dict(size=14, symbol="star", color="#E5534B"),
                                     hovertemplate=f"현재 12M 선행(NTM) PER {px / n_eps:.1f}배<extra></extra>"))
        ci += 1

    multi = len(order)
    lay = dict(height=420, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
               legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
               xaxis=dict(domain=[0, 0.86] if multi == 3 else [0, 1]))
    for i, u in enumerate(order):
        key = "yaxis" if i == 0 else f"yaxis{i + 1}"
        spec = dict(title=title_of[u], ticksuffix="%" if u == "%" else "", showgrid=(i == 0))
        if i == 1:
            spec.update(overlaying="y", side="right", anchor="x")
        elif i == 2:
            spec.update(overlaying="y", side="right", anchor="free", position=1.0)
        if u == "x" and dsel and (x_vals or (any(k in dsel for k in ("12M 트레일링 PER", "12M 선행 PER (사후 실현 EPS)")) and trail is not None)):
            parts = list(x_vals)
            if trail is not None and any(k in dsel for k in ("12M 트레일링 PER", "12M 선행 PER (사후 실현 EPS)")):
                parts += [trail.dropna().values, fwd_r.dropna().values]
            parts += [rt.loc[k].dropna().values for k in sel if RATIOS[k][0] != "%"]
            vals = np.concatenate(parts) if parts else np.array([])
            if len(vals):
                spec["range"] = [0, float(np.nanpercentile(vals, 98)) * 1.2]  # 일시적인 극단값(적자 직후 등) 때문에 축이 늘어나지 않게
        lay[key] = spec
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True)
    axes_txt = ", ".join(f"{title_of[u]} → {'왼쪽' if i == 0 else ('오른쪽' if i == 1 else '오른쪽 바깥')} 축" for i, u in enumerate(order))
    st.caption(f"축: {axes_txt}")
    try:  # 통계적 이상(공시일·분할·분기 누락·극단값) 점검
        import data_check
        tb = annual_eps_table(sec, R.get("splits"))
        ed = R.get("edates")
        issues = data_check.run_checks(hist, sec, R.get("splits"), ed, ann=ann, trail=per[0] if per is not None else None,
                                       ann_dates=None if tb is None else tb.index,
                                       trail_dates=per[2].index if per is not None else None)
        data_check.render_checks(issues, key=ticker)
    except Exception as e:  # noqa: BLE001
        st.caption(f"데이터 점검을 실행하지 못했습니다 ({e}).")

    if dv is not None:
        from valuation import summary_table
        keys = [k for k in _VAL if k in dv and dv[k].notna().sum() > 20]
        tb = summary_table(dv, keys)
        if len(tb):
            with st.expander("일별 밸류에이션 요약 — 지금이 과거보다 비싼가?", expanded=bool(set(dsel) & set(_VAL))):
                show = tb.copy().astype(object)
                for i, r in tb.iterrows():
                    unit = DAILY[r["지표"]]
                    for c in ("현재", "1년 평균", "3년 평균", "5년 평균", "10년 평균", "최저", "최고"):
                        v = r[c]
                        show.at[i, c] = "-" if pd.isna(v) else (f"{v:.2f}%" if unit == "%" else f"{v:.2f}배")
                    show.at[i, "현재 백분위"] = f"{r['현재 백분위']:.0f}%"
                from glossary import table_html
                st.markdown(table_html(show.values.tolist(), list(show.columns)), unsafe_allow_html=True)

    if sel:
        tbl = rt.loc[sel].copy()
        cols = [pd.Timestamp(c).strftime("%Y-%m") for c in rt.columns]
        tbl = tbl.apply(lambda row: [("-" if pd.isna(v) else (f"{v:.1f}%" if RATIOS[row.name][0] == "%" else f"{v:.2f}"))
                                     for v in row], axis=1, result_type="expand")
        tbl.columns = cols
        from glossary import table_html
        st.markdown(table_html([[k] + list(r) for k, r in zip(tbl.index, tbl.values.tolist())], ["재무비율"] + cols), unsafe_allow_html=True)

__all__ = [n for n in dir() if not n.startswith("__")]  # 밑줄로 시작하는 도우미도 다른 모듈에서 쓸 수 있게
