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
    per = None  # PER 그래프는 'PER'(직전 사업연도 EPS)과 '12M PER'(향후 12개월 예상 EPS) 두 가지만
    hist = R.get("hist")
    d_avail = ["주가"] if isinstance(hist, pd.Series) and len(hist) else []
    ann = annual_per_series(hist, sec, R.get("splits"))
    if ann is not None:
        d_avail.append("PER (직전 사업연도 EPS)")
    fc = None
    try:
        fc = fwd_consensus_series(hist, R.get("edates"), R.get("eps_trend"), R.get("info") or {}, n_eps)
    except Exception:  # noqa: BLE001
        fc = None
    if fc is not None:
        d_avail.append("12M 선행 PER (컨센서스 추이)")
    dv, dv_err = None, None
    if isinstance(hist, pd.Series) and len(hist) and _VAL:
        try:
            from valuation import daily_valuation, load_quarterly
            if src == "SEC 10-K":
                dv = daily_valuation(load_quarterly(ticker), hist, R.get("splits"))
            elif sec is not None:  # 국내·SEC 미등록 종목: 연간 재무제표를 공시 시점부터 반영 (다음 연간 공시까지 유지)
                from kr import annual_quarterly_dict
                info = R.get("info") or {}
                dv = daily_valuation(annual_quarterly_dict(sec, _num(info.get("sharesOutstanding")), R.get("splits")), hist, R.get("splits"), stale=470)
            else:
                raise RuntimeError("재무제표 없음")
            d_avail += [k for k in _VAL if k in dv and dv[k].notna().sum() > 20]
        except Exception as e:  # noqa: BLE001
            dv, dv_err = None, str(e)

    # 일별로 계산되는 지표가 있으면 같은 이름의 연간 밸류에이션 비율(PER·PBR·PSR)은 빼고, 시가총액 기준 PER은
    # EPS 기준 12M 트레일링 PER과 겹치므로 EPS 기준이 있을 때는 뺀다.
    if ann is not None and "PER (시가총액 ÷ 순이익 TTM)" in d_avail:
        d_avail.remove("PER (시가총액 ÷ 순이익 TTM)")
    daily_names = set(d_avail) | ({"PER"} if per is not None else set())
    a_avail = [k for k in avail if k not in daily_names]
    val_daily = [k for k in d_avail if k != "주가"]
    options = (["주가"] if "주가" in d_avail else []) + val_daily + a_avail

    def _label(k):
        return _PER_LABEL.get(k, k)

    chosen = st.multiselect("그래프 항목", options,
                            default=(["주가"] if "주가" in options else [])
                            + [k for k in ("PER (직전 사업연도 EPS)", "PER (시가총액 ÷ 순이익 TTM)") if k in options][:1]
                            + [k for k in ("12M 선행 PER (컨센서스 추이)",) if k in options],
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

    # 위에서 고른 기간만 그린다 (재무비율은 그 기간 안의 회계연도)
    period = R.get("_period")
    start = pd.Timestamp(period[0]) if period else None
    end = pd.Timestamp(period[1]) + pd.Timedelta(days=1) if period else None
    if rt is not None and period:
        rt = rt[[c for c in rt.columns if start <= pd.Timestamp(c) <= end]] if any(start <= pd.Timestamp(c) <= end for c in rt.columns) else rt
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
        if start is not None:
            sr = sr[sr.index >= start]
        return sr[sr.index <= end] if end is not None else sr

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
            fig.add_trace(go.Scatter(x=sr.index, y=sr.values, name=_PER_TRACE.get(k, k), mode="lines", yaxis=ax, line=dict(color=color, width=1.3),
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
        elif k == "PER (직전 사업연도 EPS)":
            sr = _clip(ann).dropna()
            fig.add_trace(go.Scatter(x=sr.index, y=sr.values, name="PER", mode="lines", yaxis=ax, line=dict(color=color, width=1.4),
                                     hovertemplate="%{x|%Y-%m-%d} PER %{y:.1f}배<extra></extra>"))
            x_vals.append(sr.values)
        elif k == "12M 선행 PER (컨센서스 추이)":
            sr = _clip(fc[0]).dropna()
            fig.add_trace(go.Scatter(x=sr.index, y=sr.values, name="12M PER", mode="lines", yaxis=ax,
                                     line=dict(color=color, width=1.4),
                                     hovertemplate="%{x|%Y-%m-%d} 12M PER %{y:.1f}배<extra></extra>"))
            x_vals.append(sr.values)
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
        issues = data_check.run_checks(_clip(hist), sec, R.get("splits"), ed, ann=_clip(ann) if ann is not None else None, trail=None,
                                       ann_dates=None if tb is None else tb.index, trail_dates=None)
        data_check.render_checks(issues, key=ticker)
    except Exception as e:  # noqa: BLE001
        st.caption(f"데이터 점검을 실행하지 못했습니다 ({e}).")

    try:  # 고른 재무비율로 상승·하락 판단 (outlook.py)
        import outlook
        daily = {}
        for k in dsel:
            if k == "PER (직전 사업연도 EPS)" and ann is not None:
                daily["PER"] = (ann, "배")
            elif k == "12M 선행 PER (컨센서스 추이)" and fc is not None:
                daily["12M PER"] = (fc[0], "배")
            elif k in _VAL and dv is not None and k in dv:
                daily[_PER_TRACE.get(k, k)] = (dv[k], DAILY[k])
        outlook.render_outlook(sec, R, ticker, sel, daily, options=options, ms_key=f"rs_combo_{ticker}",
                               fwd=fc[0] if fc is not None else None)
    except Exception as e:  # noqa: BLE001
        st.caption(f"상승·하락 판단을 계산하지 못했습니다 ({e}).")

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

def _default_combo(options):
    return ((["주가"] if "주가" in options else [])
            + [k for k in ("PER (직전 사업연도 EPS)", "PER (시가총액 ÷ 순이익 TTM)") if k in options][:1]
            + [k for k in ("12M 선행 PER (컨센서스 추이)",) if k in options])


def render_score_gauges(sec, R: dict, ticker: str, n_eps, src: str = "SEC 10-K") -> None:
    """기업 리서치: 재무비율 그래프 섹션 바로 위의 원형 점수 3개 (CAPM 미포함 · 업종 평균 대비 · CAPM 대비)."""
    import outlook

    hist = R.get("hist")
    if sec is None or not isinstance(hist, pd.Series) or not len(hist):
        return
    domestic = "_ksec" in R
    info = R.get("info") or {}
    # 업종 유형 (아래 판단 섹션에서 바꾼 값이 있으면 그것)
    det = outlook.classify(info.get("sector"), info.get("industry"), R.get("_ksec"))
    if det == "general" and not R.get("_ksec") and not info.get("sector"):
        try:
            det = outlook.classify_sic(outlook.sec_sic(ticker)[0])
        except Exception:  # noqa: BLE001
            pass
    prof = st.session_state.get(f"rs_prof_{ticker}") if st.session_state.get(f"rs_prof_{ticker}") in outlook.PROFILES else det
    # 고른 지표 (아래 '그래프 항목'과 같은 값, 고르기 전이면 그 기본값)
    rt = compute_ratios(sec, hist, R.get("splits"))
    ann_ok = [k for k in RATIOS if k in rt.index and rt.loc[k].notna().sum() >= 2]
    chosen = st.session_state.get(f"rs_combo_{ticker}")
    if chosen is None:
        chosen = _default_combo(["주가", "PER (직전 사업연도 EPS)", "12M 선행 PER (컨센서스 추이)"])
    daily_keys = [k for k in chosen if k in ("PER (직전 사업연도 EPS)", "12M 선행 PER (컨센서스 추이)") or k in _VAL]
    if "PER (직전 사업연도 EPS)" in daily_keys and "PER (시가총액 ÷ 순이익 TTM)" in daily_keys:
        daily_keys.remove("PER (시가총액 ÷ 순이익 TTM)")
    sel = [k for k in chosen if k in ann_ok and k not in daily_keys and k not in _VAL]

    def build():
        out = {}
        if "PER (직전 사업연도 EPS)" in daily_keys:
            ann = annual_per_series(hist, sec, R.get("splits"))
            if ann is not None:
                out["PER"] = (ann, "배")
        if "12M 선행 PER (컨센서스 추이)" in daily_keys:
            try:
                fc = fwd_consensus_series(hist, R.get("edates"), R.get("eps_trend"), info, n_eps)
                if fc is not None:
                    out["12M PER"] = (fc[0], "배")
            except Exception:  # noqa: BLE001
                pass
        vk = [k for k in daily_keys if k in _VAL]
        if vk:
            try:
                from valuation import daily_valuation, load_quarterly
                if src == "SEC 10-K":
                    dv = daily_valuation(load_quarterly(ticker), hist, R.get("splits"))
                else:
                    from kr import annual_quarterly_dict
                    dv = daily_valuation(annual_quarterly_dict(sec, _num(info.get("sharesOutstanding")), R.get("splits")), hist,
                                         R.get("splits"), stale=470)
                for k in vk:
                    if k in dv:
                        out[_PER_TRACE.get(k, k)] = (dv[k], DAILY[k])
            except Exception:  # noqa: BLE001
                pass
        return out

    ss = st.session_state
    opts = dict(quarterly=ss.get(f"rs_o_q_{ticker}", not domestic) and not domestic, momentum=ss.get(f"rs_o_m_{ticker}", True),
                surprise=ss.get(f"rs_o_s_{ticker}", True), peers=False, flags=ss.get(f"rs_o_f_{ticker}", True),
                extra=ss.get(f"rs_o_x_{ticker}", True))
    st.markdown("##### 매수·매도 점수 (6개월)")
    st.markdown('<div style="font-size:11px;opacity:.6;margin-top:-8px">투자 권유 아님 · 과거 데이터로 계산한 모델 점수</div>',
                unsafe_allow_html=True)
    try:
        with st.spinner("과거 시점마다 다시 판정해 점수를 계산하는 중... (같은 업종 대표 종목 사례를 처음 모을 때는 1~2분 걸립니다)"):
            res = outlook.summary_scores(sec, R, ticker, sel, daily_keys, build, opts, prof, domestic)
    except Exception as e:  # noqa: BLE001
        st.caption(f"점수를 계산하지 못했습니다 ({e}).")
        return
    if not res:
        st.caption("고른 지표로는 점수를 낼 수 없습니다. 아래 그래프 항목에서 재무비율이나 PER을 골라 주세요.")
        return
    S = res["scores"]

    def sub(m, what):
        d = S[m][0]
        return f"과거 비슷한 시점 {d['n']}번 중 {d['k']}번 {what}"

    items = [("CAPM 미포함", S["abs"][0], S["abs"][1], sub("abs", "플러스")),
             ("업종 평균 대비", S["ind"][0], S["ind"][1], sub("ind", "업종보다 높음")),
             ("CAPM 대비", S["capm"][0], S["capm"][1], sub("capm", "CAPM 초과"))]
    st.markdown(outlook.gauges_html(items), unsafe_allow_html=True)
    names = {"PER (직전 사업연도 EPS)": "PER", "12M 선행 PER (컨센서스 추이)": "12M PER"}
    used = [names.get(k, _PER_TRACE.get(k, k)) for k in chosen if k != "주가"]
    extra = [x for x, on in (("모멘텀", opts["momentum"]), ("실적 서프라이즈", opts["surprise"]), ("추가 신호", opts["extra"])) if on]
    st.caption(f"판정 근거: {', '.join(used + extra) or '-'} · {outlook.PROFILES[prof]['name']} 기준 · 업종 평균 = {res['sname'] or '-'}"
               + (f" · 위험 신호: {', '.join(res['flags'])} (50점 상한)" if res["flags"] else "")
               + ". 점수 = 지금과 판정 점수가 비슷했던 과거 시점들 가운데 이후 6개월 수익률이 0 · 업종 평균 · CAPM 기대수익률을 넘은 비율 "
               "(겹치는 기간만큼 사례를 줄이고 50점 쪽으로 당김). 80% 구간이 20점보다 넓으면 범위로 표시합니다. "
               + (f"같은 업종 대표 종목 {res.get('pool_n', 0)}곳의 사례를 합쳐 계산. " if res["pooled"]
                  else "국내 종목은 같은 업종 대표 종목 목록이 없어 이 종목 사례만 씁니다. " if domestic else "")
               + "지표는 아래 그래프 항목에서 바꿀 수 있습니다.")


__all__ = [n for n in dir() if not n.startswith("__")]  # 밑줄로 시작하는 도우미도 다른 모듈에서 쓸 수 있게
