"""델타·감마 헷지 화면: 보유 포지션 입력, 자동 헷지 계산, 헷지 전·후 손익."""
import html
import io
import math
import re
import time
import urllib.request
from datetime import datetime, timedelta

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import streamlit as st

import ui
import streamlit.components.v1 as components

import bsm

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None

try:
    from zoneinfo import ZoneInfo

    NY = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001
    NY = None

from options_data import *  # noqa: F401,F403
from options_chain import *  # noqa: F401,F403


def _load_from_pf(amount_krw: float, fx: float) -> None:
    res = st.session_state.get("pf_opt_result") or {}
    rows = []
    for it, w in zip(res.get("items", []), res.get("weights", [])):
        px = res.get("prices", {}).get(it["name"])
        if not px or w <= 1e-6:
            continue
        usd = not it.get("domestic")
        value = amount_krw * w
        shares = value / (px * fx) if usd else value / px
        note = f"비중 {w * 100:.1f}% · {'달러' if usd else '원'} 가격 {px:,.2f}" + (" · 환율로 원화 환산" if usd else "")
        if usd and not is_us_ticker(it["code"]):
            note += " · 선물은 1단위 기준(계약승수 미반영)"
        rows.append({"종목코드": it["code"], "종목명": it["name"], "보유 주식 수": round(shares, 2), "비고": note})
    st.session_state["hedge_stocks_cur"] = pd.DataFrame(rows) if rows else _empty_stocks()
    st.session_state["hedge_stocks_ver"] = st.session_state.get("hedge_stocks_ver", 0) + 1


def _row_for(d: pd.DataFrame, is_call: bool, strike: float):
    s = d[(d["is_call"] == is_call) & (np.isclose(d["strike"], strike))]
    return None if s.empty else s.iloc[0]


def _default_exp(exps: list) -> str:
    today = pd.Timestamp.now().normalize()
    good = [e for e in exps if (pd.Timestamp(e) - today).days >= 30] or exps
    return good[0]


def _hsel(key: str, exps: list, default_kind: str):
    """헷지 수단 옵션의 (만기, 콜여부) — 펼친 상자에서 고른 값 → 저장된 값 → 기본값(30일 이상 남은 가장 가까운 만기)."""
    ss = st.session_state
    store = ss.setdefault("_hedge_sel", {}).setdefault(key, {})
    exp = ss.get(f"{key}_exp", store.get("exp"))
    if exp not in exps:
        exp = _default_exp(exps)
    kind = ss.get(f"{key}_kind", store.get("kind", default_kind))
    store.update(exp=exp, kind=kind)
    return exp, kind == "콜", store


def _hstrike(key: str, d: pd.DataFrame, is_call: bool, exp: str, S: float, mny: float, store: dict):
    ks = sorted(d["strike"].unique().tolist())
    wk = f"{key}_k_{exp}_{is_call}"
    k = st.session_state.get(wk, store.get("k") if store.get("k_for") == (exp, is_call) else None)
    if k not in ks:
        k = ks[int(np.argmin([abs(x - S * mny) for x in ks]))]
    store.update(k=k, k_for=(exp, is_call))
    return k, ks


def render_hedge() -> None:
    st.caption("보유 주식·옵션의 델타(주가 1달러 변화에 대한 손익 변화)와 감마(델타의 변화)를 0으로 만드는 헷지 거래를 자동으로 계산합니다. "
               "헷지 수단(옵션 만기·행사가)과 계산 방식은 각 종목의 '세부 계산'에서 바꿀 수 있습니다.")
    st.session_state["hedge_trades"] = {}
    stocks = st.session_state.get("hedge_stocks_cur")
    opts = st.session_state.get("hedge_opts_cur")
    stocks = _empty_stocks() if stocks is None else stocks
    opts = _empty_opts() if opts is None else opts
    n_s = int(stocks["종목코드"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan}).notna().sum()) if len(stocks) else 0
    n_o = int(len(opts.dropna(how="all"))) if len(opts) else 0

    # ---- 포지션 입력 (입력한 뒤에는 접어 둠)
    exp_box, op = ui.more(f"보유 포지션 입력·수정 (주식 {n_s}종목 · 옵션 {n_o}건)", key="hedge_more_pos", expanded=(n_s + n_o == 0))
    if op:
        with exp_box:
            res = st.session_state.get("pf_opt_result")
            fxv = load_usdkrw()
            c1, c2, c3 = st.columns([1.6, 1.2, 1.2])
            amount = c1.number_input("투자 금액 (원)", value=100_000_000, step=10_000_000, min_value=0, key="hedge_amount")
            fx = c2.number_input("원/달러 환율", value=round(fxv[0], 2) if fxv else 1400.0, step=1.0, key="hedge_fx",
                                 help=f"{fxv[1]} 기준" if fxv else "환율 조회 실패로 기본값")
            c3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            c3.button("최적 포트폴리오 불러오기", key="hedge_load_pf", use_container_width=True, disabled=not res,
                      on_click=_load_from_pf, args=(amount, fx))
            st.caption("'포트폴리오' 페이지의 최적 비중 × 투자 금액 ÷ 최근 종가로 주식 수를 채웁니다." if res else
                       "'포트폴리오' 페이지에서 계산하면 위 버튼이 켜집니다. 아래 표에 직접 입력해도 됩니다 (예: AAPL, 100).")
            st.markdown("**보유 주식**")
            stocks = st.data_editor(
                stocks, key=f"hedge_stocks_ed_{st.session_state.get('hedge_stocks_ver', 0)}", num_rows="dynamic",
                use_container_width=True, hide_index=True,
                column_config={"보유 주식 수": st.column_config.NumberColumn(format="%.2f"),
                               "비고": st.column_config.TextColumn(disabled=True)})
            st.session_state["hedge_stocks_cur"] = stocks
            st.markdown("**보유 옵션** (1계약 = 100주, 매수 +, 매도 −)")
            opts = st.data_editor(
                opts, key=f"hedge_opts_ed_{st.session_state.get('hedge_opts_ver', 0)}", num_rows="dynamic",
                use_container_width=True, hide_index=True,
                column_config={"종류": st.column_config.SelectboxColumn(options=["콜", "풋"]),
                               "만기": st.column_config.TextColumn(help="YYYY-MM-DD"),
                               "수량(계약)": st.column_config.NumberColumn(step=1, help="매수 +, 매도 −"),
                               "행사가": st.column_config.NumberColumn(format="%.2f")})
            st.session_state["hedge_opts_cur"] = opts
            st.caption("'옵션 체인' 탭의 '포지션에 추가'로 넣을 수도 있습니다.")

    # ---- 기초자산 목록
    under = {}
    for r in stocks.to_dict("records"):
        code = str(r.get("종목코드") or "").strip().upper()
        sh = pd.to_numeric(r.get("보유 주식 수"), errors="coerce")
        if not code or code == "NAN" or not np.isfinite(sh):
            continue
        nm = r.get("종목명")
        under.setdefault(code, {"shares": 0.0, "opts": [], "name": str(nm) if isinstance(nm, str) and nm else code})
        under[code]["shares"] += float(sh)
    for r in opts.to_dict("records"):
        code = str(r.get("기초자산") or "").strip().upper()
        k = pd.to_numeric(r.get("행사가"), errors="coerce")
        n = pd.to_numeric(r.get("수량(계약)"), errors="coerce")
        if not code or code == "NAN" or r.get("종류") not in ("콜", "풋") or not np.isfinite(k) or not np.isfinite(n) or not r.get("만기"):
            continue
        under.setdefault(code, {"shares": 0.0, "opts": [], "name": code})
        under[code]["opts"].append({"exp": str(r["만기"])[:10], "is_call": r["종류"] == "콜", "K": float(k), "n": float(n)})
    if not under:
        st.info("위 '보유 포지션 입력·수정'을 펼쳐 보유 주식이나 옵션을 넣으면 헷지를 자동으로 계산합니다.")
        return

    summary, kr_only = [], []
    for code, info in under.items():
        if not is_us_ticker(code):
            kr_only.append(f"{info['name']}({code}) {info['shares']:,.0f}주")
            continue
        with st.container(border=True):
            out = _hedge_one(code, info)
            if out:
                summary.append(out)
    if kr_only:
        st.caption("국내 종목은 옵션 데이터가 없어 옵션 헷지를 계산하지 않았습니다 (델타 = 보유 주식 수, 감마 = 0): " + ", ".join(kr_only))
    if len(summary) > 1:
        exp_s, op_s = ui.more("종목별 요약 표", key="hedge_more_sum")
        if op_s:
            with exp_s:
                st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


def _hedge_pnl_curve(code: str, info: dict, trades: list, rng: float = 0.2):
    """오늘 기준으로 주가만 바뀔 때의 손익: 헷지 전 / 헷지 후. (가격 배열, 헷지 전, 헷지 후, 현재가)"""
    from capm_opt import pnl_at, price_positions

    base = {"shares": info["shares"], "opts": [dict(o, src="보유") for o in info["opts"]]}
    after = {"shares": info["shares"] + sum(t["n"] for t in trades if t["kind"] == "stock"),
             "opts": base["opts"] + [{"exp": t["exp"], "is_call": t["is_call"], "K": t["K"], "n": t["n"], "src": "헷지"}
                                     for t in trades if t["kind"] == "option" and t["n"]]}
    book, _w = price_positions({code: base})
    book2, _w2 = price_positions({code: after})
    if code not in book or code not in book2:
        return None
    S0 = book[code]["P"]["S"]
    xs = np.linspace(S0 * (1 - rng), S0 * (1 + rng), 161)
    return xs, pnl_at(book[code], xs, 0.0), pnl_at(book2[code], xs, 0.0), S0


def _hedge_one(code: str, info: dict):
    try:
        exps = load_expiries(code)
    except Exception as e:  # noqa: BLE001
        st.error(f"{code} 옵션 만기를 불러오지 못했습니다: {e}")
        return None
    if not exps:
        st.warning(f"{code}는 상장 옵션이 없어 옵션 헷지를 계산할 수 없습니다.")
        return None
    P = market_params(code, f"hedge_{code}", show=False)
    if P is None:
        return None
    S = P["S"]
    chains = {}

    def chain(exp):
        if exp not in chains:
            chains[exp] = build_chain(load_chain_raw(code, exp), P, years_to_expiry(exp), "IV")
        return chains[exp]

    # ---- 현재 포지션 그릭스
    rows = [{"구분": "주식", "내용": f"{info['shares']:,.2f}주", "단가($)": S, "델타(주)": info["shares"], "감마": 0.0,
             "베가($/1%p)": 0.0, "세타($/일)": 0.0}]
    tot = {"delta": info["shares"], "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    warns = []
    for o in info["opts"]:
        label = f"{o['exp']} {TYPE_KO[o['is_call']]} {o['K']:g} × {o['n']:+g}계약"
        if o["exp"] not in exps:
            warns.append(f"{label}: 상장된 만기가 아니라 제외")
            continue
        try:
            row = _row_for(chain(o["exp"]), o["is_call"], o["K"])
        except Exception as e:  # noqa: BLE001
            warns.append(f"{label}: 가격 조회 실패 ({e})")
            continue
        if row is None:
            warns.append(f"{label}: 해당 행사가 없음")
            continue
        m = o["n"] * MULT
        g = {k: float(row[k]) * m for k in ("delta", "gamma", "vega", "theta")}
        for k in tot:
            tot[k] += g[k]
        rows.append({"구분": "옵션", "내용": label, "단가($)": float(row["mkt"]), "델타(주)": g["delta"], "감마": g["gamma"],
                     "베가($/1%p)": g["vega"], "세타($/일)": g["theta"]})
    D, G = tot["delta"], tot["gamma"]

    # ---- 계산 방식 (자동: 주식만 있으면 B, 옵션이 있으면 A)
    ss = st.session_state
    auto = "B" if abs(G) < 1e-9 else "A"
    if f"hm_{code}" in ss:  # 사용자가 '세부 계산'에서 직접 고른 경우만 기억
        ss.setdefault("_hedge_method", {})[code] = ss[f"hm_{code}"]
    method = ss.get("_hedge_method", {}).get(code, auto)
    rounding = bool(ss.get("hedge_round", True))

    def inst_for(key, default_kind, mny):
        exp, is_call, store = _hsel(key, exps, default_kind)
        d = chain(exp)
        d = d[(d["is_call"] == is_call) & np.isfinite(d["gamma"]) & (d["gamma"] > 0) & np.isfinite(d["mkt"])]
        if d.empty:
            return None, None
        k, ks = _hstrike(key, d, is_call, exp, S, mny, store)
        r = _row_for(d, is_call, k)
        return ({"label": f"{exp} {TYPE_KO[is_call]} {k:g}", "exp": exp, "is_call": is_call, "K": float(k),
                 "d": float(r["delta"]) * MULT, "g": float(r["gamma"]) * MULT, "v": float(r["vega"]) * MULT,
                 "t": float(r["theta"]) * MULT, "px": float(r["mkt"]) * MULT}, ks)

    try:
        if method == "A":
            h1, ks1 = inst_for(f"hA_{code}", "콜", 1.0)
            spec = [(f"hA_{code}", "감마 헷지 옵션", ks1)]
            inst = None if h1 is None else [h1, {"label": "기초자산(주식)", "d": 1.0, "g": 0.0, "v": 0.0, "t": 0.0, "px": S}]
        else:
            h1, ks1 = inst_for(f"hB1_{code}", "풋", 1.0)
            h2, ks2 = inst_for(f"hB2_{code}", "콜", 1.05)
            spec = [(f"hB1_{code}", "옵션 ①", ks1), (f"hB2_{code}", "옵션 ②", ks2)]
            inst = None if h1 is None or h2 is None else [h1, h2]
    except Exception as e:  # noqa: BLE001
        st.error(f"{code} 옵션 가격을 불러오지 못했습니다: {e}")
        return None

    sol = None if inst is None else bsm.solve_hedge_two(D, G, inst[0]["d"], inst[0]["g"], inst[1]["d"], inst[1]["g"])
    after, cost, trades, hrows = dict(tot), 0.0, [], []
    n_used = []
    if sol is not None:
        n_exact = list(sol)
        n_used = [round(x) for x in n_exact] if rounding else n_exact
        for ins, ne, nu in zip(inst, n_exact, n_used):
            unit = "주" if ins["label"].startswith("기초자산") else "계약"
            hrows.append({"헷지 수단": ins["label"], "계산 수량": f"{ne:+,.3f}{unit}",
                          "적용 수량": f"{nu:+,.0f}{unit}" if rounding else f"{nu:+,.3f}{unit}",
                          "매매": "매수" if nu > 0 else ("매도" if nu < 0 else "-"),
                          "단가($)": ins["px"] / (MULT if unit == "계약" else 1), "거래 금액($)": nu * ins["px"],
                          "단위 델타": ins["d"], "단위 감마": ins["g"]})
            for k_, kk in (("delta", "d"), ("gamma", "g"), ("vega", "v"), ("theta", "t")):
                after[k_] += nu * ins[kk]
            cost += nu * ins["px"]
            trades.append({"kind": "stock", "n": float(nu)} if unit == "주" else
                          {"kind": "option", "exp": ins["exp"], "is_call": ins["is_call"], "K": ins["K"], "n": float(nu)})
        ss.setdefault("hedge_trades", {})[code] = trades

    # ---- 핵심 수치
    st.markdown(f"#### {info['name']} ({code}) <span style='font-size:0.85rem;opacity:.65;font-weight:400'>현재가 ${S:,.2f}</span>"
                .replace("$", "&#36;"), unsafe_allow_html=True)
    ui.cards([("헷지 전 델타", f"{D:,.1f}주", f"주가 1% 변동 → 약 ${D * S * 0.01:,.0f}"),
              ("헷지 전 감마", f"{G:,.3f}", "주가 1달러 변동 시 델타 변화"),
              ("헷지 후 델타", f"{after['delta']:,.2f}주" if sol is not None else "-", "0에 가까울수록 주가 방향에 중립"),
              ("헷지 후 감마", f"{after['gamma']:,.4f}" if sol is not None else "-", "0에 가까울수록 델타가 안정")], cols=4)
    if sol is None:
        st.warning("헷지 수량을 계산하지 못했습니다. '세부 계산'에서 다른 만기·행사가를 골라 보세요.")
    else:
        todo = []
        for ins, nu in zip(inst, n_used):
            if abs(nu) < 1e-9:
                continue
            what = "주식" if ins["label"].startswith("기초자산") else ins["label"]
            unit = "주" if what == "주식" else "계약"
            todo.append(f"**{what} {abs(nu):,.0f}{unit} {'매수' if nu > 0 else '매도'}**" if rounding else
                        f"**{what} {abs(nu):,.2f}{unit} {'매수' if nu > 0 else '매도'}**")
        net = "지급" if cost > 0 else "수취"
        st.markdown(("해야 할 거래: " + " · ".join(todo) if todo else "추가 거래가 필요 없습니다.")
                    + f" &nbsp;→ 순 {net} ${abs(cost):,.0f}".replace("$", "&#36;"), unsafe_allow_html=True)
        try:
            cur = _hedge_pnl_curve(code, info, trades)
        except Exception:  # noqa: BLE001
            cur = None
        if cur is not None:
            import plotly.graph_objects as go

            xs, y0, y1, S0 = cur
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=(xs / S0 - 1) * 100, y=y0, name="헷지 전", mode="lines", line=dict(color="#9AA0A6", width=2, dash="dot"),
                                     customdata=xs, hovertemplate="주가 %{customdata:,.2f} (%{x:+.1f}%)<br>손익 %{y:$,.0f}<extra>헷지 전</extra>"))
            fig.add_trace(go.Scatter(x=(xs / S0 - 1) * 100, y=y1, name="헷지 후", mode="lines", line=dict(color="#E5534B", width=2.5),
                                     customdata=xs, hovertemplate="주가 %{customdata:,.2f} (%{x:+.1f}%)<br>손익 %{y:$,.0f}<extra>헷지 후</extra>"))
            fig.add_hline(y=0, line_color="#888", line_width=1)
            fig.add_vline(x=0, line_dash="dot", line_color="#888")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), font=dict(size=11), hovermode="x unified",
                              xaxis_title="지금 주가가 갑자기 이만큼 움직이면 (%)", yaxis_title="내 포지션 손익($)", xaxis_ticksuffix="%",
                              legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("헷지 전(점선)은 주가가 움직인 만큼 손익이 커지지만, 헷지 후(빨간 선)는 현재가 근처에서 평평해집니다 — "
                       "작은 주가 변동에는 손익이 거의 없다는 뜻입니다. 옵션은 지금의 내재변동성으로 다시 계산했습니다.")
    for w in warns:
        st.caption("· " + w)

    # ---- 세부 계산 (더보기)
    exp_d, op_d = ui.more("세부 계산 · 헷지 수단 바꾸기 (포지션별 그릭스, 만기·행사가, 시장 환경)", key=f"hedge_more_{code}")
    if op_d:
        with exp_d:
            st.radio("계산 방식", ["A", "B"], index=["A", "B"].index(method), key=f"hm_{code}", horizontal=False, format_func=lambda m: {
                "A": "A. 옵션 1종으로 감마 → 주식 매매로 델타 (교과서 방식)",
                "B": "B. 옵션 2종으로 델타·감마 동시 (주식은 그대로 보유)"}[m])
            st.checkbox("옵션은 정수 계약, 주식은 정수 주로 반올림", value=rounding, key="hedge_round")
            today = pd.Timestamp.now().normalize()
            for key, lab, ks in spec:
                store = ss["_hedge_sel"][key]
                c1, c2, c3 = st.columns([1.6, 0.9, 1.3])
                c1.selectbox(f"{lab} 만기", exps, index=exps.index(store["exp"]), key=f"{key}_exp",
                             format_func=lambda x: f"{x} (D-{(pd.Timestamp(x) - today).days})")
                c2.selectbox(f"{lab} 종류", ["콜", "풋"], index=0 if store["kind"] == "콜" else 1, key=f"{key}_kind")
                if ks:
                    c3.selectbox(f"{lab} 행사가", ks, index=ks.index(store["k"]) if store.get("k") in ks else 0,
                                 key=f"{key}_k_{store['exp']}_{store['kind'] == '콜'}", format_func=lambda x: f"{x:g}")
            if method == "A" and abs(G) < 1e-9:
                st.caption("주식만 있을 때 A 방식은 주식을 전부 파는 것과 같습니다. 주식을 유지하려면 B를 고르세요.")
            st.markdown("**현재 포지션 그릭스**")
            rows.append({"구분": "합계", "내용": "헷지 전", "단가($)": float("nan"), "델타(주)": D, "감마": G,
                         "베가($/1%p)": tot["vega"], "세타($/일)": tot["theta"]})
            st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)
            if hrows:
                st.markdown("**헷지 거래**")
                st.dataframe(pd.DataFrame(hrows).round(4), use_container_width=True, hide_index=True)
                cmp = pd.DataFrame({"헷지 전": [tot["delta"], tot["gamma"], tot["vega"], tot["theta"]],
                                    "헷지 후": [after["delta"], after["gamma"], after["vega"], after["theta"]]},
                                   index=["델타(주)", "감마(주/$)", "베가($/1%p)", "세타($/일)"])
                st.dataframe(cmp.round(5), use_container_width=True)
                st.caption("옵션 단가는 중간가 기준. 반올림하면 헷지 후 델타·감마가 정확히 0이 아닐 수 있습니다.")
            st.markdown("**시장 환경 (금리·배당·변동성)**")
            market_params(code, f"hedge_{code}", show=True)
    if sol is None:
        return {"기초자산": code, "헷지 전 델타(주)": round(D, 3), "헷지 전 감마": round(G, 5), "헷지 후 델타(주)": None,
                "헷지 후 감마": None, "헷지 비용($)": None, "비고": "계산 실패"}
    return {"기초자산": code, "헷지 전 델타(주)": round(D, 3), "헷지 전 감마": round(G, 5),
            "헷지 후 델타(주)": round(after["delta"], 3), "헷지 후 감마": round(after["gamma"], 5),
            "헷지 비용($)": round(cost, 2), "비고": " / ".join(f"{i['label']} {n:+,.2f}" for i, n in zip(inst, n_used))}


# ======================================================================
# 탭 진입점
# ======================================================================


__all__ = [n for n in dir() if not n.startswith("__")]  # 밑줄로 시작하는 도우미도 다른 모듈에서 쓸 수 있게
