"""옵션 체인 화면 (콜·풋 표, 포지션에 추가)."""
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


def _f(x, nd=2, pct=False):
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x * 100:.{nd}f}%" if pct else f"{x:,.{nd}f}"


def _tip(row, ticker, T) -> str:
    kind = TYPE_KO[bool(row["is_call"])]
    lines = [
        f"<b>{html.escape(ticker)} {kind} {row['strike']:g}</b> <span class='mut'>{html.escape(str(row.get('contractSymbol', '')))}</span>",
        f"시장가 {_f(row['mkt'])} <span class='mut'>({row['mkt_src']} · 매수 {_f(row['bid'])} / 매도 {_f(row['ask'])})</span>",
        f"이론가 {_f(row['theo'])} <span class='mut'>· 괴리 {_f(row['diff'])}</span>",
        f"내재변동성 {_f(row['iv'], 1, True)} <span class='mut'>(Yahoo {_f(row['impliedVolatility'], 1, True)})</span>",
        "<hr>",
        f"<table class='g'><tr><td>Δ 델타</td><td>{row['delta']:.4f}</td><td>Γ 감마</td><td>{row['gamma']:.5f}</td></tr>"
        f"<tr><td>ν 베가</td><td>{row['vega']:.4f}</td><td>Θ 세타</td><td>{row['theta']:.4f}</td></tr>"
        f"<tr><td>ρ 로</td><td>{row['rho']:.4f}</td><td>σ 기준</td><td>{row['sig_g'] * 100:.1f}%</td></tr></table>",
        f"<span class='mut'>1주 기준 · 베가/로 = 1%p당, 세타 = 1일당 · 1계약 = {MULT}주 · 잔존 {T * 365:.1f}일</span>",
    ]
    return "<br>".join(lines)


def chain_html(d: pd.DataFrame, ticker: str, T: float, S: float, dark: bool) -> str:
    calls = d[d["is_call"]].set_index("strike", drop=False)
    puts = d[~d["is_call"]].set_index("strike", drop=False)
    strikes = sorted(set(calls.index).union(puts.index))
    atm = min(strikes, key=lambda k: abs(k - S)) if strikes else None

    def side(df, k, order):
        if k not in df.index:
            return "".join("<td class='na'>-</td>" for _ in order)
        row = df.loc[k]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        cls = " itm" if row["itm"] else ""
        cells = []
        for col in order:
            if col == "mkt":
                tip = html.escape(_tip(row, ticker, T), quote=True)
                cells.append(f"<td class='px{cls}' data-g=\"{tip}\">{_f(row['mkt'])}</td>")
            elif col == "theo":
                cells.append(f"<td class='{cls}'>{_f(row['theo'])}</td>")
            elif col == "iv":
                cells.append(f"<td class='{cls}'>{_f(row['iv'], 1, True)}</td>")
            elif col == "delta":
                cells.append(f"<td class='{cls}'>{row['delta']:.3f}</td>")
            elif col in ("volume", "openInterest"):
                v = row[col]
                cells.append(f"<td class='{cls} mut'>{'-' if not np.isfinite(v) else f'{v:,.0f}'}</td>")
        return "".join(cells)

    left = ["openInterest", "volume", "delta", "iv", "theo", "mkt"]
    right = ["mkt", "theo", "iv", "delta", "volume", "openInterest"]
    head_l = ["미결제", "거래량", "델타", "IV", "이론가", "시장가"]
    rows = []
    for k in strikes:
        a = " atm" if k == atm else ""
        rows.append(f"<tr class='{a}'>{side(calls, k, left)}<td class='k'>{k:g}</td>{side(puts, k, right)}</tr>")

    fg, bg, line, mut, itm, atmc, tipbg = (
        ("#E6E6E6", "#0E1117", "#2B2F3A", "#9AA0A6", "rgba(66,133,244,0.16)", "rgba(255,196,0,0.18)", "#1C1F26")
        if dark else
        ("#1F2328", "#FFFFFF", "#E1E4E8", "#6A737D", "rgba(66,133,244,0.12)", "rgba(255,196,0,0.25)", "#FFFFFF"))
    return f"""
<style>
body{{margin:0;font-family:"Source Sans Pro",sans-serif;color:{fg};background:transparent;font-size:12.5px}}
.wrap{{max-height:540px;overflow:auto;border:1px solid {line};border-radius:6px}}
table.c{{border-collapse:collapse;width:100%;}}
table.c th{{position:sticky;top:0;background:{bg};z-index:2;font-weight:600;padding:5px 6px;border-bottom:1px solid {line};white-space:nowrap}}
table.c td{{padding:3px 6px;text-align:right;border-bottom:1px solid {line};white-space:nowrap}}
td.k{{text-align:center;font-weight:700;background:{line}}}
td.itm{{background:{itm}}}
tr.atm td.k{{background:{atmc}}}
td.px{{font-weight:700;cursor:help;text-decoration:underline dotted}}
.mut{{color:{mut}}}
td.na{{color:{mut}}}
th.grp{{text-align:center}}
#tip{{position:fixed;display:none;z-index:99;background:{tipbg};color:{fg};border:1px solid {line};border-radius:6px;
padding:7px 9px;font-size:11.5px;line-height:1.45;box-shadow:0 4px 14px rgba(0,0,0,.35);max-width:330px;pointer-events:none}}
#tip hr{{border:0;border-top:1px solid {line};margin:4px 0}}
#tip table.g td{{padding:0 8px 0 0;border:0;text-align:left}}
#tip table.g td:nth-child(even){{text-align:right;font-variant-numeric:tabular-nums;padding-right:12px}}
</style>
<div class="wrap"><table class="c">
<thead><tr><th class="grp" colspan="6">콜 (CALL)</th><th></th><th class="grp" colspan="6">풋 (PUT)</th></tr>
<tr>{''.join(f'<th>{h}</th>' for h in head_l)}<th>행사가</th>{''.join(f'<th>{h}</th>' for h in reversed(head_l))}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<div id="tip"></div>
<script>
const tip=document.getElementById('tip');
function pos(e){{const w=tip.offsetWidth,h=tip.offsetHeight;let x=e.clientX+14,y=e.clientY+14;
 if(x+w>innerWidth-4)x=e.clientX-w-14; if(y+h>innerHeight-4)y=e.clientY-h-14;
 tip.style.left=Math.max(4,x)+'px';tip.style.top=Math.max(4,y)+'px';}}
document.querySelectorAll('td[data-g]').forEach(td=>{{
 td.addEventListener('mouseenter',e=>{{tip.innerHTML=td.dataset.g;tip.style.display='block';pos(e);}});
 td.addEventListener('mousemove',pos);
 td.addEventListener('mouseleave',()=>{{tip.style.display='none';}});
 td.addEventListener('click',e=>{{tip.innerHTML=td.dataset.g;tip.style.display='block';pos(e);}});
}});
document.addEventListener('scroll',()=>{{tip.style.display='none';}},true);
</script>"""


def _pf_us_tickers() -> list:
    res = st.session_state.get("pf_opt_result") or {}
    return [it["code"] for it in res.get("items", []) if is_us_ticker(it["code"]) and not it.get("domestic")]


def _set_ticker(tk):
    st.session_state["opt_ticker"] = tk


def _add_position(ticker, expiry):
    base = st.session_state.get("hedge_opts_cur")
    if base is None:
        base = _empty_opts()
    kind = st.session_state.get("opt_add_kind", "콜")
    strike = st.session_state.get(f"opt_add_strike_{ticker}_{expiry}")
    qty = st.session_state.get("opt_add_qty", 1)
    if strike is None or not qty:
        return
    row = pd.DataFrame([{"기초자산": ticker, "만기": expiry, "종류": kind, "행사가": float(strike), "수량(계약)": int(qty)}])
    st.session_state["hedge_opts_cur"] = pd.concat([base, row], ignore_index=True)
    st.session_state["hedge_opts_ver"] = st.session_state.get("hedge_opts_ver", 0) + 1
    st.session_state["opt_add_msg"] = f"{ticker} {expiry} {kind} {float(strike):g} × {int(qty)}계약을 포지션에 추가했습니다."


def render_chain() -> None:
    pf = _pf_us_tickers()
    presets = list(dict.fromkeys(pf + OPT_PRESETS))[:10]
    if hasattr(st, "pills"):
        st.pills("빠른 선택", presets, key="opt_pill", on_change=lambda: _set_ticker(st.session_state.get("opt_pill")) if st.session_state.get("opt_pill") else None)
    else:
        cols = st.columns(len(presets))
        for c, tk in zip(cols, presets):
            c.button(tk, key=f"opt_preset_{tk}", use_container_width=True, on_click=_set_ticker, args=(tk,))
    if pf:
        st.caption("앞쪽 버튼은 '포트폴리오' 페이지에 들어 있는 미국 종목입니다.")
    raw_t = st.text_input("미국 티커 (예: AAPL, TSLA, SPY)", key="opt_ticker", placeholder="AAPL")
    ticker = (raw_t or "").strip().upper()
    if not ticker:
        st.info("티커를 입력하거나 위 버튼을 누르세요.")
        return
    if not is_us_ticker(ticker):
        st.warning("국내 종목·선물은 옵션 가격 조회를 지원하지 않습니다 (무료로 받을 수 있는 데이터가 없음). 미국 티커를 입력해 주세요.")
        return

    try:
        exps = load_expiries(ticker)
    except Exception as e:  # noqa: BLE001
        st.error(f"옵션 만기 목록을 불러오지 못했습니다: {e}. 잠시 후 다시 시도해 주세요.")
        return
    if not exps:
        st.warning(f"{ticker}에 상장된 옵션이 없거나 Yahoo에서 조회되지 않았습니다.")
        return

    P = market_params(ticker, "chain")
    if P is None:
        return

    c1, c2, c3 = st.columns([2, 1.4, 2.6])
    today = pd.Timestamp.now().normalize()
    first_i = next((i for i, e in enumerate(exps) if (pd.Timestamp(e) - today).days >= 7), 0)
    expiry = c1.selectbox("만기일", exps, index=first_i, key=f"opt_expiry_{ticker}",
                          format_func=lambda x: f"{x} (D-{(pd.Timestamp(x) - today).days})")
    basis = c2.radio("그릭스 기준 변동성", ["IV", "HV"], horizontal=True, key="opt_greek_basis",
                     format_func=lambda x: "내재(IV)" if x == "IV" else "역사적(HV)")
    try:
        raw = load_chain_raw(ticker, expiry)
    except Exception as e:  # noqa: BLE001
        st.error(f"옵션 가격을 불러오지 못했습니다: {e}. 잠시 후 다시 시도해 주세요.")
        return
    T = years_to_expiry(expiry)
    d = build_chain(raw, P, T, basis)
    if d.empty:
        st.warning("해당 만기의 옵션 가격이 없습니다.")
        return

    lo, hi = float(d["strike"].min()), float(d["strike"].max())
    S = P["S"]
    d_lo, d_hi = max(lo, S * 0.85), min(hi, S * 1.15)
    if d_lo >= d_hi:
        d_lo, d_hi = lo, hi
    rng = c3.slider("행사가 범위", min_value=lo, max_value=hi, value=(d_lo, d_hi), key=f"opt_rng_{ticker}_{expiry}")
    v = d[d["strike"].between(*rng)]

    st.caption("시장가에 마우스를 올리면(모바일은 탭) 그릭스가 뜹니다. 파란 칸 = 내가격, 노란 행사가 = 등가격.")
    components.html(chain_html(v, ticker, T, S, _is_dark()), height=560, scrolling=False)

    with st.expander("전체 수치 표 · CSV 내려받기"):
        show = v[["is_call", "contractSymbol", "strike", "bid", "ask", "lastPrice", "mkt", "theo", "diff", "iv",
                  "impliedVolatility", "delta", "gamma", "vega", "theta", "rho", "volume", "openInterest"]].copy()
        show["is_call"] = show["is_call"].map(TYPE_KO)
        show["iv"] *= 100
        show["impliedVolatility"] *= 100
        show.columns = ["종류", "계약코드", "행사가", "매수호가", "매도호가", "최근가", "시장가", "이론가(BSM)", "괴리",
                        "IV 계산(%)", "IV Yahoo(%)", "델타", "감마", "베가(1%p)", "세타(1일)", "로(1%p)", "거래량", "미결제약정"]
        st.dataframe(show.round(4), use_container_width=True, hide_index=True, height=420)
        st.download_button("CSV 내려받기", show.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{ticker}_{expiry}_chain.csv", mime="text/csv", key="opt_dl")

    # ---- 포지션 추가 (헷지 탭에서 사용)
    st.markdown("**이 만기의 옵션을 헷지 계산용 포지션에 추가**")
    a1, a2, a3, a4 = st.columns([1, 1.6, 1.2, 1.2])
    a1.selectbox("종류", ["콜", "풋"], key="opt_add_kind")
    strikes = sorted(d["strike"].unique().tolist())
    atm_i = int(np.argmin([abs(k - S) for k in strikes]))
    a2.selectbox("행사가", strikes, index=atm_i, key=f"opt_add_strike_{ticker}_{expiry}", format_func=lambda k: f"{k:g}")
    a3.number_input("수량(계약, 매도는 −)", value=-1, step=1, key="opt_add_qty")
    a4.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    a4.button("포지션에 추가", key="opt_add_btn", on_click=_add_position, args=(ticker, expiry), use_container_width=True)
    msg = st.session_state.pop("opt_add_msg", None)
    if msg:
        st.success(msg + " '델타·감마 헷지' 탭에서 확인하세요.")


# ======================================================================
# 2) 델타·감마 헷지 화면
# ======================================================================


__all__ = [n for n in dir() if not n.startswith("__")]  # 밑줄로 시작하는 도우미도 다른 모듈에서 쓸 수 있게
