"""옵션 탭
1) 옵션 체인: 미국 상장 종목·ETF의 콜/풋 시장가, 블랙-숄즈-머튼 이론가, 내재변동성, 역사적 변동성, 그릭스(마우스 오버 팝업)
2) 델타·감마 헷지: 최적 포트폴리오(또는 직접 입력한 보유 주식) + 옵션 포지션의 그릭스를 구하고, 델타와 감마를 동시에 0으로 만드는 헷지 수량 계산
데이터: 옵션 = Yahoo Finance(yfinance, 약 15분 지연), 기초자산 가격 = FinanceDataReader, 무위험금리 = FRED 3개월 국채.
"""
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

MULT = 100  # 미국 주식옵션 1계약 = 100주
OPT_PRESETS = ["AAPL", "NVDA", "TSLA", "MSFT", "SPY", "QQQ"]
HV_WINDOWS = {"20일": 20, "60일": 60, "120일": 120, "1년(252일)": 252}
TYPE_KO = {True: "콜", False: "풋"}


# ======================================================================
# 데이터 불러오기 (느려도 되도록 넉넉히 재시도)
# ======================================================================
def _retry(fn, tries=5, base=2.0):
    err = None
    for i in range(tries):
        try:
            out = fn()
            if out is not None:
                return out
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(base * (i + 1))
    if err:
        raise err
    return None


def is_us_ticker(code: str) -> bool:
    code = str(code).strip().upper()
    return bool(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", code)) and "=" not in code and not code.startswith("^") \
        and code not in ("US500", "KS11", "KQ11", "DX-Y.NYB")


@st.cache_data(ttl=600, show_spinner="옵션 만기 목록을 불러오는 중... (Yahoo 응답이 느리면 몇 번 다시 시도합니다)")
def load_expiries(ticker: str) -> list:
    """빈 목록이 오면(일시적 차단일 수 있음) 다시 시도하고, 끝까지 비면 오류로 처리해 캐시에 남기지 않는다."""
    for i in range(5):
        try:
            exps = list(yf.Ticker(ticker).options or [])
            if exps:
                return exps
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0 * (i + 1))
    raise RuntimeError("만기 목록이 비어 있습니다 (옵션이 없는 종목이거나 Yahoo 응답 지연)")


@st.cache_data(ttl=300, show_spinner="옵션 가격을 불러오는 중... (시간이 걸릴 수 있습니다)")
def load_chain_raw(ticker: str, expiry: str) -> pd.DataFrame:
    ch = _retry(lambda: yf.Ticker(ticker).option_chain(expiry))
    c, p = ch.calls.copy(), ch.puts.copy()
    c["is_call"], p["is_call"] = True, False
    return pd.concat([c, p], ignore_index=True)


@st.cache_data(ttl=600, show_spinner="기초자산 가격을 불러오는 중...")
def load_underlying(ticker: str) -> dict:
    """현재가(최근 종가), 1년 이상 일별 종가, 최근 12개월 배당 합계."""
    close = None
    start = (datetime.now() - timedelta(days=800)).strftime("%Y-%m-%d")
    try:
        d = _retry(lambda: fdr.DataReader(ticker, start), tries=3)
        if d is not None and "Close" in d:
            close = d["Close"].dropna()
            close = close[close > 0]
    except Exception:  # noqa: BLE001
        close = None
    t = yf.Ticker(ticker)
    if close is None or len(close) < 30:
        h = _retry(lambda: t.history(period="2y", auto_adjust=False), tries=3)
        close = h["Close"].dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
    spot = float(close.iloc[-1])
    asof = str(pd.Timestamp(close.index[-1]).date())
    try:  # 장중이면 Yahoo의 최근 체결가가 더 최신
        h1 = t.history(period="1d", interval="1m")
        if h1 is not None and len(h1):
            spot = float(h1["Close"].dropna().iloc[-1])
            asof = pd.Timestamp(h1.index[-1]).strftime("%Y-%m-%d %H:%M (뉴욕)")
    except Exception:  # noqa: BLE001
        pass
    div12 = 0.0
    try:
        dv = t.dividends
        if dv is not None and len(dv):
            idx = pd.to_datetime(dv.index).tz_localize(None)
            dv = pd.Series(dv.values, index=idx)
            div12 = float(dv[dv.index >= pd.Timestamp.now() - pd.Timedelta(days=365)].sum())
    except Exception:  # noqa: BLE001
        pass
    return {"spot": spot, "asof": asof, "close": close, "div12": div12}


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_us_rate_3m() -> tuple:
    """FRED 3개월 미 국채(DGS3MO) 최근값. 실패하면 None."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
            data = pd.read_csv(io.StringIO(raw), na_values=["."]).dropna()
            return float(data.iloc[-1, 1]) / 100, str(data.iloc[-1, 0])
        except Exception:  # noqa: BLE001
            time.sleep(1.0)
    return None


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_usdkrw() -> tuple:
    try:
        d = fdr.DataReader("USD/KRW", (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d"))
        s = d["Close"].dropna()
        return float(s.iloc[-1]), str(s.index[-1].date())
    except Exception:  # noqa: BLE001
        return None


def years_to_expiry(expiry: str) -> float:
    """만기일 뉴욕 16:00까지 남은 기간(연, 달력일/365)."""
    exp = pd.Timestamp(expiry)
    if NY is not None:
        exp_dt = datetime(exp.year, exp.month, exp.day, 16, 0, tzinfo=NY)
        now = datetime.now(NY)
    else:
        exp_dt = datetime(exp.year, exp.month, exp.day, 21, 0)
        now = datetime.utcnow()
    return max((exp_dt - now).total_seconds() / (365.0 * 86400), 1.0 / (365 * 24))


# ======================================================================
# 시장 환경(기초자산·금리·배당·변동성)과 체인 계산
# ======================================================================
def market_params(ticker: str, key: str, show: bool = True) -> dict | None:
    """기초자산 정보 + 사용자가 고친 r, q, 이론가용 변동성. show=True면 화면에 입력칸과 지표를 그리고,
    False면 그리지 않고 (이전에 고친 값이 있으면 그 값, 없으면 기본값으로) 계산만 한다."""
    try:
        u = load_underlying(ticker)
    except Exception as e:  # noqa: BLE001
        st.error(f"{ticker} 기초자산 가격을 불러오지 못했습니다: {e}")
        return None
    hv = {k: bsm.hist_vol(u["close"], w) for k, w in HV_WINDOWS.items()}
    rr = load_us_rate_3m()
    r_default = rr[0] if rr else 0.04
    q_default = u["div12"] / u["spot"] if u["spot"] > 0 else 0.0
    if not show:
        ss = st.session_state
        r_pct = float(ss.get(f"{key}_r_{ticker}", round(r_default * 100, 3)))
        q_pct = float(ss.get(f"{key}_q_{ticker}", round(q_default * 100, 3)))
        hv_pick = ss.get(f"{key}_hvpick_{ticker}", list(HV_WINDOWS)[1])
        sig_theo = (float(ss.get(f"{key}_hvman_{ticker}", 30.0)) / 100 if hv_pick == "직접 입력" else hv.get(hv_pick, float("nan")))
        if not (sig_theo == sig_theo) or sig_theo <= 0:
            sig_theo = 0.3
        return {"S": u["spot"], "asof": u["asof"], "r": math.log1p(r_pct / 100), "r_simple": r_pct / 100,
                "q": q_pct / 100, "sig_theo": sig_theo, "hv": hv, "hv_pick": hv_pick}

    mgrid([("기초자산 가격", f"{u['spot']:,.2f}", f"기준: {u['asof']}")] +
          [(f"HV {k}", f"{v * 100:.1f}%" if v == v else "-", f"역사적 변동성 {k}") for k, v in hv.items()], cols=5)

    c1, c2, c3 = st.columns(3)
    r_pct = c1.number_input("무위험금리 r (%)", value=round(r_default * 100, 3), step=0.05, format="%.3f",
                            key=f"{key}_r_{ticker}",
                            help=(f"FRED 미 국채 3개월 {rr[0] * 100:.2f}% ({rr[1]} 기준)" if rr else "FRED 조회 실패로 4% 기본값") + ", 연속복리로 환산해 사용")
    q_pct = c2.number_input("배당수익률 q (%)", value=round(q_default * 100, 3), step=0.05, format="%.3f",
                            key=f"{key}_q_{ticker}", help=f"최근 12개월 배당 {u['div12']:.2f} ÷ 현재가")
    hv_opts = list(HV_WINDOWS) + ["직접 입력"]
    hv_pick = c3.selectbox("이론가에 쓸 변동성", hv_opts, index=1, key=f"{key}_hvpick_{ticker}")
    if hv_pick == "직접 입력":
        sig_theo = st.number_input("이론가 변동성 (%)", value=round((hv["60일"] if hv["60일"] == hv["60일"] else 0.3) * 100, 2),
                                   step=0.5, key=f"{key}_hvman_{ticker}") / 100
    else:
        sig_theo = hv[hv_pick]
    if not (sig_theo == sig_theo) or sig_theo <= 0:
        sig_theo = 0.3
    return {"S": u["spot"], "asof": u["asof"], "r": math.log1p(r_pct / 100), "r_simple": r_pct / 100,
            "q": q_pct / 100, "sig_theo": sig_theo, "hv": hv, "hv_pick": hv_pick}


def build_chain(raw: pd.DataFrame, P: dict, T: float, greek_basis: str = "IV") -> pd.DataFrame:
    d = raw.copy()
    for c in ("bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility", "strike"):
        if c not in d:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    bid, ask, last = d["bid"].fillna(0), d["ask"].fillna(0), d["lastPrice"]
    has_q = (bid > 0) & (ask > 0) & (ask >= bid)
    d["mid"] = np.where(has_q, (bid + ask) / 2, np.nan)
    d["mkt"] = np.where(has_q, d["mid"], last)
    d["mkt_src"] = np.where(has_q, "중간가", "최근체결가")
    S, r, q = P["S"], P["r"], P["q"]
    K, c = d["strike"].values, d["is_call"].values.astype(bool)
    d["iv"] = bsm.implied_vol(d["mkt"].values, S, K, T, r, q, c)
    d["theo"] = bsm.price(S, K, T, r, q, P["sig_theo"], c)
    d["diff"] = d["mkt"] - d["theo"]
    sig_g = np.where(np.isfinite(d["iv"]), d["iv"], P["sig_theo"]) if greek_basis == "IV" else np.full(len(d), P["sig_theo"])
    g = bsm.greeks(S, K, T, r, q, sig_g, c)
    for k, v in g.items():
        d[k] = v
    d["sig_g"] = sig_g
    d["itm"] = np.where(c, K < S, K > S)
    return d.sort_values(["is_call", "strike"], ascending=[False, True]).reset_index(drop=True)


# ======================================================================
# 체인 표 (HTML, 시장가에 마우스를 올리면 그릭스 팝업)
# ======================================================================
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


def _is_dark() -> bool:
    try:
        t = st.context.theme.type
        if t:
            return t == "dark"
    except Exception:  # noqa: BLE001
        pass
    return (st.get_option("theme.base") or "dark") != "light"


def mgrid(items, cols: int = 4) -> None:
    """지표를 가로 격자로 나열 (휴대폰에서도 세로로 길게 쌓이지 않게). items: (라벨, 값[, 보조 설명])
    라벨 옆 ⓘ(동그라미 ?)에 마우스를 올리거나 탭하면 값이 높을 때·낮을 때의 의미가 뜬다."""
    import re

    from glossary import CSS, icon, tip

    cells = ""
    for i, it in enumerate(items):
        lab, val = it[0], it[1]
        extra = it[2] if len(it) > 2 and it[2] else ""
        t = tip(lab)
        if extra and not re.search(r"[÷×=√]| [+−] |N\(", extra):  # 계산식이 아닌 보조 정보(기준일 등)만 덧붙임
            t = f"{t} ({extra})" if t else extra
        right = " qr" if cols > 1 and (i % cols) >= cols / 2 else ""
        cells += (f'<div class="kc-c{right}"><div class="kc-l"><span>{html.escape(lab)}</span>{icon(t)}</div>'
                  f'<div class="kc-v">{html.escape(val)}</div></div>')
    cells = cells.replace("$", "&#36;")  # 마크다운 수식($...$)으로 해석되지 않게
    st.markdown(CSS + f'<div class="kc" style="grid-template-columns:repeat({cols},minmax(0,1fr))">{cells}</div>',
                unsafe_allow_html=True)


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
def _empty_opts() -> pd.DataFrame:
    return pd.DataFrame({"기초자산": pd.Series(dtype=str), "만기": pd.Series(dtype=str), "종류": pd.Series(dtype=str),
                         "행사가": pd.Series(dtype=float), "수량(계약)": pd.Series(dtype=int)})


def _empty_stocks() -> pd.DataFrame:
    return pd.DataFrame({"종목코드": pd.Series(dtype=str), "종목명": pd.Series(dtype=str),
                         "보유 주식 수": pd.Series(dtype=float), "비고": pd.Series(dtype=str)})


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
def render_options() -> None:
    if yf is None:
        st.error("yfinance 패키지가 설치되지 않았습니다. requirements.txt에 yfinance를 추가해 주세요.")
        return
    st.caption("미국 상장 종목·ETF 옵션 (Yahoo Finance, 약 15분 지연). 국내 개별주식옵션은 무료 데이터가 없어 지원하지 않습니다. "
               "조회는 느릴 수 있으며, 실패하면 자동으로 여러 번 다시 시도합니다.")
    from capm_opt import render_capm_opt

    from optvol import render_vol

    from american import render_american

    (t1, t1_open), (t4, t4_open), (t5, t5_open), (t2, t2_open), (t3, t3_open) = ui.lazy_tabs(["옵션 체인", "변동성·수급 분석", "아메리칸 옵션", "델타·감마 헷지", "옵션 포함 CAPM·손익"], key="opt_tabs")
    if t1_open:
        with t1:
            render_chain()
    if t4_open:
        with t4:
            render_vol()
    if t5_open:
        with t5:
            render_american()
    if t2_open:
        with t2:
            render_hedge()
    if t3_open:
        with t3:
            render_capm_opt()
