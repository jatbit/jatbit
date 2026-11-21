"""국내 옵션 (주식옵션 · 코스피200 옵션)
- 시세는 KRX가 로그인을 요구해 자동으로 받지 못한다 → KRX 정보데이터시스템에서 내려받은 CSV를 올리거나 직접 입력
- 만기 = 해당 월 둘째 목요일, 유럽형(만기일에만 행사) → 블랙-숄즈-머튼으로 IV·그릭스
- 승수: 주식옵션 1계약 = 10주, 코스피200 옵션 = 지수 1포인트당 250,000원
"""
import io
import math
import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import bsm

KS200 = "KOSPI200"
MULT_STOCK, MULT_K200 = 10, 250_000


def second_thursday(y: int, m: int) -> date:
    d = date(y, m, 1)
    first_thu = d + timedelta(days=(3 - d.weekday()) % 7)
    return first_thu + timedelta(days=7)


def kr_expiries(n: int = 8) -> list:
    """앞으로의 월물 만기 (둘째 목요일) n개."""
    t = date.today()
    out, y, m = [], t.year, t.month
    while len(out) < n:
        e = second_thursday(y, m)
        if e >= t:
            out.append(e.isoformat())
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def mult_of(code: str) -> int:
    return MULT_K200 if str(code).upper() in (KS200, "KS200", "K200") else MULT_STOCK


@st.cache_data(ttl=30 * 60, show_spinner=False)
def kr_close(code: str) -> pd.Series:
    import FinanceDataReader as fdr

    sym = "KS200" if str(code).upper() in (KS200, "KS200", "K200") else str(code).zfill(6)
    start = (datetime.now() - timedelta(days=365 * 12)).strftime("%Y-%m-%d")
    s = fdr.DataReader(sym, start)["Close"].dropna()
    return s[s > 0]


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def kr_rf() -> tuple:
    """무위험금리(단순, 소수): 국고채 3년 → 실패하면 3%."""
    try:
        from kr import kr_rate

        v = kr_rate("3Y")
        if v and np.isfinite(v[0]):
            return float(v[0]), f"{v[2]} {v[1]}"
    except Exception:  # noqa: BLE001
        pass
    return 0.03, "기본값 3%"


def name_of(code: str) -> str:
    if str(code).upper() in (KS200, "KS200", "K200"):
        return "코스피200"
    try:
        from kr import resolve

        hit = resolve(str(code))
        return hit[0]["name"] if hit else str(code)
    except Exception:  # noqa: BLE001
        return str(code)


def resolve_under(q: str):
    """이름·코드 → (코드, 이름). 코스피200은 'KOSPI200'."""
    q = (q or "").strip()
    if not q:
        return None
    if q.upper() in (KS200, "KS200", "K200") or q.replace(" ", "") in ("코스피200", "KOSPI200"):
        return KS200, "코스피200"
    try:
        from kr import resolve

        hit = resolve(q)
        if hit:
            return hit[0]["code"], hit[0]["name"]
    except Exception:  # noqa: BLE001
        pass
    return (q.zfill(6), q) if re.fullmatch(r"\d{1,6}", q) else None


# ----------------------------------------------------------------------
# KRX CSV
# ----------------------------------------------------------------------
_NAME_RE = re.compile(r"(?P<under>.+?)\s*(?P<cp>[CP]|콜|풋)\s*(?P<ym>\d{4}(?:\d{2})?)\s+(?P<k>[\d,]+(?:\.\d+)?)\s*$")


def _num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.replace(",", "").str.strip(), errors="coerce")


def parse_krx_csv(raw: bytes) -> pd.DataFrame:
    """KRX '주식옵션/코스피200옵션 전종목 시세' CSV → [기초자산명, 종류, 만기, 행사가, 종가, 거래량, 미결제약정, KRX IV]."""
    df = None
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            break
        except Exception:  # noqa: BLE001
            continue
    if df is None or df.empty:
        raise ValueError("CSV를 읽지 못했습니다.")
    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)

    def col(*keys):
        for k in keys:
            for c in df.columns:
                if k in c:
                    return c
        return None

    c_name, c_px = col("종목명", "종목"), col("종가", "현재가", "정산가")
    if c_name is None or c_px is None:
        raise ValueError("'종목명'과 '종가' 열이 필요합니다.")
    rows = []
    for nm, px, vol, oi, iv in zip(df[c_name].astype(str), _num(df[c_px]),
                                   _num(df[col("거래량")]) if col("거래량") else [np.nan] * len(df),
                                   _num(df[col("미결제")]) if col("미결제") else [np.nan] * len(df),
                                   _num(df[col("내재변동성", "IV")]) if col("내재변동성", "IV") else [np.nan] * len(df)):
        m = _NAME_RE.search(re.sub(r"\s+", " ", nm.strip()))
        if not m:
            continue
        ym = m.group("ym")
        y, mo = (2000 + int(ym[:2]), int(ym[2:4])) if len(ym) == 4 else (int(ym[:4]), int(ym[4:6]))
        if not 1 <= mo <= 12:
            continue
        rows.append({"기초자산명": m.group("under").strip(), "종류": "콜" if m.group("cp") in ("C", "콜") else "풋",
                     "만기": second_thursday(y, mo).isoformat(), "행사가": float(m.group("k").replace(",", "")),
                     "종가": px, "거래량": vol, "미결제약정": oi, "KRX IV": iv})
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("종목명에서 '기초자산 C/P 연월 행사가' 형식을 찾지 못했습니다 (예: '삼성전자 C 202612 60,000').")
    return out


def chain_greeks(ch: pd.DataFrame, S: float, r: float, q: float = 0.0) -> pd.DataFrame:
    d = ch.copy()
    today = pd.Timestamp(date.today())
    d["T"] = ((pd.to_datetime(d["만기"]) - today).dt.days.clip(lower=0) + 0.5) / 365
    c = (d["종류"] == "콜").values
    K, T, px = d["행사가"].values.astype(float), d["T"].values.astype(float), d["종가"].values.astype(float)
    iv = np.array([float(bsm.implied_vol(np.array([p]), S, np.array([k]), t, r, q, np.array([cc]))[0])
                   if np.isfinite(p) and p > 0 else np.nan for p, k, t, cc in zip(px, K, T, c)])
    d["IV"] = iv
    sig = np.where(np.isfinite(iv), iv, np.nanmedian(iv) if np.isfinite(iv).any() else 0.3)
    g = bsm.greeks(S, K, T, r, q, sig, c)
    for k, v in g.items():
        d[k] = v
    d["sig"] = sig
    return d


# ----------------------------------------------------------------------
# 포지션 (세션: kr_opts_cur)
# ----------------------------------------------------------------------
KR_COLS = ["기초자산", "만기", "종류", "행사가", "수량(계약)", "가격"]


def empty_kr() -> pd.DataFrame:
    return pd.DataFrame(columns=KR_COLS)


def kr_book() -> pd.DataFrame:
    b = st.session_state.get("kr_opts_cur")
    return empty_kr() if b is None else b


def add_kr_rows(rows: list) -> None:
    b = kr_book()
    st.session_state["kr_opts_cur"] = pd.concat([b, pd.DataFrame(rows, columns=KR_COLS)], ignore_index=True)
    st.session_state["kr_opts_ver"] = st.session_state.get("kr_opts_ver", 0) + 1


def price_kr_book(book: pd.DataFrame) -> tuple:
    """{코드: {"S","r","q","close","name","mult","legs":[...]}} — 가격을 넣었으면 그 가격의 IV, 없으면 HV 60일."""
    r_s, r_lab = kr_rf()
    r = math.log1p(r_s)
    out, warns = {}, []
    for rw in book.to_dict("records"):
        code = str(rw.get("기초자산") or "").strip()
        K = pd.to_numeric(rw.get("행사가"), errors="coerce")
        n = pd.to_numeric(rw.get("수량(계약)"), errors="coerce")
        if not code or code == "nan" or rw.get("종류") not in ("콜", "풋") or not np.isfinite(K) or not np.isfinite(n) or not rw.get("만기"):
            continue
        if code not in out:
            try:
                cl = kr_close(code)
            except Exception as e:  # noqa: BLE001
                warns.append(f"{code}: 기초자산 가격을 불러오지 못해 제외 ({e})")
                out[code] = None
                continue
            hv = bsm.hist_vol(cl, 60)
            out[code] = {"S": float(cl.iloc[-1]), "asof": str(cl.index[-1].date()), "r": r, "r_simple": r_s, "q": 0.0,
                         "close": cl, "hv": hv if hv == hv else 0.3, "name": name_of(code), "mult": mult_of(code), "legs": []}
        u = out.get(code)
        if u is None:
            continue
        T = max((pd.Timestamp(str(rw["만기"])[:10]) - pd.Timestamp(date.today())).days + 0.5, 0.5) / 365
        is_call = rw["종류"] == "콜"
        px = pd.to_numeric(rw.get("가격"), errors="coerce")
        if np.isfinite(px) and px > 0:
            sig = float(bsm.implied_vol(np.array([px]), u["S"], np.array([K]), T, r, 0.0, np.array([is_call]))[0])
            src = "입력 가격"
            if not np.isfinite(sig):
                sig, src = u["hv"], "HV 60일 (IV 계산 실패)"
        else:
            sig, src = u["hv"], "HV 60일"
            px = float(bsm.price(u["S"], K, T, r, 0.0, sig, is_call))
        g = bsm.greeks(u["S"], K, T, r, 0.0, sig, is_call)
        u["legs"].append({"exp": str(rw["만기"])[:10], "is_call": is_call, "K": float(K), "n": float(n), "T": T, "mkt": float(px),
                          "sig": sig, "src": src, **{k: float(v) for k, v in g.items()},
                          "label": f"{u['name']} {str(rw['만기'])[:7]} {rw['종류']} {K:,.0f}"})
    return {k: v for k, v in out.items() if v}, warns, r_lab


def _add_msg(row: dict, msg: str) -> None:
    add_kr_rows([row])
    st.session_state["kr_add_msg"] = msg


def render_kr_add() -> None:
    """국내 옵션 추가: CSV 업로드(체인) 또는 직접 입력."""
    import ui

    st.caption("국내 옵션 시세는 KRX가 로그인을 요구해 자동으로 받지 못합니다. KRX 정보데이터시스템 → 파생상품 → "
               "주식옵션(또는 코스피200옵션) 전종목 시세를 CSV로 내려받아 올리거나, 아래에 직접 입력하세요.")
    up = st.file_uploader("KRX 옵션 시세 CSV", type=["csv"], key="kr_csv")
    if up is not None:
        try:
            st.session_state["kr_chain"] = parse_krx_csv(up.getvalue())
        except Exception as e:  # noqa: BLE001
            st.error(f"CSV를 해석하지 못했습니다: {e}")
    ch = st.session_state.get("kr_chain")
    c1, c2 = st.columns([1.4, 1])
    q = c1.text_input("기초자산 (종목명·코드, 코스피200은 'KOSPI200')", key="kr_under_q", placeholder="두산에너빌리티")
    hit = resolve_under(q)
    if q and not hit:
        st.warning("종목을 찾지 못했습니다.")
    if not hit:
        return
    code, name = hit
    try:
        cl = kr_close(code)
        S = float(cl.iloc[-1])
    except Exception as e:  # noqa: BLE001
        st.error(f"{name} 가격을 불러오지 못했습니다: {e}")
        return
    c2.metric(f"{name} 현재가", f"{S:,.0f}" if S > 1000 else f"{S:,.2f}", f"기준 {cl.index[-1]:%Y-%m-%d}", delta_color="off")
    sub = None
    if ch is not None and len(ch):
        sub = ch[ch["기초자산명"].str.replace(" ", "").str.contains(name.replace(" ", ""), regex=False)]
        if sub.empty:
            st.caption(f"올린 CSV에 {name} 옵션이 없습니다 — 직접 입력으로 넣으세요.")
            sub = None
    r_s, r_lab = kr_rf()
    if sub is not None:
        exps = sorted(sub["만기"].unique())
        e = st.selectbox("만기 (CSV)", exps, key=f"kr_exp_csv_{code}")
        d = chain_greeks(sub[sub["만기"] == e], S, math.log1p(r_s))
        calls = d[d["종류"] == "콜"].set_index("행사가")
        puts = d[d["종류"] == "풋"].set_index("행사가")
        ks = sorted(set(calls.index) | set(puts.index))
        tb = pd.DataFrame({"콜 델타": calls["delta"].reindex(ks).round(3), "콜 IV(%)": (calls["IV"].reindex(ks) * 100).round(1),
                           "콜 종가": calls["종가"].reindex(ks), "행사가": ks, "풋 종가": puts["종가"].reindex(ks),
                           "풋 IV(%)": (puts["IV"].reindex(ks) * 100).round(1), "풋 델타": puts["delta"].reindex(ks).round(3)})
        st.dataframe(tb, hide_index=True, use_container_width=True, height=300)
        st.caption(f"IV는 종가로 직접 계산 (무위험금리 {r_s * 100:.2f}%, {r_lab}, 배당 0 가정). 1계약 = {mult_of(code):,}"
                   + ("원 × 지수" if mult_of(code) > 1000 else "주"))
        a1, a2, a3, a4 = st.columns([1, 1.4, 1.2, 1.2])
        kind = a1.selectbox("종류", ["콜", "풋"], key=f"kr_k_{code}")
        atm = int(np.argmin([abs(k - S) for k in ks]))
        K = a2.selectbox("행사가", ks, index=atm, key=f"kr_K_{code}_{e}", format_func=lambda x: f"{x:,.2f}" if x < 1000 else f"{x:,.0f}")
        n = a3.number_input("수량(계약, 매도 −)", value=1, step=1, key=f"kr_n_{code}")
        row = (calls if kind == "콜" else puts)
        px = float(row.loc[K, "종가"]) if K in row.index else np.nan
        a4.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        a4.button("추가", key=f"kr_add_csv_{code}", use_container_width=True, on_click=_add_msg,
                  args=({"기초자산": code, "만기": e, "종류": kind, "행사가": K, "수량(계약)": n, "가격": px},
                        f"{name} {e} {kind} {K:,.0f} × {n}계약을 추가했습니다."))
    else:
        a1, a2, a3, a4, a5 = st.columns([1.3, 0.8, 1.1, 1, 1.1])
        e = a1.selectbox("만기 (둘째 목요일)", kr_expiries(), key=f"kr_exp_{code}")
        kind = a2.selectbox("종류", ["콜", "풋"], key=f"kr_k2_{code}")
        from strat import nice_step

        stp = nice_step(S)
        K = a3.number_input("행사가", value=float(round(S / stp) * stp), step=float(stp), key=f"kr_K2_{code}")
        n = a4.number_input("수량(계약, 매도 −)", value=1, step=1, key=f"kr_n2_{code}")
        px = a5.number_input("가격 (모르면 0)", value=0.0, step=float(stp) / 50, key=f"kr_px2_{code}",
                             help="0이면 HV 60일로 이론가를 계산합니다")
        st.button("추가", key=f"kr_add_man_{code}", on_click=_add_msg,
                  args=({"기초자산": code, "만기": e, "종류": kind, "행사가": K, "수량(계약)": n, "가격": px or np.nan},
                        f"{name} {e} {kind} {K:,.0f} × {n}계약을 추가했습니다."))
    msg = st.session_state.pop("kr_add_msg", None)
    if msg:
        st.success(msg)


def kr_chain_for(code: str, name: str):
    ch = st.session_state.get("kr_chain")
    if ch is None or not len(ch):
        return None
    sub = ch[ch["기초자산명"].str.replace(" ", "").str.contains(name.replace(" ", ""), regex=False)]
    return sub if len(sub) else None


__all__ = [n for n in dir() if not n.startswith("__")]
