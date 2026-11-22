"""기업 리서치 판단용 데이터 준비: 업종 벤치마크·외국인 수급, 분기 TTM 재무표, 위험 신호 항목, 동종업계, 실적 서프라이즈."""
import numpy as np
import pandas as pd
import streamlit as st

from research_data import RATIOS, _split_factor, annual_eps_table, compute_ratios

from outlook_rules import *  # noqa: F401,F403

# ----------------------------------------------------------------------
# 신호 계산
# ----------------------------------------------------------------------
def _fmt(unit):
    return (lambda x: f"{x:.1f}%") if unit == "%" else (lambda x: f"{x:.2f}{unit}")


def _eps_growth(sec, splits):
    """공시일 기준 3년 연평균 EPS 성장률(%) 계단 시계열과 회계연도별 값."""
    tb = annual_eps_table(sec, splits)
    if tb is None:
        return None, None
    v, g_by_date, g_by_fy = tb["EPS"].values, {}, {}
    for j in range(len(tb)):
        g = np.nan
        if j >= 3 and v[j] > 0 and v[j - 3] > 0:
            g = ((v[j] / v[j - 3]) ** (1 / 3) - 1) * 100
        g_by_date[tb.index[j]] = g
        g_by_fy[pd.Timestamp(tb["회계연도말"].iloc[j])] = g
    return pd.Series(g_by_date).sort_index(), pd.Series(g_by_fy).sort_index()


def _step_at(step: pd.Series, idx) -> pd.Series:
    if step is None or not len(step):
        return pd.Series(np.nan, index=idx)
    pos = np.searchsorted(step.index.values, np.asarray(idx, dtype="datetime64[ns]"), side="right") - 1
    vals = np.where(pos >= 0, step.values[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=idx)


SECTOR_ETF = {  # Yahoo 섹터 → 미국 섹터 ETF (업종 평균 수익률 대용)
    "Technology": ("XLK", "기술"), "Financial Services": ("XLF", "금융"), "Healthcare": ("XLV", "헬스케어"),
    "Consumer Cyclical": ("XLY", "경기소비재"), "Consumer Defensive": ("XLP", "필수소비재"), "Energy": ("XLE", "에너지"),
    "Industrials": ("XLI", "산업재"), "Basic Materials": ("XLB", "소재"), "Real Estate": ("XLRE", "부동산"),
    "Utilities": ("XLU", "유틸리티"), "Communication Services": ("XLC", "커뮤니케이션"),
}


def sector_bench(ticker, info, domestic):
    """업종 평균 수익률로 쓸 지수 (심볼, 이름). 미국: 섹터 ETF(SPDR), 국내: 업종 지수가 없어 시장 지수."""
    if domestic:
        return ("^KQ11", "코스닥 (국내 업종 지수 없음)") if ticker.endswith(".KQ") else ("^KS11", "코스피 (국내 업종 지수 없음)")
    sec_name = (info or {}).get("sector")
    if sec_name in SECTOR_ETF:
        sym, nm = SECTOR_ETF[sec_name]
        return sym, f"{nm} 업종 ETF {sym}"
    try:
        sic, _d = sec_sic(ticker)
    except Exception:  # noqa: BLE001
        sic = 0
    rng = [((6798, 6798), "Real Estate"), ((6500, 6599), "Real Estate"), ((6000, 6799), "Financial Services"),
           ((4900, 4999), "Utilities"), ((4800, 4899), "Communication Services"), ((1300, 1399), "Energy"),
           ((2900, 2999), "Energy"), ((2830, 2836), "Healthcare"), ((3841, 3851), "Healthcare"), ((8000, 8099), "Healthcare"),
           ((7370, 7379), "Technology"), ((3570, 3579), "Technology"), ((3660, 3679), "Technology"),
           ((2000, 2199), "Consumer Defensive"), ((2840, 2844), "Consumer Defensive"), ((5400, 5499), "Consumer Defensive"),
           ((3710, 3716), "Consumer Cyclical"), ((5000, 5999), "Consumer Cyclical"), ((7000, 7999), "Consumer Cyclical"),
           ((1000, 1499), "Basic Materials"), ((2600, 2829), "Basic Materials"), ((3300, 3399), "Basic Materials"),
           ((1500, 1799), "Industrials"), ((3400, 3569), "Industrials"), ((3580, 3659), "Industrials"),
           ((3720, 3799), "Industrials"), ((4000, 4799), "Industrials")]
    for (a, b), nm in rng:
        if sic and a <= sic <= b:
            sym, kn = SECTOR_ETF[nm]
            return sym, f"{kn} 업종 ETF {sym}"
    return "^GSPC", "S&P 500 (업종 판별 실패)"


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def foreign_flow(code: str):
    """국내 종목 일별 외국인 지분율(%)·순매수(주) — 다음 금융(finance.daum.net) 투자자별 매매 동향, 최근 약 5년.
    열: rate(외국인 지분율 %), net(외국인 순매수 주식 수). 못 받으면 None."""
    import json
    import urllib.request

    rows = []
    for pg in range(1, 9):
        url = f"https://finance.daum.net/api/investor/days?symbolCode=A{code}&page={pg}&perPage=200&pagination=true"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            "Referer": f"https://finance.daum.net/quotes/A{code}", "Accept": "application/json, text/plain, */*"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            j = json.loads(resp.read().decode("utf-8"))
        d = j.get("data") or []
        rows += d
        if not d or pg >= int(j.get("totalPages") or 1):
            break
    if not rows:
        return None
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["date"]).dt.normalize()
    out = pd.DataFrame({"rate": pd.to_numeric(df["foreignOwnSharesRate"], errors="coerce").values * 100,
                        "net": pd.to_numeric(df["foreignStraightPurchaseVolume"], errors="coerce").values}, index=idx.values)
    out = out[~out.index.duplicated()].sort_index()
    return out if out["rate"].notna().sum() > 60 else None


# ----------------------------------------------------------------------
# 데이터 준비 (분기 TTM 재무표, 위험 신호용 항목, 동종업계, 실적 서프라이즈)
# ----------------------------------------------------------------------
MRP = 0.05  # 시장 위험프리미엄 가정 (CAPM 기준점)
SHRINK_K = 3.0  # 사례가 적을수록 CAPM 쪽으로 당기는 정도
VAL_TICKERS = {  # 여러 종목 검증용 대표 종목
    "growth": ["MSFT", "ADBE", "CRM", "ORCL", "GOOGL", "META", "AMZN", "NOW", "INTU"],
    "cyclical": ["NVDA", "AMD", "MU", "NUE", "DOW", "F", "GM", "CAT", "XOM"],
    "financial": ["JPM", "BAC", "WFC", "C", "GS", "MS", "PNC", "USB", "TRV"],
    "reit": ["O", "PLD", "SPG", "AMT", "PSA", "EQR", "AVB", "WELL", "VTR"],
    "defensive": ["KO", "PEP", "PG", "DUK", "SO", "VZ", "JNJ", "MO", "CL"],
    "general": ["HD", "COST", "WMT", "NKE", "UNP", "UPS", "LMT", "HON", "MCD"],
}


def _qflow(facts, tags, unit):
    """분기별 기간 값 (valuation.quarterly_flow와 같은 방식, 단위 지정)."""
    import valuation as V

    ents = [e for e in V._entries(facts, "us-gaap", tags, unit) if e["start"]]
    if not ents:
        return pd.DataFrame(columns=["val", "filed"])
    best = V._first_by_key(ents, lambda e: (e["start"], e["end"]))
    q, by_start = {}, {}
    for (s_, e_), v in best.items():
        days = (pd.Timestamp(e_) - pd.Timestamp(s_)).days
        if 80 <= days <= 100:
            q[e_] = (v["val"], v["filed"])
        by_start.setdefault(s_, []).append((e_, days, v))
    for s_, lst in by_start.items():
        prev = None
        for e_, days, v in sorted(lst, key=lambda x: x[0]):
            if prev is not None:
                gap = (pd.Timestamp(e_) - pd.Timestamp(prev[0])).days
                if 80 <= gap <= 100 and e_ not in q:
                    q[e_] = (v["val"] - prev[2]["val"], max(v["filed"], prev[2]["filed"]))
            if days >= 80:
                prev = (e_, days, v)
    if not q:
        return pd.DataFrame(columns=["val", "filed"])
    return pd.DataFrame({"val": {pd.Timestamp(k): v[0] for k, v in q.items()},
                         "filed": {pd.Timestamp(k): pd.Timestamp(v[1]) for k, v in q.items()}}).sort_index()


def _cik(ticker):
    from research_data import _sec_cik_map

    m = _sec_cik_map()
    return m.get(ticker.upper()) or m.get(ticker.upper().replace(".", "-"))


@st.cache_data(ttl=24 * 60 * 60, show_spinner="SEC 분기 공시로 최근 12개월 재무표를 만드는 중...")
def quarterly_frame(ticker: str):
    """분기 말마다 최근 12개월(TTM) 실적 + 분기 말 재무상태표. 열 = 분기 말, '_avail_희석 EPS' = 그 TTM을 알 수 있게 된 날."""
    import valuation as V
    from research_data import SEC_ITEMS, sec_facts

    cik = _cik(ticker)
    if not cik:
        return None
    facts = sec_facts(cik)
    rows, filed = {}, None
    for name, tags, unit, dur in SEC_ITEMS:
        if dur and name == "희석주식수":
            t = _qflow(facts, tags, unit)
        elif dur:
            t = V.ttm(_qflow(facts, tags, unit))
        else:
            t = V.instant(facts, tags, unit=unit)
        if len(t):
            rows[name] = t
    base = rows.get("매출") if "매출" in rows else rows.get("순이익")
    if base is None or len(base) < 8:
        return None
    ends = list(base.index)
    df = pd.DataFrame(index=list(rows), columns=ends, dtype=float)
    for name, t in rows.items():
        for d in ends:
            if d in t.index:
                df.loc[name, d] = t.loc[d, "val"]
            else:
                near = t.index[abs((t.index - d).days) <= 10]
                if len(near):
                    df.loc[name, d] = t.loc[near[-1], "val"]
    if "설비투자" in df.index and "영업현금흐름" in df.index:
        df.loc["잉여현금흐름"] = df.loc["영업현금흐름"] - df.loc["설비투자"].fillna(0)
    av = pd.Series([float(pd.Timestamp(base.loc[d, "filed"]).toordinal()) for d in ends], index=ends)
    df.loc["_avail_희석 EPS"] = av
    df.loc["_filed_희석 EPS"] = av
    df.loc["_filed_희석주식수"] = av
    return df.dropna(how="all")


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def extra_annual(ticker: str) -> dict:
    """Beneish M-스코어용 연간 항목 (매출채권·판관비·유형자산)."""
    from research_data import _sec_series, sec_facts

    cik = _cik(ticker)
    if not cik:
        return {}
    facts = sec_facts(cik)
    items = {"매출채권": (["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"], False),
             "판관비": (["SellingGeneralAndAdministrativeExpense"], True),
             "유형자산": (["PropertyPlantAndEquipmentNet"], False)}
    return {k: _sec_series(facts, tags, "USD", dur) for k, (tags, dur) in items.items()}


def _align(ser, cols):
    out = pd.Series(np.nan, index=cols)
    if ser is None or not len(ser):
        return out
    for c in cols:
        near = ser.index[abs((ser.index - c).days) <= 10]
        if len(near):
            out[c] = ser[near[-1]]
    return out


def red_flags(sec, extra: dict, prof: str) -> dict:
    """회계연도 말 → 위험 신호 목록. Altman Z''(비제조·제조 공용) 1.1 미만 = 부실 위험, Beneish M −1.78 초과 = 이익 조작 의심."""
    cols = list(sec.columns)
    g = lambda k: pd.to_numeric(sec.loc[k], errors="coerce") if k in sec.index else pd.Series(np.nan, index=sec.columns)  # noqa: E731
    out = {c: [] for c in cols}
    ta, tl, eq = g("총자산"), g("총부채"), g("자기자본")
    tl = tl.fillna(ta - eq)
    if prof not in ("financial", "reit"):
        z2 = (6.56 * (g("유동자산") - g("유동부채")) + 3.26 * g("이익잉여금") + 6.72 * g("영업이익") + 1.05 * eq * ta / tl) / ta
        for c in cols:
            if np.isfinite(z2[c]) and z2[c] < 1.1:
                out[c].append(f"부실 위험 (Altman Z'' {z2[c]:.2f})")
    if prof != "financial" and extra:
        rec, sga, ppe = (_align(extra.get(k), sec.columns) for k in ("매출채권", "판관비", "유형자산"))
        sales, gp, ca, dep = g("매출"), g("매출총이익"), g("유동자산"), g("감가상각비")
        ni, cfo = g("순이익"), g("영업현금흐름")
        p = lambda x: x.shift(1)  # noqa: E731
        dsri = (rec / sales) / p(rec / sales)
        gmi = p(gp / sales) / (gp / sales)
        aqi = (1 - (ca + ppe) / ta) / p(1 - (ca + ppe) / ta)
        sgi = sales / p(sales)
        depi = p(dep / (dep + ppe)) / (dep / (dep + ppe))
        sgai = (sga / sales) / p(sga / sales)
        tata = (ni - cfo) / ta
        lvgi = (tl / ta) / p(tl / ta)
        m = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi - 0.172 * sgai.fillna(1)
             + 4.679 * tata - 0.327 * lvgi)
        for c in cols:
            if np.isfinite(m[c]) and m[c] > -1.78:
                out[c].append(f"이익 조작 의심 (Beneish M {m[c]:.2f})")
    return out


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _hist_close(sym: str):
    import yfinance as yf

    t = yf.Ticker(sym)
    h = t.history(period="max", auto_adjust=False)
    sp = None
    try:
        sp = t.splits
    except Exception:  # noqa: BLE001
        pass
    return h["Close"].dropna(), sp


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _peer_ratios(sym: str):
    """동종업계 한 종목의 연간 재무비율과 공시일."""
    from research_data import load_sec_financials

    sec = load_sec_financials(sym)
    h, sp = _hist_close(sym)
    rt = compute_ratios(sec, h, sp)
    rt.columns = [pd.Timestamp(c) for c in rt.columns]
    av = _avail_dates(sec, rt.columns)
    return rt, av


def _surprises(edates) -> pd.Series:
    """발표일 → 실적 서프라이즈 (실제 EPS − 직전 컨센서스) ÷ |컨센서스|."""
    if not isinstance(edates, pd.DataFrame) or edates.empty or not {"EPS Estimate", "Reported EPS"} <= set(edates.columns):
        return pd.Series(dtype=float)
    e = edates.copy()
    e.index = pd.to_datetime(e.index)
    if e.index.tz is not None:
        e.index = e.index.tz_localize(None)
    e.index = e.index.normalize()
    est, rep = pd.to_numeric(e["EPS Estimate"], errors="coerce"), pd.to_numeric(e["Reported EPS"], errors="coerce")
    s = ((rep - est) / est.abs().clip(lower=0.01)).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _px_before(h, d):
    x = h[h.index <= pd.Timestamp(d)]
    return float(x.iloc[-1]) if len(x) else np.nan


def _rf_at(d, domestic, rf_now):
    if domestic:
        return rf_now
    try:
        import stats

        got = stats.us10y_asof(pd.Timestamp(d).date())
        return float(got[0]) if got else rf_now
    except Exception:  # noqa: BLE001
        return rf_now


__all__ = [n for n in dir() if not n.startswith("__")]  # 밑줄로 시작하는 도우미도 다음 모듈에서 쓸 수 있게
