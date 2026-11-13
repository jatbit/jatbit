"""일별 밸류에이션 배수 (기업 리서치 탭)
SEC 분기 공시(10-Q·10-K)에서 최근 12개월(TTM) 실적과 분기 말 재무상태표를 만들고, '공시가 제출된 날'부터
매일 주가 × 주식 수(시가총액)로 나눠 PBR·PSR·EV/EBITDA 등을 매일 계산한다. 제출일 기준이라 미래 정보를 쓰지 않는다.
"""
import numpy as np
import pandas as pd
import streamlit as st

DUR = {  # 기간 항목: 앞의 태그일수록 우선
    "매출": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
           "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "영업이익": ["OperatingIncomeLoss"],
    "순이익": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "영업현금흐름": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "설비투자": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "감가상각비": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet",
              "Depreciation"],
    "배당금": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
}
INST = {  # 시점 항목
    "자기자본": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "단기차입금": ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"],
    "장기차입금": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "현금성자산": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "단기투자": ["MarketableSecuritiesCurrent", "ShortTermInvestments", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
}
FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT")

# 이름: (단위, 설명)
VAL = {
    "PER (시가총액 ÷ 순이익 TTM)": ("배", "시가총액 ÷ 최근 12개월 순이익"),
    "PBR": ("배", "시가총액 ÷ 분기 말 자기자본"),
    "PSR": ("배", "시가총액 ÷ 최근 12개월 매출"),
    "EV/EBITDA": ("배", "기업가치 ÷ (최근 12개월 영업이익 + 감가상각비)"),
    "EV/EBIT": ("배", "기업가치 ÷ 최근 12개월 영업이익"),
    "EV/매출": ("배", "기업가치 ÷ 최근 12개월 매출"),
    "P/FCF": ("배", "시가총액 ÷ 최근 12개월 잉여현금흐름(영업현금흐름 − 설비투자)"),
    "P/영업현금흐름": ("배", "시가총액 ÷ 최근 12개월 영업현금흐름"),
    "이익수익률 (E/P)": ("%", "최근 12개월 순이익 ÷ 시가총액"),
    "FCF 수익률": ("%", "최근 12개월 잉여현금흐름 ÷ 시가총액"),
    "배당수익률 (TTM)": ("%", "최근 12개월 배당금 지급액 ÷ 시가총액"),
}
STALE_DAYS = 200  # 마지막 공시가 이보다 오래되면 값을 비움


def _entries(facts: dict, ns: str, tags: list, unit: str) -> list:
    out = []
    node = facts.get("facts", {}).get(ns, {})
    for pri, tag in enumerate(tags):
        for e in node.get(tag, {}).get("units", {}).get(unit, []):
            if e.get("form") not in FORMS or e.get("val") is None:
                continue
            out.append({"pri": pri, "start": e.get("start"), "end": e["end"], "val": float(e["val"]), "filed": e.get("filed") or e["end"],
                        "accn": e.get("accn")})
    return out


def _first_by_key(entries: list, key) -> dict:
    """같은 기간(키)은 '가장 먼저 제출된 값'(그 시점에 시장이 알던 값), 태그는 우선순위가 높은 것."""
    best = {}
    for e in entries:
        k = key(e)
        b = best.get(k)
        if b is None or (e["pri"], e["filed"]) < (b["pri"], b["filed"]):
            best[k] = e
    return best


def quarterly_flow(facts: dict, tags: list) -> pd.DataFrame:
    """기간 항목 → 분기별 값(end, val, filed). 10-Q의 누적(6·9개월) 값과 10-K 연간 값의 차이로 빠진 분기를 채운다."""
    ents = [e for e in _entries(facts, "us-gaap", tags, "USD") if e["start"]]
    if not ents:
        return pd.DataFrame(columns=["val", "filed"])
    best = _first_by_key(ents, lambda e: (e["start"], e["end"]))
    q = {}
    by_start = {}
    for (s, e_), v in best.items():
        days = (pd.Timestamp(e_) - pd.Timestamp(s)).days
        if 80 <= days <= 100:
            q[e_] = (v["val"], v["filed"])
        by_start.setdefault(s, []).append((e_, days, v))
    # 같은 시작일의 누적값(3·6·9·12개월)을 차분
    for s, lst in by_start.items():
        lst = sorted(lst, key=lambda x: x[0])
        prev = None
        for e_, days, v in lst:
            if prev is not None:
                gap = (pd.Timestamp(e_) - pd.Timestamp(prev[0])).days
                if 80 <= gap <= 100 and e_ not in q:
                    q[e_] = (v["val"] - prev[2]["val"], max(v["filed"], prev[2]["filed"]))
            if days >= 80:
                prev = (e_, days, v)
    if not q:
        return pd.DataFrame(columns=["val", "filed"])
    df = pd.DataFrame({"val": {pd.Timestamp(k): v[0] for k, v in q.items()},
                       "filed": {pd.Timestamp(k): pd.Timestamp(v[1]) for k, v in q.items()}}).sort_index()
    return df


def ttm(qdf: pd.DataFrame) -> pd.DataFrame:
    """연속된 4개 분기 합. 이용 가능 시점 = 그중 가장 늦은 제출일."""
    if qdf is None or len(qdf) < 4:
        return pd.DataFrame(columns=["val", "filed"])
    idx = list(qdf.index)
    out = {}
    for i in range(3, len(idx)):
        span = (idx[i] - idx[i - 3]).days
        if 250 <= span <= 290:
            w = qdf.iloc[i - 3:i + 1]
            out[idx[i]] = (float(w["val"].sum()), w["filed"].max())
    return pd.DataFrame({"val": {k: v[0] for k, v in out.items()}, "filed": {k: v[1] for k, v in out.items()}}).sort_index()


def instant(facts: dict, tags: list, ns: str = "us-gaap", unit: str = "USD", sum_by_accn: bool = False) -> pd.DataFrame:
    ents = [e for e in _entries(facts, ns, tags, unit) if not e["start"]]
    if not ents:
        return pd.DataFrame(columns=["val", "filed"])
    if sum_by_accn:  # 주식 종류가 여럿이면(예: GOOGL A·B·C) 같은 공시의 값을 합산
        agg = {}
        for e in ents:
            k = (e["accn"], e["end"])
            a = agg.setdefault(k, dict(e, val=0.0))
            a["val"] += e["val"]
        ents = list(agg.values())
    best = _first_by_key(ents, lambda e: e["end"])
    return pd.DataFrame({"val": {pd.Timestamp(k): v["val"] for k, v in best.items()},
                         "filed": {pd.Timestamp(k): pd.Timestamp(v["filed"]) for k, v in best.items()}}).sort_index()


@st.cache_data(ttl=24 * 60 * 60, show_spinner="SEC 분기 공시로 일별 밸류에이션을 계산하는 중...")
def load_quarterly(ticker: str) -> dict:
    from research import _sec_cik_map, _sec_json, _try

    cik = _sec_cik_map().get(ticker.upper()) or _sec_cik_map().get(ticker.upper().replace(".", "-"))
    if not cik:
        raise RuntimeError("SEC 등록 기업이 아닙니다")
    facts = _try(lambda: _sec_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"), tries=3)
    out = {}
    for k, tags in DUR.items():
        out[k] = ttm(quarterly_flow(facts, tags))
    for k, tags in INST.items():
        out[k] = instant(facts, tags)
    sh = instant(facts, ["EntityCommonStockSharesOutstanding"], ns="dei", unit="shares", sum_by_accn=True)
    if len(sh) < 4:
        sh = instant(facts, ["CommonStockSharesOutstanding"], unit="shares")
    out["주식수"] = sh
    return out


def _asof(df: pd.DataFrame, days: pd.DatetimeIndex, stale: int = STALE_DAYS) -> pd.Series:
    """제출일 기준으로 그날 알 수 있던 가장 최근 값. 너무 오래된 값은 비움."""
    if df is None or len(df) == 0:
        return pd.Series(np.nan, index=days)
    s = df.sort_values("filed").reset_index().rename(columns={"index": "end"})
    s = s.drop_duplicates("filed", keep="last")
    # 날짜 정밀도(초·마이크로초·나노초)가 섞이면 merge_asof가 실패하므로 모두 나노초로 맞춤
    s["filed"] = pd.to_datetime(s["filed"]).astype("datetime64[ns]")
    s["end"] = pd.to_datetime(s["end"]).astype("datetime64[ns]")
    left = pd.DataFrame({"d": pd.DatetimeIndex(days).astype("datetime64[ns]")})
    m = pd.merge_asof(left, s[["filed", "val", "end"]].rename(columns={"filed": "d"}), on="d", direction="backward")
    age = (m["d"] - m["end"]).dt.days
    v = m["val"].where(age <= stale)
    return pd.Series(v.values, index=days)


def daily_valuation(Q: dict, hist: pd.Series, splits, stale: int = STALE_DAYS) -> pd.DataFrame:
    h = hist.dropna().astype(float).copy()
    h.index = pd.to_datetime(h.index)
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    days = h.index
    sh = Q.get("주식수")
    if sh is None or len(sh) == 0:
        return pd.DataFrame(index=days)
    sh = sh.copy()
    # 액면분할: 공시 제출일 이후 분할 비율만큼 주식 수를 늘려 오늘 기준(수정주가와 같은 기준)으로 맞춤
    if splits is not None and len(splits):
        sp = splits.copy()
        sp.index = pd.to_datetime(sp.index)
        if sp.index.tz is not None:
            sp.index = sp.index.tz_localize(None)
        sh["val"] = [v * float(np.prod(sp[sp.index > f].values)) if (sp.index > f).any() else v for v, f in zip(sh["val"], sh["filed"])]
    shares = _asof(sh, days, max(stale, 500))
    mcap = h * shares
    g = {k: _asof(Q.get(k), days, stale) for k in list(DUR) + list(INST)}
    debt = g["단기차입금"].fillna(0) + g["장기차입금"].fillna(0)
    cash = g["현금성자산"].fillna(0) + g["단기투자"].fillna(0)
    ev = mcap + debt - cash
    ebitda = g["영업이익"] + g["감가상각비"].fillna(0)
    fcf = g["영업현금흐름"] - g["설비투자"].fillna(0)
    pos = lambda x: x.where(x > 0)  # noqa: E731  (적자·음수 분모는 배수가 의미 없음)
    out = pd.DataFrame({
        "시가총액": mcap,
        "PER (시가총액 ÷ 순이익 TTM)": mcap / pos(g["순이익"]),
        "PBR": mcap / pos(g["자기자본"]),
        "PSR": mcap / pos(g["매출"]),
        "EV/EBITDA": ev / pos(ebitda),
        "EV/EBIT": ev / pos(g["영업이익"]),
        "EV/매출": ev / pos(g["매출"]),
        "P/FCF": mcap / pos(fcf),
        "P/영업현금흐름": mcap / pos(g["영업현금흐름"]),
        "이익수익률 (E/P)": g["순이익"] / mcap * 100,
        "FCF 수익률": fcf / mcap * 100,
        "배당수익률 (TTM)": g["배당금"].abs() / mcap * 100,
    }, index=days)
    return out.replace([np.inf, -np.inf], np.nan)


def summary_table(dv: pd.DataFrame, keys: list) -> pd.DataFrame:
    rows = []
    last = dv.index[-1]
    for k in keys:
        s = dv[k].dropna()
        if s.empty:
            continue
        cur = float(s.iloc[-1])
        row = {"지표": k, "현재": cur}
        for y in (1, 3, 5, 10):
            w = s[s.index >= last - pd.DateOffset(years=y)]
            row[f"{y}년 평균"] = float(w.mean()) if len(w) > 60 and (w.index[0] - (last - pd.DateOffset(years=y))).days < 60 else np.nan
        row["최저"], row["최고"] = float(s.min()), float(s.max())
        row["현재 백분위"] = float((s < cur).mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows)
