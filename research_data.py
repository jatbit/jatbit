"""기업 리서치 데이터: Yahoo·SEC 10-K 불러오기, 재무비율, 일별 PER 계열, 숫자 표시 도우미, PER·PBR 계산."""
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


REC_KO = {"strong_buy": "적극 매수", "buy": "매수", "hold": "보유", "underperform": "비중 축소", "sell": "매도",
          "strongBuy": "적극 매수", "strongSell": "적극 매도", "none": "-"}


# ----------------------------------------------------------------------
# 데이터
# ----------------------------------------------------------------------
def _try(fn, tries=4, base=1.5):
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


def _safe(fn, default=None):
    try:
        v = fn()
        return default if v is None else v
    except Exception:  # noqa: BLE001
        return default


@st.cache_data(ttl=60 * 60, show_spinner="기업 정보를 불러오는 중... (Yahoo 응답이 느리면 다시 시도합니다)")
def load_research(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = _try(lambda: t.info) or {}
    if not info or (not info.get("longName") and not info.get("shortName") and not info.get("quoteType")):
        raise RuntimeError("기업 정보를 찾지 못했습니다 (티커를 확인해 주세요)")
    out = {"info": info}
    jobs = {  # 서로 독립적인 요청이라 동시에 보낸다 (차례로 보내면 10초 넘게 걸리기도 함)
        "income": (lambda: t.income_stmt, pd.DataFrame()),
        "balance": (lambda: t.balance_sheet, pd.DataFrame()),
        "cashflow": (lambda: t.cashflow, pd.DataFrame()),
        "recs": (lambda: t.recommendations, pd.DataFrame()),
        "targets": (lambda: dict(t.analyst_price_targets), {}),
        "calendar": (lambda: dict(t.calendar), {}),
        "news": (lambda: list(t.news), []),
        "qincome": (lambda: t.quarterly_income_stmt, pd.DataFrame()),
        "eps_est": (lambda: t.earnings_estimate, pd.DataFrame()),
        "eps_trend": (lambda: t.eps_trend, pd.DataFrame()),
        "edates": (lambda: t.get_earnings_dates(limit=60), pd.DataFrame()),
        "hist": (lambda: t.history(period="10y", auto_adjust=False)["Close"].dropna(), pd.Series(dtype=float)),
        "splits": (lambda: t.splits, pd.Series(dtype=float)),
    }
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {k: ex.submit(_safe, fn, d) for k, (fn, d) in jobs.items()}
        for k, f in futs.items():
            out[k] = f.result()
    return out


# ----------------------------------------------------------------------
# SEC EDGAR (미국 공시) 10개년 재무제표 — 키 없이 무료, Yahoo는 4~5년치만 주므로 이것을 우선 사용
# ----------------------------------------------------------------------
SEC_UA = "jatbit-stock-app/1.0 (educational Streamlit app; https://jatbit-jatbit-app-qni3e3.streamlit.app)"
SEC_ITEMS = [  # (표시 이름, us-gaap 태그 우선순위, 단위, 기간형 여부)
    ("매출", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
             "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"], "USD", True),
    ("매출총이익", ["GrossProfit"], "USD", True),
    ("영업이익", ["OperatingIncomeLoss"], "USD", True),
    ("순이익", ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"], "USD", True),
    ("희석 EPS", ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"], "USD/shares", True),
    ("영업현금흐름", ["NetCashProvidedByUsedInOperatingActivities",
                  "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"], "USD", True),
    ("설비투자", ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"], "USD", True),
    ("총자산", ["Assets"], "USD", False),
    ("총부채", ["Liabilities"], "USD", False),
    ("자기자본", ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "USD", False),
    # 아래는 표에는 안 보이고 재무비율 계산에만 쓰는 항목
    ("유동자산", ["AssetsCurrent"], "USD", False),
    ("유동부채", ["LiabilitiesCurrent"], "USD", False),
    ("이자비용", ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"], "USD", True),
    ("희석주식수", ["WeightedAverageNumberOfDilutedSharesOutstanding"], "shares", True),
    ("배당금", ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"], "USD", True),
    # 신용위험(credit.py) 계산용
    ("이익잉여금", ["RetainedEarningsAccumulatedDeficit"], "USD", False),
    ("단기차입금", ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"], "USD", False),
    ("장기차입금", ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebt"], "USD", False),
    ("현금성자산", ["CashAndCashEquivalentsAtCarryingValue",
               "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"], "USD", False),
    ("감가상각비", ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
               "DepreciationAmortizationAndAccretionNet", "Depreciation"], "USD", True),
]
SEC_SHOW = ["매출", "매출총이익", "영업이익", "순이익", "희석 EPS", "영업현금흐름", "잉여현금흐름", "총자산", "총부채", "자기자본"]


def _sec_json(url: str):
    import json
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": SEC_UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _sec_cik_map() -> dict:
    data = _try(lambda: _sec_json("https://www.sec.gov/files/company_tickers.json"), tries=3)
    return {str(v["ticker"]).upper(): int(v["cik_str"]) for v in data.values()}


def _sec_series(facts: dict, tags: list, unit: str, duration: bool, with_filed: bool = False):
    """10-K에 실린 연간 값. 회계연도 말 날짜별로, 앞 순위 태그를 우선·같은 태그는 가장 최근 제출값(정정 반영).
    with_filed=True면 (값, 그 값의 제출일, 그 해 숫자가 처음 공시된 날)을 돌려준다.
    한 해의 EPS는 그 뒤 2~3년치 10-K에 다시 실리는데(액면분할 뒤엔 분할 반영값으로), 값은 최신 제출본을 쓰되
    '그 숫자를 언제부터 알 수 있었나'는 처음 공시된 날로 따져야 일별 PER이 맞다."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    out = {}
    first = {}
    for tag in tags:
        arr = gaap.get(tag, {}).get("units", {}).get(unit, [])
        for e in arr:
            if e.get("form") not in ("10-K", "10-K/A", "10-KT"):
                continue
            end, start = e.get("end"), e.get("start")
            if duration:
                if not start:
                    continue
                days = (pd.Timestamp(end) - pd.Timestamp(start)).days
                if not 340 <= days <= 380:
                    continue
            elif start:
                continue
            prev = out.get(end)
            if prev is None or (prev[0] == tag and e.get("filed", "") > prev[2]):
                out[end] = (tag, e.get("val"), e.get("filed", ""))
            f0 = e.get("filed", "")
            if f0 and (first.get((end, tag)) is None or f0 < first[(end, tag)]):
                first[(end, tag)] = f0
    if not out:
        return (pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)) if with_filed else pd.Series(dtype=float)
    vals = pd.Series({pd.Timestamp(k): float(v[1]) for k, v in out.items()}).sort_index()
    if not with_filed:
        return vals
    filed = pd.Series({pd.Timestamp(k): float(pd.Timestamp(v[2] or k).toordinal()) for k, v in out.items()}).sort_index()
    avail = pd.Series({pd.Timestamp(k): float(pd.Timestamp(first.get((k, v[0])) or v[2] or k).toordinal())
                       for k, v in out.items()}).sort_index()
    return vals, filed, avail


@st.cache_data(ttl=24 * 60 * 60, show_spinner="SEC 공시에서 10개년 재무제표를 불러오는 중...")
def load_sec_financials(ticker: str, years: int = 10) -> pd.DataFrame:
    cik = _sec_cik_map().get(ticker.upper()) or _sec_cik_map().get(ticker.upper().replace(".", "-"))
    if not cik:
        raise RuntimeError("SEC에 등록된 미국 공시 기업이 아닙니다")
    facts = _try(lambda: _sec_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"), tries=3)
    rows = {name: _sec_series(facts, tags, unit, dur) for name, tags, unit, dur in SEC_ITEMS}
    for name, tags, unit, dur in SEC_ITEMS:  # 주당 값·주식 수는 제출일을 같이 저장(액면분할 보정용)
        if name in ("희석 EPS", "희석주식수"):
            rows[name], rows["_filed_" + name], av = _sec_series(facts, tags, unit, dur, with_filed=True)
            if name == "희석 EPS":
                rows["_avail_" + name] = av  # 그 해 EPS가 처음 공시된 날 (일별 PER에서 '언제부터 쓰나')
    base = rows["매출"] if len(rows["매출"]) else rows["순이익"]
    if len(base) == 0:
        raise RuntimeError("연간 재무 데이터(us-gaap)가 없습니다")
    # 회계연도 말 = 손익계산서 기간 끝 날짜. 며칠 차이 나는 재무상태표 날짜는 가까운 연도 말에 맞춘다.
    ends = sorted(base.index)[-years:]
    df = pd.DataFrame(index=list(rows), columns=ends, dtype=float)
    for name, ser in rows.items():
        for d in ends:
            if d in ser.index:
                df.loc[name, d] = ser[d]
            elif len(ser):
                near = ser.index[abs((ser.index - d).days) <= 10]
                if len(near):
                    df.loc[name, d] = ser[near[-1]]
    if df.loc["설비투자"].notna().any():
        df.loc["잉여현금흐름"] = df.loc["영업현금흐름"] - df.loc["설비투자"].fillna(0)
    df = df.drop(index="설비투자")
    return df.dropna(how="all")


# ----------------------------------------------------------------------
# 재무비율 (SEC 10개년 + 회계연도 말 주가)
# ----------------------------------------------------------------------
RATIOS = {  # 이름: (단위, 분류)
    "매출총이익률": ("%", "수익성"), "영업이익률": ("%", "수익성"), "순이익률": ("%", "수익성"), "FCF 마진": ("%", "수익성"),
    "ROE": ("%", "수익성"), "ROA": ("%", "수익성"),
    "매출 성장률": ("%", "성장성"), "영업이익 성장률": ("%", "성장성"), "순이익 성장률": ("%", "성장성"), "EPS 성장률": ("%", "성장성"),
    "부채비율": ("%", "안정성"), "자기자본비율": ("%", "안정성"), "유동비율": ("%", "안정성"), "이자보상배율": ("배", "안정성"),
    "총자산회전율": ("회", "효율성"), "영업현금흐름/순이익": ("배", "현금흐름"), "배당성향": ("%", "현금흐름"),
    "PER": ("배", "밸류에이션"), "PBR": ("배", "밸류에이션"), "PSR": ("배", "밸류에이션"),
}


def _split_factor(filed_ord: pd.Series, splits) -> pd.Series:
    """제출일 이후에 있었던 액면분할 비율의 곱 (그 뒤 분할만큼 주당 값을 나누고 주식 수를 곱해야 오늘 기준이 됨)."""
    if splits is None or len(splits) == 0 or filed_ord is None:
        return pd.Series(1.0, index=filed_ord.index if filed_ord is not None else [])
    sp = splits.copy()
    sp.index = pd.to_datetime(sp.index)
    if sp.index.tz is not None:
        sp.index = sp.index.tz_localize(None)
    out = {}
    for d, f in filed_ord.items():
        if not np.isfinite(f):
            out[d] = 1.0
            continue
        fdate = pd.Timestamp.fromordinal(int(f))
        out[d] = float(np.prod(sp[sp.index > fdate].values)) if (sp.index > fdate).any() else 1.0
    return pd.Series(out)


def compute_ratios(sec: pd.DataFrame, hist: pd.Series, splits) -> pd.DataFrame:
    g = lambda k: sec.loc[k] if k in sec.index else pd.Series(np.nan, index=sec.columns)  # noqa: E731
    rev, gp, oi, ni = g("매출"), g("매출총이익"), g("영업이익"), g("순이익")
    ocf, fcf, assets, eq = g("영업현금흐름"), g("잉여현금흐름"), g("총자산"), g("자기자본")
    liab = g("총부채").fillna(assets - eq)
    avg = lambda x: ((x + x.shift(1)) / 2).fillna(x)  # noqa: E731  평균잔액(전년 말과 올해 말의 평균)
    growth = lambda x: (x - x.shift(1)) / x.shift(1).abs() * 100  # noqa: E731
    fac_e = _split_factor(g("_filed_희석 EPS"), splits).reindex(sec.columns).fillna(1.0)
    fac_s = _split_factor(g("_filed_희석주식수"), splits).reindex(sec.columns).fillna(1.0)
    eps = g("희석 EPS") / fac_e
    shares = g("희석주식수") * fac_s
    price = pd.Series(np.nan, index=sec.columns)
    if isinstance(hist, pd.Series) and len(hist):
        h = hist.copy()
        h.index = pd.to_datetime(h.index)
        if h.index.tz is not None:
            h.index = h.index.tz_localize(None)
        for d in sec.columns:
            before = h[h.index <= d]
            if len(before) and (d - before.index[-1]).days <= 10:
                price[d] = before.iloc[-1]
    mcap = price * shares
    r = {
        "매출총이익률": gp / rev * 100, "영업이익률": oi / rev * 100, "순이익률": ni / rev * 100, "FCF 마진": fcf / rev * 100,
        "ROE": ni / avg(eq) * 100, "ROA": ni / avg(assets) * 100,
        "매출 성장률": growth(rev), "영업이익 성장률": growth(oi), "순이익 성장률": growth(ni), "EPS 성장률": growth(eps),
        "부채비율": liab / eq * 100, "자기자본비율": eq / assets * 100, "유동비율": g("유동자산") / g("유동부채") * 100,
        "이자보상배율": oi / g("이자비용"), "총자산회전율": rev / avg(assets), "영업현금흐름/순이익": ocf / ni,
        "배당성향": g("배당금") / ni * 100,
        "PER": (mcap / ni).where(ni > 0), "PBR": (mcap / eq).where(eq > 0), "PSR": mcap / rev,
    }
    out = pd.DataFrame(r).T
    return out.replace([np.inf, -np.inf], np.nan)


DAILY = {  # 이름: 단위
    "주가": "$", "12M 트레일링 PER": "배", "12M 선행 PER (사후 실현 EPS)": "배",
    "현재 12M 선행 PER (컨센서스)": "배", "트레일링 PER 평균 ±1σ 밴드": "배", "PER (직전 사업연도 EPS)": "배",
    "12M 선행 PER (컨센서스 추이)": "배",
}
try:  # 일별 밸류에이션 배수 (valuation.py, SEC 분기 공시 기준)
    from valuation import VAL as _VAL
    DAILY.update({k: v[0] for k, v in _VAL.items()})
except Exception:  # noqa: BLE001
    _VAL = {}


_PER_LABEL = {  # 목록에 보이는 이름 (안쪽 키는 그대로)
    "PER (직전 사업연도 EPS)": "PER · 일별 (그날 주가 ÷ 그때 이미 발표된 직전 사업연도 EPS)",
    "12M 트레일링 PER": "12M PER · 일별 (그날 주가 ÷ 그때까지 발표된 최근 4개 분기 EPS 합)",
    "PER (시가총액 ÷ 순이익 TTM)": "PER · 일별 (그날 시가총액 ÷ 그때까지 발표된 1년치 순이익)",
    "12M 선행 PER (사후 실현 EPS)": "선행 PER · 일별 (그날 주가 ÷ 그 뒤 1년 동안 실제로 나온 EPS)",
    "트레일링 PER 평균 ±1σ 밴드": "PER 평균 ±1σ 밴드",
    "현재 12M 선행 PER (컨센서스)": "오늘 선행 PER (애널리스트 예상 EPS 기준, 점 하나)",
    "12M 선행 PER (컨센서스 추이)": "12M 선행 PER · 일별 (그날 주가 ÷ 그때 애널리스트가 예상한 향후 4개 분기 EPS 합)",
}
_PER_TRACE = {"12M 선행 PER (컨센서스 추이)": "12M 선행 PER", "12M 트레일링 PER": "12M PER", "PER (시가총액 ÷ 순이익 TTM)": "PER", "12M 선행 PER (사후 실현 EPS)": "선행 PER(사후)"}


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner="한국어로 번역하는 중...")
def translate_ko(text: str):
    """Google 번역(키 없는 공개 엔드포인트)으로 한국어 번역. 실패하면 None."""
    import json
    import urllib.parse
    import urllib.request

    if not text or not text.strip():
        return None
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=ko&dt=t"
    for i in range(3):
        try:
            data = urllib.parse.urlencode({"q": text}).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                j = json.loads(resp.read().decode("utf-8"))
            out = "".join(seg[0] for seg in j[0] if seg and seg[0])
            if out.strip():
                return out
        except Exception:  # noqa: BLE001
            time.sleep(1.0 + i)
    return None


# ----------------------------------------------------------------------
# 표시 도우미
# ----------------------------------------------------------------------
def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:  # noqa: BLE001
        return None


def big(x, cur="$"):
    v = _num(x)
    if v is None:
        return "-"
    if cur == "₩":
        import money
        return money.big(v, "KRW")
    a = abs(v)
    for unit, d in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if a >= d:
            return f"{'-' if v < 0 else ''}{cur}{a / d:,.2f}{unit}"
    return f"{'-' if v < 0 else ''}{cur}{a:,.2f}"


def pct(x, nd=1):
    v = _num(x)
    return "-" if v is None else f"{v * 100:.{nd}f}%"


def ratio(x, nd=2):
    v = _num(x)
    return "-" if v is None else f"{v:,.{nd}f}"


def _row(df: pd.DataFrame, *names):
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def ntm_eps(info: dict, est: pd.DataFrame):
    """12개월 선행(NTM) EPS = 이번 회계연도(0y)와 다음 회계연도(+1y) 컨센서스를 남은 기간 비율로 가중평균."""
    try:
        e0, e1 = float(est.loc["0y", "avg"]), float(est.loc["+1y", "avg"])
        fye = _num(info.get("nextFiscalYearEnd")) or _num(info.get("lastFiscalYearEnd"))
        if not fye:
            return None, None
        end = pd.Timestamp(int(fye), unit="s")
        while end < pd.Timestamp.now():
            end += pd.DateOffset(years=1)
        w = min(max((end - pd.Timestamp.now()).days / 365.0, 0.0), 1.0)
        return w * e0 + (1 - w) * e1, (e0, e1, w, end)
    except Exception:  # noqa: BLE001
        return None, None


def _div_yield(info: dict):
    px = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    rate = _num(info.get("dividendRate"))
    if px and rate:
        return rate / px
    y = _num(info.get("trailingAnnualDividendYield"))
    return y


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
STALE_EPS_DAYS = 490  # 회계연도 말에서 이만큼 지난 EPS로는 PER을 내지 않는다 (다음 연간 공시가 늦어도 보통 15개월 안)


def annual_eps_table(sec, splits):
    """연간 희석 EPS 표: 공시일(그 숫자를 처음 알 수 있게 된 날) 순서, 열 = EPS(액면분할 반영)·회계연도 말·분할계수."""
    if not isinstance(sec, pd.DataFrame) or "희석 EPS" not in sec.index:
        return None
    eps = pd.to_numeric(sec.loc["희석 EPS"], errors="coerce")
    if "_filed_희석 EPS" in sec.index:
        filed = pd.to_numeric(sec.loc["_filed_희석 EPS"], errors="coerce")
    else:
        filed = pd.Series([float((pd.Timestamp(c) + pd.Timedelta(days=90)).toordinal()) for c in sec.columns], index=sec.columns)
    filed = filed.fillna(pd.Series([float((pd.Timestamp(c) + pd.Timedelta(days=90)).toordinal()) for c in sec.columns], index=sec.columns))
    fac = _split_factor(filed, splits).reindex(sec.columns).fillna(1.0)  # 값의 기준(최신 제출일) 뒤 액면분할만 보정
    eps = eps / fac
    avail = filed
    if "_avail_희석 EPS" in sec.index:  # 처음 공시된 날부터 그 EPS를 쓴다 (나중에 다시 실린 날이 아니라)
        avail = pd.to_numeric(sec.loc["_avail_희석 EPS"], errors="coerce").fillna(filed)
    tb = pd.DataFrame({"EPS": eps.values, "회계연도말": pd.to_datetime(list(sec.columns)), "분할계수": fac.values},
                      index=[pd.Timestamp.fromordinal(int(f)) for f in avail.values]).dropna(subset=["EPS"])
    tb = tb[~tb.index.duplicated(keep="last")].sort_index()
    return tb if len(tb) else None


def _annual_eps_points(sec, splits):
    """연간 재무제표의 희석 EPS를 '공시된 날' 기준으로 늘어놓는다 (액면분할 반영). (공시일 → EPS)."""
    tb = annual_eps_table(sec, splits)
    return None if tb is None else tb["EPS"]


def annual_per_series(hist, sec, splits, max_age: int = STALE_EPS_DAYS):
    """일별 PER = 그날 주가 ÷ 그날까지 공시된 가장 최근 사업연도 EPS (적자면 비움).
    그날 쓰는 EPS의 회계연도 말이 max_age일보다 오래됐으면(공시일 오류·자료 누락) 틀린 값 대신 비운다."""
    tb = annual_eps_table(sec, splits)
    if tb is None or not isinstance(hist, pd.Series) or not len(hist):
        return None
    pts = tb["EPS"]
    h = hist.copy()
    h.index = pd.to_datetime(h.index)
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    h = h[h.index >= pts.index[0]]
    if not len(h):
        return None
    pos = np.searchsorted(pts.index.values, h.index.values, side="right") - 1
    e = np.where(pos >= 0, pts.values[np.clip(pos, 0, None)], np.nan)
    fy = tb["회계연도말"].values[np.clip(pos, 0, None)]
    stale = (h.index.values - fy) > np.timedelta64(int(max_age), "D")
    e = np.where(stale, np.nan, e)
    out = pd.Series(np.where(e > 0, h.values / e, np.nan), index=h.index)
    return out if out.notna().sum() > 5 else None


def annual_pe_now(px, sec, splits):
    """오늘 PER (오늘 주가 ÷ 이미 공시된 직전 사업연도 EPS)."""
    tb = annual_eps_table(sec, splits)
    if tb is None or not px:
        return None
    tb = tb[tb.index <= pd.Timestamp.today()]
    if not len(tb) or not tb["EPS"].iloc[-1] > 0:
        return None
    if (pd.Timestamp.today() - tb["회계연도말"].iloc[-1]).days > STALE_EPS_DAYS:  # 너무 오래된 EPS → 12M PER로 대신
        return None
    return px / float(tb["EPS"].iloc[-1])


def per_series(hist: pd.Series, edates: pd.DataFrame):
    """분기 실적(발표일·발표 EPS)으로 일별 12M PER 두 가지를 만든다.
    트레일링: 주가 ÷ 그날까지 발표된 최근 4개 분기 EPS 합.
    선행(사후): 주가 ÷ 그날 이후 발표된 4개 분기 EPS 합 (실제로 나온 이익 기준이라 최근 1년은 계산 불가)."""
    if not isinstance(edates, pd.DataFrame) or edates.empty or "Reported EPS" not in edates or not len(hist):
        return None
    e = edates[["Reported EPS"]].dropna().copy()
    idx = pd.to_datetime(e.index)
    e.index = idx.tz_localize(None) if idx.tz is not None else idx
    e = e[~e.index.duplicated()].sort_index()["Reported EPS"].astype(float)
    if len(e) < 4:
        return None
    h = hist.copy()
    h.index = pd.to_datetime(h.index)
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    h = h[h.index >= e.index[3]]
    ttm = e.rolling(4).sum()                       # 발표일 기준 최근 4분기 합
    fwd = e[::-1].rolling(4).sum()[::-1].shift(-1)  # 다음 발표부터 4분기 합
    # 분기 실적이 빠져 '4개 분기'가 1년 넘게 흩어져 있으면 12M EPS가 아니므로 비운다 (보통 첫·넷째 발표 간격 ≈ 270일)
    span = pd.Series(e.index, index=e.index).diff(3).dt.days
    ttm[span > 400] = np.nan
    fwd[span.shift(-4) > 400] = np.nan
    pos = np.searchsorted(e.index.values, h.index.values, side="right") - 1
    t_eps = np.where(pos >= 0, ttm.values[np.clip(pos, 0, None)], np.nan)
    f_eps = np.where(pos >= 0, fwd.values[np.clip(pos, 0, None)], np.nan)
    trail = pd.Series(np.where(t_eps > 0, h.values / t_eps, np.nan), index=h.index)
    fwd_r = pd.Series(np.where(f_eps > 0, h.values / f_eps, np.nan), index=h.index)
    return trail, fwd_r, e


def _naive_idx(x):
    idx = pd.to_datetime(x.index)
    x = x.copy()
    x.index = idx.tz_localize(None) if idx.tz is not None else idx
    return x[~x.index.duplicated(keep="last")].sort_index()


def _trend_ntm_points(trend, info):
    """Yahoo eps_trend(이번·다음 회계연도 컨센서스의 현재·7·30·60·90일 전 값) → (날짜, 그날의 NTM EPS) 목록."""
    if not isinstance(trend, pd.DataFrame) or trend.empty or "0y" not in trend.index or "+1y" not in trend.index:
        return []
    fye = _num((info or {}).get("nextFiscalYearEnd")) or _num((info or {}).get("lastFiscalYearEnd"))
    if not fye:
        return []
    today = pd.Timestamp.now().normalize()
    end = pd.Timestamp(int(fye), unit="s").normalize()
    while end < today:
        end += pd.DateOffset(years=1)
    pts = []
    for col, lag in (("90daysAgo", 90), ("60daysAgo", 60), ("30daysAgo", 30), ("7daysAgo", 7), ("current", 0)):
        if col not in trend.columns:
            continue
        e0, e1 = _num(trend.loc["0y", col]), _num(trend.loc["+1y", col])
        if e0 is None or e1 is None:
            continue
        t = today - pd.Timedelta(days=lag)
        w = min(max((end - t).days / 365.0, 0.0), 1.0)  # 그날 기준 이번 회계연도에 남은 비중
        pts.append((t, w * e0 + (1 - w) * e1))
    return pts


def fwd_consensus_series(hist, edates, trend=None, info=None, n_eps=None):
    """일별 12M 선행 PER (컨센서스) = 그날 주가 ÷ 그날 기준 '앞으로 4개 분기' 예상 EPS 합.
    - 최근 90일: Yahoo 컨센서스 추이(eps_trend: 현재·7·30·60·90일 전)로 계산한 NTM EPS를 날짜별로 이어 붙인 값 (실제 컨센서스).
    - 그 이전: 각 분기 실적 발표 직전의 컨센서스(EPS Estimate)를 다음 4개 분기만큼 더한 근사치.
      아직 컨센서스가 없는 먼 분기는 오늘의 NTM EPS ÷ 4로 채운다.
    돌려주는 값: (PER 시리즈, 실제 컨센서스 구간 시작일 또는 None)."""
    if not isinstance(hist, pd.Series) or not len(hist):
        return None
    h = _naive_idx(hist.dropna().astype(float))
    today = pd.Timestamp.now().normalize()
    ntm = pd.Series(np.nan, index=h.index)
    if isinstance(edates, pd.DataFrame) and not edates.empty and "EPS Estimate" in edates:
        est = _naive_idx(pd.to_numeric(edates["EPS Estimate"], errors="coerce"))
        est.index = est.index.normalize()
        est = est[~est.index.duplicated(keep="last")]
        if len(est) >= 4:
            fill = (n_eps / 4.0) if n_eps and n_eps > 0 else np.nan
            est[(est.index > today) & est.isna()] = fill
            last = est.index[-1]
            extra = pd.Series(fill, index=[last + pd.Timedelta(days=91 * k) for k in range(1, 5)])
            q = pd.concat([est, extra])
            nxt4 = q[::-1].rolling(4).sum()[::-1].shift(-1)  # 각 발표일 다음부터 4개 분기 예상 합
            dates = pd.Series(q.index, index=q.index)
            span = (dates.shift(-4) - dates.shift(-1)).dt.days
            nxt4[span > 400] = np.nan  # 분기가 빠져 1년 넘게 흩어진 경우
            pos = np.searchsorted(q.index.values, h.index.values, side="right") - 1
            v = np.where(pos >= 0, nxt4.values[np.clip(pos, 0, None)], np.nan)
            ntm = pd.Series(v, index=h.index)
    exact_from = None
    pts = _trend_ntm_points(trend, info)
    if len(pts) >= 2:
        xs = np.array([p[0].toordinal() for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        m = h.index >= pts[0][0]
        if m.any():
            ntm[m] = np.interp(np.array([d.toordinal() for d in h.index[m]], dtype=float), xs, ys)
            exact_from = pts[0][0]
    out = pd.Series(np.where(ntm > 0, h.values / ntm.values, np.nan), index=h.index)
    if out.notna().sum() < 20:
        return None
    return out, exact_from


def _set_research(tk):
    st.session_state["rs_ticker"] = tk


def _send_to_options(tk):
    st.session_state["opt_ticker"] = tk


def _financials(ticker: str, R: dict, domestic: bool, code: str | None):
    """연간 재무제표 (행: 매출·영업이익… / 열: 회계연도 말)와 출처. 미국: SEC 10-K → Yahoo, 국내: DART → Yahoo."""
    import kr

    errs = []
    if domestic:
        key = kr.dart_key()
        if key:
            try:
                F = kr.load_dart_financials(code, key)
                if F.shape[1] >= 2:
                    return F, "DART 사업보고서", errs
            except Exception as e:  # noqa: BLE001
                errs.append(f"DART: {e}")
    else:
        try:
            F = load_sec_financials(ticker)
            if F.shape[1] >= 2:
                return F, "SEC 10-K", errs
        except Exception as e:  # noqa: BLE001
            errs.append(f"SEC: {e}")
    info = R.get("info") or {}
    fx = 1.0
    fin_cur, px_cur = info.get("financialCurrency") or info.get("currency"), info.get("currency")
    if fin_cur and px_cur and fin_cur != px_cur:
        try:
            from dcf import load_fx

            fx = load_fx(fin_cur, px_cur) or 1.0
        except Exception:  # noqa: BLE001
            fx = 1.0
    F = kr.yahoo_frame(R, fx)
    return F, "Yahoo", errs


def _last_pos(F, row):
    """연간 재무제표 F에서 row의 가장 최근 값 (없으면 None)."""
    try:
        if F is None or row not in F.index:
            return None
        s = pd.to_numeric(F.loc[row], errors="coerce").dropna()
        return float(s.iloc[-1]) if len(s) else None
    except Exception:  # noqa: BLE001
        return None


def _pe_pb(info: dict, px, F=None, R: dict | None = None):
    """오늘의 PER·PBR. Yahoo 값이 비어 있으면 (현재가 ÷ 최근 4개 분기 발표 EPS 합, 그래프와 같은 기준)
    → (현재가 ÷ EPS) → (시가총액 ÷ 순이익) → 재무제표 순으로 직접 계산."""
    mcap = _num(info.get("marketCap"))
    pe = _num(info.get("trailingPE"))
    if (not pe or pe <= 0) and R is not None and px:
        try:
            per = per_series(R.get("hist"), R.get("edates"))
            if per is not None:
                ttm = float(per[2].rolling(4).sum().iloc[-1])
                pe = px / ttm if ttm > 0 else None
        except Exception:  # noqa: BLE001
            pe = None
    if not pe or pe <= 0:
        pe = None
        eps = _num(info.get("trailingEps"))
        ni = _num(info.get("netIncomeToCommon"))
        if px and eps and eps > 0:
            pe = px / eps
        elif mcap and ni and ni > 0:
            pe = mcap / ni
        else:
            ni = _last_pos(F, "순이익")
            if mcap and ni and ni > 0:
                pe = mcap / ni
    pb = _num(info.get("priceToBook"))
    if not pb or pb <= 0:
        pb = None
        bv = _num(info.get("bookValue"))
        if px and bv and bv > 0:
            pb = px / bv
        else:
            eq = _last_pos(F, "자기자본")
            if mcap and eq and eq > 0:
                pb = mcap / eq
    return pe, pb

__all__ = [n for n in dir() if not n.startswith("__")]  # 밑줄로 시작하는 도우미도 다른 모듈에서 쓸 수 있게
