"""국내 종목 리서치 지원
- 종목 찾기: KRX 상장 종목 목록(FinanceDataReader)에서 이름·6자리 코드로 찾고 Yahoo 티커(.KS 코스피 / .KQ 코스닥)로 바꾼다.
- 재무제표: OpenDART API 키가 있으면 DART 사업보고서(최대 10년, 연결 기준), 없으면 Yahoo 연간 재무제표(약 4년).
  둘 다 SEC 재무제표와 같은 행 이름(매출·영업이익·순이익…)으로 맞춰 리서치 탭의 다른 화면(재무비율·DCF·신용위험)이 그대로 쓴다.
- 지분: DART 최대주주 현황·임원/주요주주 소유보고·5% 대량보유 보고.
"""
import io
import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

DART = "https://opendart.fss.or.kr/api/"


# ----------------------------------------------------------------------
# 종목 목록
# ----------------------------------------------------------------------
@st.cache_data(ttl=24 * 60 * 60, show_spinner="국내 종목 목록을 불러오는 중...")
def load_krx() -> pd.DataFrame:
    import FinanceDataReader as fdr

    df = fdr.StockListing("KRX")
    if "Symbol" in df.columns and "Code" not in df.columns:
        df = df.rename(columns={"Symbol": "Code"})
    keep = [c for c in ["Code", "Name", "Market", "Marcap", "Stocks"] if c in df.columns]
    df = df[keep].dropna(subset=["Code", "Name"]).copy()
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    try:  # 업종(산업) 정보
        desc = fdr.StockListing("KRX-DESC")
        if "Symbol" in desc.columns and "Code" not in desc.columns:
            desc = desc.rename(columns={"Symbol": "Code"})
        desc["Code"] = desc["Code"].astype(str).str.zfill(6)
        cols = [c for c in ("Code", "Sector", "Industry") if c in desc.columns]
        df = df.merge(desc[cols], on="Code", how="left")
    except Exception:  # noqa: BLE001
        pass
    for c in ("Sector", "Industry", "Marcap", "Market"):
        if c not in df.columns:
            df[c] = np.nan
    return df.reset_index(drop=True)


def is_kr_query(q: str) -> bool:
    q = (q or "").strip()
    return bool(re.fullmatch(r"\d{6}(\.K[SQ])?", q.upper())) or bool(re.search(r"[가-힣]", q))


def yahoo_symbol(code: str, market: str) -> str:
    m = str(market or "").upper()
    return f"{code}.KQ" if m.startswith("KOSDAQ") else f"{code}.KS"


def resolve(q: str, listing: pd.DataFrame | None = None) -> list:
    """이름 또는 6자리 코드 → 후보 목록 [{code, name, market, yahoo, sector, industry}] (시가총액 큰 순)."""
    q = (q or "").strip()
    if not q:
        return []
    df = listing if listing is not None else load_krx()
    code = q.upper().replace(".KS", "").replace(".KQ", "")
    if re.fullmatch(r"\d{6}", code):
        hit = df[df["Code"] == code]
    else:
        exact = df[df["Name"] == q]
        part = df[df["Name"].str.contains(q, case=False, regex=False, na=False)]
        hit = pd.concat([exact, part.sort_values("Marcap", ascending=False)]).drop_duplicates("Code")
    out = []
    for r in hit.head(15).itertuples():
        out.append({"code": r.Code, "name": r.Name, "market": str(r.Market), "yahoo": yahoo_symbol(r.Code, r.Market),
                    "sector": None if pd.isna(getattr(r, "Sector", np.nan)) else str(r.Sector),
                    "industry": None if pd.isna(getattr(r, "Industry", np.nan)) else str(r.Industry),
                    "marcap": float(r.Marcap) if pd.notna(r.Marcap) else np.nan})
    return out


# ----------------------------------------------------------------------
# 금리
# ----------------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def kr_rate(tenor: str = "10Y"):
    """국고채 금리 (소수, 기준일, 이름). 10년물을 못 받으면 3년물."""
    import FinanceDataReader as fdr

    start = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")
    for sym, nm in ((f"KR{tenor}T=RR", f"국고채 {tenor.replace('Y', '년')}"), ("KR3YT=RR", "국고채 3년")):
        try:
            s = fdr.DataReader(sym, start)["Close"].dropna()
            if len(s):
                return float(s.iloc[-1]) / 100, str(pd.Timestamp(s.index[-1]).date()), nm
        except Exception:  # noqa: BLE001
            pass
    try:
        from rates import load_rate

        s = load_rate("KR3Y", start, datetime.now().strftime("%Y-%m-%d"))
        if s is not None and len(s):
            return float(s.iloc[-1]) / 100, str(pd.Timestamp(s.index[-1]).date()), "국고채 3년"
    except Exception:  # noqa: BLE001
        pass
    return None


# ----------------------------------------------------------------------
# Yahoo 연간 재무제표 → SEC와 같은 행 이름 (국내 종목, SEC에 없는 해외 종목 공통)
# ----------------------------------------------------------------------
Y_MAP = [  # (행 이름, 표, Yahoo 항목 후보, 부호 반전·절댓값 여부)
    ("매출", "income", ["Total Revenue", "Operating Revenue"], False),
    ("매출총이익", "income", ["Gross Profit"], False),
    ("영업이익", "income", ["Operating Income", "Total Operating Income As Reported"], False),
    ("순이익", "income", ["Net Income Common Stockholders", "Net Income", "Net Income From Continuing Operation Net Minority Interest"], False),
    ("희석 EPS", "income", ["Diluted EPS", "Basic EPS"], False),
    ("이자비용", "income", ["Interest Expense", "Interest Expense Non Operating"], True),
    ("희석주식수", "income", ["Diluted Average Shares", "Basic Average Shares"], False),
    ("감가상각비", "cashflow", ["Depreciation And Amortization", "Reconciled Depreciation", "Depreciation Amortization Depletion"], True),
    ("영업현금흐름", "cashflow", ["Operating Cash Flow"], False),
    ("설비투자", "cashflow", ["Capital Expenditure", "Purchase Of PPE"], True),
    ("잉여현금흐름", "cashflow", ["Free Cash Flow"], False),
    ("배당금", "cashflow", ["Cash Dividends Paid", "Common Stock Dividend Paid"], True),
    ("총자산", "balance", ["Total Assets"], False),
    ("총부채", "balance", ["Total Liabilities Net Minority Interest"], False),
    ("자기자본", "balance", ["Stockholders Equity", "Common Stock Equity"], False),
    ("유동자산", "balance", ["Current Assets"], False),
    ("유동부채", "balance", ["Current Liabilities"], False),
    ("이익잉여금", "balance", ["Retained Earnings"], False),
    ("단기차입금", "balance", ["Current Debt", "Current Debt And Capital Lease Obligation"], False),
    ("장기차입금", "balance", ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], False),
    ("현금성자산", "balance", ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"], False),
]


def yahoo_frame(R: dict, fx: float = 1.0) -> pd.DataFrame | None:
    """Yahoo 연간 재무제표 → 행: 항목, 열: 회계연도 말. 공시일은 회계연도 말 + 90일로 가정."""
    rows = {}
    for nm, tb, keys, absval in Y_MAP:
        df = R.get(tb)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        for k in keys:
            if k in df.index:
                v = pd.to_numeric(df.loc[k], errors="coerce")
                if v.notna().any():
                    rows[nm] = v.abs() if absval else v
                    break
    if "매출" not in rows and "순이익" not in rows:
        return None
    out = pd.DataFrame(rows).T
    out.columns = pd.to_datetime(out.columns)
    out = out[sorted(out.columns)].astype(float)
    out = out.loc[:, out.notna().sum() >= 3]  # 대부분 비어 있는 오래된 연도는 뺌
    if out.shape[1] == 0:
        return None
    if "잉여현금흐름" not in out.index and "영업현금흐름" in out.index:
        out.loc["잉여현금흐름"] = out.loc["영업현금흐름"] - out.loc["설비투자"].fillna(0) if "설비투자" in out.index else np.nan
    per_share = {"희석 EPS"}
    for nm in out.index:
        if nm not in per_share and nm != "희석주식수":
            out.loc[nm] = out.loc[nm] * fx
    if "희석 EPS" in out.index:
        out.loc["희석 EPS"] = out.loc["희석 EPS"] * fx
    filed = pd.Series([float((c + pd.Timedelta(days=90)).toordinal()) for c in out.columns], index=out.columns)
    out.loc["_filed_희석 EPS"] = filed
    out.loc["_filed_희석주식수"] = filed
    return out


# ----------------------------------------------------------------------
# OpenDART
# ----------------------------------------------------------------------
def dart_key() -> str | None:
    k = st.session_state.get("dart_key_input") or None
    if k:
        return k.strip()
    try:
        k = st.secrets.get("DART_API_KEY")
    except Exception:  # noqa: BLE001
        k = None
    return (k or "").strip() or None


def _dart_json(api: str, params: dict, key: str) -> dict:
    url = DART + api + ".json?" + urllib.parse.urlencode(dict(params, crtfc_key=key))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner="DART 기업 코드 목록을 받는 중...")
def dart_corp_map(key: str) -> dict:
    req = urllib.request.Request(DART + "corpCode.xml?crtfc_key=" + urllib.parse.quote(key), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    if not raw.startswith(b"PK"):
        try:
            msg = json.loads(raw.decode("utf-8")).get("message")
        except Exception:  # noqa: BLE001
            msg = raw[:200].decode("utf-8", "ignore")
        raise RuntimeError(f"DART 응답 오류: {msg}")
    import xml.etree.ElementTree as ET

    z = zipfile.ZipFile(io.BytesIO(raw))
    root = ET.fromstring(z.read(z.namelist()[0]))
    out = {}
    for el in root.iter("list"):
        sc = (el.findtext("stock_code") or "").strip()
        if sc:
            out[sc] = (el.findtext("corp_code") or "").strip()
    return out


# (행 이름, 재무제표 구분, account_id 후보, 계정명 후보, 절댓값 여부)
D_MAP = [
    ("매출", ("IS", "CIS"), ["ifrs-full_Revenue", "ifrs_Revenue"], ["매출액", "수익(매출액)", "영업수익", "매출"], False),
    ("매출총이익", ("IS", "CIS"), ["ifrs-full_GrossProfit", "ifrs_GrossProfit"], ["매출총이익"], False),
    ("영업이익", ("IS", "CIS"), ["dart_OperatingIncomeLoss"], ["영업이익", "영업이익(손실)"], False),
    ("순이익", ("IS", "CIS"), ["ifrs-full_ProfitLossAttributableToOwnersOfParent", "ifrs_ProfitLossAttributableToOwnersOfParent",
                              "ifrs-full_ProfitLoss", "ifrs_ProfitLoss"], ["지배기업의 소유주에게 귀속되는 당기순이익", "당기순이익"], False),
    ("희석 EPS", ("IS", "CIS"), ["ifrs-full_DilutedEarningsLossPerShare", "ifrs-full_BasicEarningsLossPerShare",
                                "ifrs_DilutedEarningsLossPerShare", "ifrs_BasicEarningsLossPerShare"], ["희석주당이익", "기본주당이익"], False),
    ("이자비용", ("CF", "IS", "CIS"), ["ifrs-full_InterestPaidClassifiedAsOperatingActivities", "ifrs-full_InterestPaidClassifiedAsFinancingActivities",
                                     "ifrs-full_InterestExpense"], ["이자의 지급", "이자의지급", "이자지급", "이자비용"], True),
    ("영업현금흐름", ("CF",), ["ifrs-full_CashFlowsFromUsedInOperatingActivities", "ifrs_CashFlowsFromUsedInOperatingActivities"],
     ["영업활동현금흐름", "영업활동으로 인한 현금흐름"], False),
    ("설비투자", ("CF",), ["ifrs-full_PurchaseOfPropertyPlantAndEquipment", "ifrs_PurchaseOfPropertyPlantAndEquipment"],
     ["유형자산의 취득", "유형자산 취득"], True),
    ("배당금", ("CF",), ["ifrs-full_DividendsPaidClassifiedAsFinancingActivities", "ifrs_DividendsPaidClassifiedAsFinancingActivities"],
     ["배당금의 지급", "배당금 지급", "배당금지급"], True),
    ("감가상각비", ("CF",), ["ifrs-full_AdjustmentsForDepreciationExpense", "ifrs-full_AdjustmentsForDepreciationAndAmortisationExpense"],
     ["감가상각비", "유형자산감가상각비"], True),
    ("총자산", ("BS",), ["ifrs-full_Assets", "ifrs_Assets"], ["자산총계"], False),
    ("총부채", ("BS",), ["ifrs-full_Liabilities", "ifrs_Liabilities"], ["부채총계"], False),
    ("자기자본", ("BS",), ["ifrs-full_EquityAttributableToOwnersOfParent", "ifrs_EquityAttributableToOwnersOfParent", "ifrs-full_Equity",
                          "ifrs_Equity"], ["지배기업의 소유주에게 귀속되는 자본", "자본총계"], False),
    ("유동자산", ("BS",), ["ifrs-full_CurrentAssets", "ifrs_CurrentAssets"], ["유동자산"], False),
    ("유동부채", ("BS",), ["ifrs-full_CurrentLiabilities", "ifrs_CurrentLiabilities"], ["유동부채"], False),
    ("이익잉여금", ("BS",), ["ifrs-full_RetainedEarnings", "ifrs_RetainedEarnings"], ["이익잉여금", "이익잉여금(결손금)"], False),
    ("현금성자산", ("BS",), ["ifrs-full_CashAndCashEquivalents", "ifrs_CashAndCashEquivalents"], ["현금및현금성자산"], False),
]
DEBT_ST = ["ifrs-full_ShortTermBorrowings", "dart_ShortTermBorrowings", "ifrs-full_CurrentPortionOfLongtermBorrowings",
           "dart_CurrentPortionOfBonds", "ifrs-full_CurrentPortionOfNoncurrentBondsIssued"]
DEBT_LT = ["ifrs-full_LongtermBorrowings", "dart_LongTermBorrowings", "dart_BondsIssued", "ifrs-full_NoncurrentPortionOfNoncurrentBondsIssued",
           "ifrs-full_BondsIssued"]


def _amt(x):
    try:
        s = str(x).replace(",", "").strip()
        if s in ("", "-"):
            return np.nan
        return float(s)
    except Exception:  # noqa: BLE001
        return np.nan


def parse_dart_year(lst: list, col: str = "thstrm_amount") -> dict:
    """fnlttSinglAcntAll 한 해 응답 → {행 이름: 값}. col="frmtrm_amount"면 같은 보고서에 다시 실린 전기(직전 연도) 값."""
    out = {}
    for nm, sjs, ids, names, absval in D_MAP:
        val = np.nan
        for want in ids:
            for e in lst:
                if e.get("sj_div") in sjs and e.get("account_id") == want:
                    val = _amt(e.get(col))
                    if np.isfinite(val):
                        break
            if np.isfinite(val):
                break
        if not np.isfinite(val):
            for e in lst:
                if e.get("sj_div") in sjs and str(e.get("account_nm", "")).replace(" ", "") in [n.replace(" ", "") for n in names]:
                    val = _amt(e.get(col))
                    if np.isfinite(val):
                        break
        if np.isfinite(val):
            out[nm] = abs(val) if absval else val
    if "이자비용" not in out:  # 이자 지급액이 없으면 금융비용(이자 외 항목 포함)으로 대신
        for e in lst:
            if e.get("sj_div") in ("IS", "CIS") and (e.get("account_id") == "ifrs-full_FinanceCosts" or
                                                     str(e.get("account_nm", "")).replace(" ", "") in ("금융비용", "금융원가")):
                v = _amt(e.get(col))
                if np.isfinite(v):
                    out["이자비용"] = abs(v)
                    break
    for nm, ids in (("단기차입금", DEBT_ST), ("장기차입금", DEBT_LT)):
        vals = [_amt(e.get(col)) for e in lst if e.get("sj_div") == "BS" and e.get("account_id") in ids]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[nm] = float(np.sum(vals))
    rc = next((e.get("rcept_no") for e in lst if e.get("rcept_no")), None)
    out["_rcept"] = rc
    return out


@st.cache_data(ttl=24 * 60 * 60, show_spinner="DART 사업보고서에서 10개년 재무제표를 불러오는 중...")
def load_dart_financials(code: str, key: str, years: int = 10) -> pd.DataFrame:
    corp = dart_corp_map(key).get(code)
    if not corp:
        raise RuntimeError("DART에서 이 종목의 기업 코드를 찾지 못했습니다")
    this = datetime.now().year
    yrs = list(range(this - years, this))

    def one(y):
        for fs in ("CFS", "OFS"):  # 연결 → 별도
            for _ in range(2):
                try:
                    j = _dart_json("fnlttSinglAcntAll", {"corp_code": corp, "bsns_year": str(y), "reprt_code": "11011", "fs_div": fs}, key)
                    if j.get("status") == "000" and j.get("list"):
                        d = parse_dart_year(j["list"])
                        d["_prev"] = parse_dart_year(j["list"], "frmtrm_amount")  # 다음 해 보고서에 다시 실린(수정된) 전년도 값
                        return y, d
                    if j.get("status") in ("010", "011", "012", "020", "901"):  # 키 오류·사용 한도
                        raise RuntimeError(j.get("message"))
                    break
                except RuntimeError:
                    raise
                except Exception:  # noqa: BLE001
                    time.sleep(0.6)
        return y, None

    with ThreadPoolExecutor(max_workers=5) as ex:
        res = list(ex.map(one, yrs))
    cols, data, filed, avail = [], {}, {}, {}
    by_year = {y: d for y, d in res if d}
    for y, d in res:
        if not d:
            continue
        end = pd.Timestamp(f"{y}-12-31")
        cols.append(end)
        data[end] = d
        rc = d.get("_rcept")
        rdate = pd.Timestamp(rc[:8]) if rc and len(rc) >= 8 else end + pd.Timedelta(days=90)
        filed[end] = float(rdate.toordinal())
        # 정정 사업보고서면 접수일이 늦어지므로, '처음 알 수 있던 날'은 제출 기한(회계연도 말 + 90일) 안으로 본다
        avail[end] = float(min(rdate, end + pd.Timedelta(days=100)).toordinal())
        # 다음 해 보고서에 전기 값이 다시 실려 있으면(소급 수정·액면분할 반영) 그 값을 쓴다. 값의 기준일은 다음 해 보고서 접수일.
        nxt = by_year.get(y + 1)
        if nxt and isinstance(nxt.get("_prev"), dict):
            changed = False
            for k, v in nxt["_prev"].items():
                if k.startswith("_") or v is None or not np.isfinite(v):
                    continue
                if k not in d or not np.isfinite(d[k]) or abs(v - d[k]) > 1e-9 * max(1.0, abs(v)):
                    changed = True
                d[k] = v
            rc2 = nxt.get("_rcept")
            if rc2 and len(rc2) >= 8 and "희석 EPS" in nxt["_prev"]:
                filed[end] = float(pd.Timestamp(rc2[:8]).toordinal())
            d["_restated"] = changed
    if not cols:
        raise RuntimeError("DART 사업보고서 재무제표가 없습니다")
    rows = [r[0] for r in D_MAP] + ["단기차입금", "장기차입금"]
    df = pd.DataFrame(index=rows, columns=sorted(cols), dtype=float)
    for c in cols:
        for r in rows:
            v = data[c].get(r)
            if v is not None:
                df.loc[r, c] = v
    if df.loc["설비투자"].notna().any():
        df.loc["잉여현금흐름"] = df.loc["영업현금흐름"] - df.loc["설비투자"].fillna(0)
    # 주식 수 = 지배주주 순이익 ÷ EPS (DART 재무제표에는 주식 수가 없음)
    eps = df.loc["희석 EPS"]
    df.loc["희석주식수"] = (df.loc["순이익"] / eps).where(eps.abs() > 0)
    fl = pd.Series(filed).reindex(df.columns)
    df.loc["_filed_희석 EPS"] = fl
    df.loc["_filed_희석주식수"] = fl
    df.loc["_avail_희석 EPS"] = pd.Series(avail).reindex(df.columns)
    return df.dropna(how="all")


def annual_quarterly_dict(F: pd.DataFrame, shares_now: float | None = None, splits=None) -> dict:
    """연간 재무제표 → valuation.daily_valuation에 넣을 형식(항목별 값·공시일). 연간 값을 그대로 최근 12개월 값으로 쓴다.
    값은 나중에 수정된 값을 쓰되, 반영 시점은 그 해 숫자가 처음 공시된 날(_avail_)로 잡는다."""
    out = {}
    basis = F.loc["_filed_희석 EPS"] if "_filed_희석 EPS" in F.index else pd.Series(
        [float((c + pd.Timedelta(days=90)).toordinal()) for c in F.columns], index=F.columns)
    filed = pd.to_numeric(F.loc["_avail_희석 EPS"], errors="coerce").fillna(basis) if "_avail_희석 EPS" in F.index else basis

    def mk(nm):
        if nm not in F.index:
            return pd.DataFrame(columns=["val", "filed"])
        s = F.loc[nm].dropna()
        return pd.DataFrame({"val": s.values.astype(float),
                             "filed": [pd.Timestamp.fromordinal(int(filed[c])) for c in s.index]}, index=s.index)

    for nm in ("매출", "영업이익", "순이익", "영업현금흐름", "감가상각비", "배당금", "자기자본", "단기차입금", "장기차입금", "현금성자산"):
        out[nm] = mk(nm)
    if "영업현금흐름" in F.index and "잉여현금흐름" in F.index:
        cap = (F.loc["영업현금흐름"] - F.loc["잉여현금흐름"]).dropna()
        out["설비투자"] = pd.DataFrame({"val": cap.values, "filed": [pd.Timestamp.fromordinal(int(filed[c])) for c in cap.index]}, index=cap.index)
    else:
        out["설비투자"] = pd.DataFrame(columns=["val", "filed"])
    out["단기투자"] = pd.DataFrame(columns=["val", "filed"])
    sh = mk("희석주식수")
    if len(sh) and splits is not None and len(splits):
        # 주식 수는 값의 기준일(수정된 보고서 접수일) 기준이다. daily_valuation은 반영일(처음 공시일) 뒤 분할만 곱하므로
        # 두 날 사이에 있었던 분할만큼 미리 나눠 둔다 (예: 분할 전 해의 주식 수가 다음 해 보고서에 분할 반영값으로 다시 실린 경우).
        from research_data import _split_factor
        fb = _split_factor(basis.reindex(sh.index), splits).reindex(sh.index).fillna(1.0)
        fa = _split_factor(filed.reindex(sh.index), splits).reindex(sh.index).fillna(1.0)
        sh["val"] = sh["val"].values * (fb / fa).values
    if shares_now and np.isfinite(shares_now):
        today = pd.Timestamp.now().normalize()
        sh = pd.concat([sh, pd.DataFrame({"val": [float(shares_now)], "filed": [today - pd.Timedelta(days=1)]}, index=[today - pd.Timedelta(days=1)])])
    out["주식수"] = sh
    return out


# ----------------------------------------------------------------------
# DART 지분 공시
# ----------------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner="DART 지분 공시를 불러오는 중...")
def load_dart_ownership(code: str, key: str) -> dict:
    corp = dart_corp_map(key).get(code)
    if not corp:
        raise RuntimeError("DART 기업 코드를 찾지 못했습니다")
    out = {}
    try:
        j = _dart_json("elestock", {"corp_code": corp}, key)
        out["elestock"] = pd.DataFrame(j.get("list") or [])
    except Exception:  # noqa: BLE001
        out["elestock"] = pd.DataFrame()
    try:
        j = _dart_json("majorstock", {"corp_code": corp}, key)
        out["majorstock"] = pd.DataFrame(j.get("list") or [])
    except Exception:  # noqa: BLE001
        out["majorstock"] = pd.DataFrame()
    out["hyslr"] = pd.DataFrame()
    for y in (datetime.now().year - 1, datetime.now().year - 2):
        try:
            j = _dart_json("hyslrSttus", {"corp_code": corp, "bsns_year": str(y), "reprt_code": "11011"}, key)
            if j.get("status") == "000" and j.get("list"):
                out["hyslr"] = pd.DataFrame(j["list"])
                out["hyslr_year"] = y
                break
        except Exception:  # noqa: BLE001
            pass
    return out


def _num_col(s):
    t = s.astype(str).str.strip()
    return pd.to_numeric(t.str.replace(",", "", regex=False).where(~t.isin(["-", "", "nan", "None"])), errors="coerce")


def render_kr_ownership(code: str, R: dict) -> None:
    """국내 종목 '내부자 거래·지분 구조' 탭."""
    from glossary import col_help
    from options import mgrid

    key = dart_key()
    info = R.get("info") or {}
    if not key:
        st.info("국내 종목의 최대주주·임원 지분 변동은 금융감독원 OpenDART 공시로 보여줍니다. 위 'OpenDART API 키'를 넣으면 볼 수 있습니다.")
        mgrid([("내부자 보유", f"{info['heldPercentInsiders'] * 100:.1f}%" if info.get("heldPercentInsiders") else "-"),
               ("기관 보유", f"{info['heldPercentInstitutions'] * 100:.1f}%" if info.get("heldPercentInstitutions") else "-"),
               ("발행주식수", f"{info['sharesOutstanding'] / 1e6:,.1f}백만주" if info.get("sharesOutstanding") else "-")], cols=3)
        return
    try:
        O = load_dart_ownership(code, key)
    except Exception as e:  # noqa: BLE001
        st.error(f"DART 지분 공시를 불러오지 못했습니다: {e}")
        return
    hy = O.get("hyslr")
    st.markdown("##### 최대주주 및 특수관계인 (사업보고서 기준)")
    if isinstance(hy, pd.DataFrame) and not hy.empty:
        h = hy.copy()
        rate = _num_col(h.get("trmend_posesn_stock_qota_rt", pd.Series(dtype=str)))
        cnt = _num_col(h.get("trmend_posesn_stock_co", pd.Series(dtype=str)))
        show = pd.DataFrame({"이름": h.get("nm"), "관계": h.get("relate"), "주식 종류": h.get("stock_knd"),
                             "보유 주식수": cnt.map(lambda v: f"{v:,.0f}" if np.isfinite(v) else "-"),
                             "지분율": rate.map(lambda v: f"{v:.2f}%" if np.isfinite(v) else "-")})
        common = h.get("stock_knd", pd.Series(dtype=str)).astype(str).str.contains("보통")
        is_total = h.get("nm", pd.Series(dtype=str)).astype(str).str.strip().isin(["계", "합계", "소계"])
        tot = rate[common & is_total].sum() if (common & is_total).any() else (rate[common].sum() if common.any() else np.nan)
        mgrid([("최대주주 지분", f"{tot:.2f}%" if np.isfinite(tot) and tot > 0 else "-", f"{O.get('hyslr_year')}년 사업보고서, 보통주"),
               ("발행주식수", f"{info['sharesOutstanding'] / 1e6:,.1f}백만주" if info.get("sharesOutstanding") else "-")], cols=2)
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.caption("최근 사업보고서의 최대주주 현황을 찾지 못했습니다.")

    el = O.get("elestock")
    st.markdown("##### 임원·주요주주 지분 변동 (최근 보고)")
    if isinstance(el, pd.DataFrame) and not el.empty:
        e = el.copy()
        e["rcept_dt"] = pd.to_datetime(e.get("rcept_dt"), errors="coerce")
        e = e.sort_values("rcept_dt", ascending=False)
        chg = _num_col(e.get("sp_stock_lmp_irds_cnt", pd.Series(dtype=str)))
        recent = e["rcept_dt"] >= pd.Timestamp.now() - pd.Timedelta(days=365)
        buys = int(((chg > 0) & recent).sum())
        sells = int(((chg < 0) & recent).sum())
        mgrid([("임원 지분 변동", f"증가 {buys}건 · 감소 {sells}건", "최근 1년 보고 기준"),
               ("공시 건수", f"{len(e):,}건")], cols=2)
        show = pd.DataFrame({"보고일": e["rcept_dt"].dt.strftime("%Y-%m-%d"), "보고자": e.get("repror"), "직위": e.get("isu_exctv_ofcps"),
                             "주요주주": e.get("isu_main_shrholdr"),
                             "증감 주식수": chg.map(lambda v: f"{v:+,.0f}" if np.isfinite(v) else "-"),
                             "보유 주식수": _num_col(e.get("sp_stock_lmp_cnt", pd.Series(dtype=str))).map(lambda v: f"{v:,.0f}" if np.isfinite(v) else "-"),
                             "보유 비율": e.get("sp_stock_lmp_rate")})
        st.dataframe(show.head(60), use_container_width=True, hide_index=True)
    else:
        st.caption("임원·주요주주 소유 보고가 없습니다.")

    ms = O.get("majorstock")
    st.markdown("##### 5% 이상 대량보유 보고")
    if isinstance(ms, pd.DataFrame) and not ms.empty:
        m = ms.copy()
        m["rcept_dt"] = pd.to_datetime(m.get("rcept_dt"), errors="coerce")
        m = m.sort_values("rcept_dt", ascending=False)
        show = pd.DataFrame({"보고일": m["rcept_dt"].dt.strftime("%Y-%m-%d"), "보고자": m.get("repror"), "보고 구분": m.get("report_tp"),
                             "보유 비율(%)": m.get("stkrt"), "비율 증감(%p)": m.get("stkrt_irds"), "보유 사유": m.get("report_resn")})
        st.dataframe(show.head(40), use_container_width=True, hide_index=True, column_config=col_help(show.columns))
    else:
        st.caption("대량보유 보고가 없습니다.")
