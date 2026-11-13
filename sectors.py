"""업종(섹터) 분류: 국내는 WICS(와이즈인덱스 산업분류, 국내 증권사·HTS에서 쓰는 GICS 방식), 미국은 GICS(S&P 500).

- 국내: wiseindex.com의 WICS 중분류(28개) 구성 종목을 받아 '섹터(대분류) → 업종(중분류) → 종목'으로 묶는다.
  받지 못하면 KRX 업종(한국표준산업분류)을 같은 WICS 이름으로 바꿔서 쓴다.
- 미국: S&P 500 구성 종목의 GICS 섹터·세부 업종(위키백과)과 나스닥 스크리너 시세(등락률·시가총액)를 합친다.
"""
from __future__ import annotations

import datetime as dt
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "application/json, text/plain, */*", "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"}

# WICS 중분류 코드 → (섹터, 업종)  * 섹터 이름은 증권사 화면에서 흔히 쓰는 짧은 이름
WICS = {
    "G1010": ("에너지", "에너지"),
    "G1510": ("소재", "화학·철강·소재"),
    "G2010": ("산업재", "자본재 (조선·방산·기계·건설)"),
    "G2020": ("산업재", "상업서비스"),
    "G2030": ("산업재", "운송 (항공·해운·물류)"),
    "G2510": ("경기소비재", "자동차와 부품"),
    "G2520": ("경기소비재", "내구소비재·의류·화장품"),
    "G2530": ("경기소비재", "호텔·레저·카지노"),
    "G2550": ("경기소비재", "소매·유통"),
    "G2560": ("경기소비재", "교육"),
    "G3010": ("필수소비재", "식품 유통"),
    "G3020": ("필수소비재", "식품·음료·담배"),
    "G3030": ("필수소비재", "가정·개인용품"),
    "G3510": ("헬스케어", "의료기기·서비스"),
    "G3520": ("헬스케어", "제약·바이오"),
    "G4010": ("금융", "은행"),
    "G4020": ("금융", "증권"),
    "G4030": ("금융", "기타 금융 (카드·캐피탈·창투)"),
    "G4040": ("금융", "보험"),
    "G4050": ("금융", "부동산·리츠"),
    "G4510": ("IT", "소프트웨어·IT서비스"),
    "G4520": ("IT", "IT 하드웨어·전자부품"),
    "G4530": ("IT", "반도체·반도체장비"),
    "G4535": ("IT", "전기·전자제품 (2차전지·가전)"),
    "G4540": ("IT", "디스플레이"),
    "G5010": ("커뮤니케이션", "통신"),
    "G5020": ("커뮤니케이션", "인터넷·게임·미디어·엔터"),
    "G5510": ("유틸리티", "유틸리티 (전력·가스)"),
}

# KRX 업종(한국표준산업분류) 글자 → WICS 중분류 코드 (WICS를 받지 못할 때만 씀, 위에서부터 먼저 맞는 것)
KSIC_RULES = [
    (r"반도체", "G4530"), (r"축전지|일차전지|전동기|발전기|전기 변환|전기장비|절연선|가정용 기기|전구|조명", "G4535"),
    (r"전자부품|인쇄회로|통신 및 방송 장비|컴퓨터 및 주변|영상 및 음향|측정|광학|사진장비|전자집적", "G4520"),
    (r"소프트웨어|프로그래밍|시스템 통합|자료처리|호스팅|포털|정보서비스|컴퓨터 프로그래밍", "G4510"),
    (r"의약품|의약물질|생물학적|생명과학|자연과학 및 공학 연구", "G3520"), (r"의료|치과|정형외과|안경", "G3510"),
    (r"자동차|차체|트레일러", "G2510"), (r"선박|보트|항공기|우주선|무기|총포|철도|기계|엔진|터빈|펌프|건설|토목|건물|공사|엔지니어링", "G2010"),
    (r"운송|해운|항공 운송|창고|택배|물류|여객", "G2030"), (r"석유|원유|천연가스 채굴|코크스", "G1010"),
    (r"전기업|가스 제조|증기|발전업|수도", "G5510"), (r"전기 통신|전기통신|통신업", "G5010"),
    (r"영화|방송|출판|오디오|광고|게임|창작|예술|음악", "G5020"), (r"은행|저축", "G4010"), (r"보험|연금", "G4040"),
    (r"증권|투자|신탁|금융 지원", "G4020"), (r"금융|지주|여신|카드", "G4030"), (r"부동산|임대", "G4050"),
    (r"식료품|음료|담배|곡물|도축|육류|수산물|제과|낙농|주류", "G3020"), (r"교육|학원", "G2560"),
    (r"숙박|음식점|여행|카지노|오락|스포츠|유원지", "G2530"), (r"소매|도매|상품 중개|무점포|백화점", "G2550"),
    (r"섬유|의복|가죽|신발|가구|화장품|생활용품|귀금속|악기|완구", "G2520"),
    (r"화학|합성|비료|플라스틱|고무|금속|철강|제철|시멘트|유리|요업|종이|펄프|목재|비금속|도료|광업", "G1510"),
    (r"사업 지원|경비|인력|시설 관리|폐기물|환경", "G2020"),
]

GICS_KO = {
    "Information Technology": "IT", "Health Care": "헬스케어", "Financials": "금융", "Consumer Discretionary": "경기소비재",
    "Communication Services": "커뮤니케이션", "Industrials": "산업재", "Consumer Staples": "필수소비재", "Energy": "에너지",
    "Utilities": "유틸리티", "Real Estate": "부동산", "Materials": "소재",
}

# 자주 나오는 GICS 세부 업종 한글 이름 (없는 것은 영어 그대로)
GICS_IND_KO = {
    "Semiconductors": "반도체", "Semiconductor Materials & Equipment": "반도체 장비", "Systems Software": "시스템 소프트웨어",
    "Application Software": "응용 소프트웨어", "Technology Hardware, Storage & Peripherals": "IT 하드웨어", "Communications Equipment": "통신장비",
    "IT Consulting & Other Services": "IT 서비스", "Electronic Components": "전자부품", "Electronic Equipment & Instruments": "전자장비",
    "Electronic Manufacturing Services": "전자제품 위탁생산", "Technology Distributors": "IT 유통",
    "Interactive Media & Services": "인터넷·검색·SNS", "Movies & Entertainment": "영화·엔터", "Interactive Home Entertainment": "게임",
    "Broadcasting": "방송", "Cable & Satellite": "케이블·위성", "Advertising": "광고", "Publishing": "출판",
    "Integrated Telecommunication Services": "통신", "Wireless Telecommunication Services": "무선통신",
    "Broadline Retail": "종합 소매(이커머스)", "Automobile Manufacturers": "자동차", "Automotive Parts & Equipment": "자동차 부품",
    "Restaurants": "외식", "Hotels, Resorts & Cruise Lines": "호텔·크루즈", "Casinos & Gaming": "카지노", "Home Improvement Retail": "주택용품 소매",
    "Apparel Retail": "의류 소매", "Apparel, Accessories & Luxury Goods": "의류·명품", "Footwear": "신발", "Homebuilding": "주택건설",
    "Other Specialty Retail": "전문 소매", "Automotive Retail": "자동차 소매", "Leisure Products": "레저용품", "Consumer Electronics": "가전",
    "Hypermarkets & Super Centers": "대형마트", "Consumer Staples Merchandise Retail": "생필품 소매", "Soft Drinks & Non-alcoholic Beverages": "음료",
    "Packaged Foods & Meats": "가공식품", "Household Products": "생활용품", "Personal Care Products": "개인용품·화장품", "Tobacco": "담배",
    "Brewers": "맥주", "Distillers & Vintners": "주류", "Food Distributors": "식품 유통", "Drug Retail": "약국", "Agricultural Products & Services": "농산물",
    "Pharmaceuticals": "제약", "Biotechnology": "바이오", "Health Care Equipment": "의료기기", "Health Care Supplies": "의료용품",
    "Managed Health Care": "건강보험", "Health Care Services": "의료 서비스", "Health Care Facilities": "병원",
    "Health Care Distributors": "의약품 유통", "Life Sciences Tools & Services": "생명과학 장비", "Health Care Technology": "의료 IT",
    "Diversified Banks": "대형 은행", "Regional Banks": "지역 은행", "Investment Banking & Brokerage": "투자은행·증권",
    "Asset Management & Custody Banks": "자산운용·수탁", "Financial Exchanges & Data": "거래소·금융데이터",
    "Transaction & Payment Processing Services": "결제", "Consumer Finance": "소비자금융·카드", "Multi-Sector Holdings": "복합 지주",
    "Property & Casualty Insurance": "손해보험", "Life & Health Insurance": "생명보험", "Insurance Brokers": "보험 중개",
    "Multi-line Insurance": "종합 보험", "Reinsurance": "재보험", "Diversified Financial Services": "종합 금융",
    "Aerospace & Defense": "항공우주·방산", "Industrial Machinery & Supplies & Components": "산업 기계",
    "Construction Machinery & Heavy Transportation Equipment": "건설기계·중장비", "Electrical Components & Equipment": "전기장비",
    "Heavy Electrical Equipment": "중전기기", "Industrial Conglomerates": "복합 기업", "Building Products": "건자재",
    "Construction & Engineering": "건설·엔지니어링", "Rail Transportation": "철도", "Passenger Airlines": "항공",
    "Air Freight & Logistics": "항공화물·물류", "Cargo Ground Transportation": "육상 화물", "Passenger Ground Transportation": "차량 호출",
    "Trading Companies & Distributors": "산업재 유통", "Environmental & Facilities Services": "환경·시설 서비스",
    "Human Resource & Employment Services": "인력 서비스", "Research & Consulting Services": "리서치·컨설팅",
    "Data Processing & Outsourced Services": "데이터 처리", "Diversified Support Services": "지원 서비스",
    "Agricultural & Farm Machinery": "농기계", "Security & Alarm Services": "보안 서비스",
    "Integrated Oil & Gas": "종합 석유", "Oil & Gas Exploration & Production": "석유·가스 탐사·생산",
    "Oil & Gas Equipment & Services": "유전 서비스", "Oil & Gas Refining & Marketing": "정유", "Oil & Gas Storage & Transportation": "석유·가스 수송",
    "Electric Utilities": "전력", "Multi-Utilities": "종합 유틸리티", "Gas Utilities": "가스", "Water Utilities": "수도",
    "Independent Power Producers & Energy Traders": "민자 발전", "Specialty Chemicals": "특수 화학", "Commodity Chemicals": "범용 화학",
    "Industrial Gases": "산업용 가스", "Fertilizers & Agricultural Chemicals": "비료·농화학", "Construction Materials": "건설 자재",
    "Steel": "철강", "Copper": "구리", "Gold": "금", "Paper & Plastic Packaging Products & Materials": "포장재", "Metal, Glass & Plastic Containers": "용기",
    "Data Center REITs": "데이터센터 리츠", "Telecom Tower REITs": "통신탑 리츠", "Industrial REITs": "물류 리츠", "Retail REITs": "상업 리츠",
    "Health Care REITs": "헬스케어 리츠", "Multi-Family Residential REITs": "주거 리츠", "Self-Storage REITs": "창고 리츠",
    "Office REITs": "오피스 리츠", "Hotel & Resort REITs": "호텔 리츠", "Single-Family Residential REITs": "단독주택 리츠",
    "Timber REITs": "산림 리츠", "Other Specialized REITs": "기타 특수 리츠", "Real Estate Services": "부동산 서비스",
}


def _get(url: str, timeout: float = 10.0, params: dict | None = None):
    import requests

    r = requests.get(url, headers=UA, timeout=timeout, params=params)
    r.raise_for_status()
    return r


# ----------------------------------------------------------------------
# 국내: WICS
# ----------------------------------------------------------------------
def _wics_one(code: str, day: str) -> pd.DataFrame:
    j = _get("https://www.wiseindex.com/Index/GetIndexComponets", params={"ceil_yn": 0, "dt": day, "sec_cd": code}).json()
    rows = j.get("list") or []
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    out = pd.DataFrame({"Code": d["CMP_CD"].astype(str).str.zfill(6), "WName": d["CMP_KOR"],
                        "Mcap": pd.to_numeric(d.get("MKT_VAL"), errors="coerce") * 1e6})  # 백만원 → 원
    out["Sector"], out["Industry"] = WICS[code]
    return out


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def wics_map() -> pd.DataFrame:
    """종목코드 → WICS 섹터·업종. 가장 최근 영업일 자료를 찾는다 (휴일이면 하루씩 앞으로)."""
    today = dt.datetime.utcnow() + dt.timedelta(hours=9)
    day = None
    for back in range(0, 12):
        d = (today - dt.timedelta(days=back)).strftime("%Y%m%d")
        try:
            if not _wics_one("G2510", d).empty:
                day = d
                break
        except Exception:  # noqa: BLE001
            continue
    if day is None:
        raise RuntimeError("WICS 구성 종목을 받지 못했습니다")
    with ThreadPoolExecutor(max_workers=8) as ex:
        parts = list(ex.map(lambda c: _wics_one(c, day), list(WICS)))
    m = pd.concat([p for p in parts if not p.empty], ignore_index=True).drop_duplicates("Code")
    if len(m) < 500:
        raise RuntimeError("WICS 구성 종목이 너무 적습니다")
    return m, day


def _ksic_to_wics(text) -> tuple:
    t = str(text or "")
    if t and not re.search(r"기업부|소속부|^nan$", t):
        for pat, code in KSIC_RULES:
            if re.search(pat, t):
                return WICS[code]
    return ("분류 없음", "분류 없음")


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def ksic_map() -> pd.DataFrame:
    import kr

    d = kr.load_krx()
    txt = d["Industry"].where(d["Industry"].notna() & ~d["Industry"].astype(str).str.contains("기업부"), d["Sector"])
    pairs = [_ksic_to_wics(x) for x in txt]
    return pd.DataFrame({"Code": d["Code"].astype(str).str.zfill(6),
                         "Sector": [p[0] for p in pairs], "Industry": [p[1] for p in pairs]})


def kr_sector_map() -> tuple:
    """(종목코드·섹터·업종 표, 출처 이름)."""
    try:
        m, day = wics_map()
        return m[["Code", "Sector", "Industry"]], f"WICS 산업분류 ({day[:4]}-{day[4:6]}-{day[6:]} 기준)"
    except Exception:  # noqa: BLE001
        return ksic_map(), "KRX 업종(한국표준산업분류)을 WICS 이름으로 변환"


# ----------------------------------------------------------------------
# 미국: S&P 500 + 나스닥 시세
# ----------------------------------------------------------------------
def _norm(sym) -> str:
    return re.sub(r"[./-]", ".", str(sym).strip().upper())


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def sp500_list() -> pd.DataFrame:
    import FinanceDataReader as fdr

    d = fdr.StockListing("S&P500")
    if "Symbol" not in d.columns and "Code" in d.columns:
        d = d.rename(columns={"Code": "Symbol"})
    for c in ("Name", "Sector", "Industry"):
        if c not in d.columns:
            d[c] = np.nan
    d = d[["Symbol", "Name", "Sector", "Industry"]].dropna(subset=["Symbol"]).copy()
    d["key"] = d["Symbol"].map(_norm)
    d["SectorKo"] = d["Sector"].map(lambda x: GICS_KO.get(str(x), str(x) if x == x else "분류 없음"))
    d["Industry"] = d["Industry"].fillna(d["Sector"]).astype(str).map(lambda x: GICS_IND_KO.get(x, x))
    return d.reset_index(drop=True)


def _num(x) -> float:
    try:
        return float(str(x).replace("$", "").replace(",", "").replace("%", "").strip())
    except Exception:  # noqa: BLE001
        return np.nan


@st.cache_data(ttl=300, show_spinner=False)
def nasdaq_snapshot() -> pd.DataFrame:
    """나스닥 스크리너 API: 미국 상장 전 종목의 현재가·등락률·거래량·시가총액 (장중엔 지연)."""
    j = _get("https://api.nasdaq.com/api/screener/stocks", timeout=20,
             params={"tableonly": "true", "limit": 10000, "offset": 0, "download": "true"}).json()
    rows = ((j.get("data") or {}).get("rows")) or []
    if not rows:
        raise RuntimeError("나스닥 시세가 비어 있습니다")
    d = pd.DataFrame(rows)
    out = pd.DataFrame({"key": d["symbol"].map(_norm), "Close": d["lastsale"].map(_num), "ChangesRatio": d["pctchange"].map(_num),
                        "Volume": d["volume"].map(_num), "Marcap": d["marketCap"].map(_num)})
    out["Amount"] = out["Close"] * out["Volume"]
    return out.drop_duplicates("key")


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _yf_mcaps(symbols: tuple) -> dict:
    import yfinance as yf

    def one(s):
        try:
            return s, float(yf.Ticker(s).fast_info["market_cap"])
        except Exception:  # noqa: BLE001
            return s, np.nan

    with ThreadPoolExecutor(max_workers=24) as ex:
        return dict(ex.map(one, symbols))


@st.cache_data(ttl=300, show_spinner=False)
def _yf_snapshot(symbols: tuple) -> pd.DataFrame:
    """나스닥 API가 막혔을 때: Yahoo 일봉(최근 5일)으로 등락률·거래대금, 시가총액은 종목별 조회."""
    import yfinance as yf

    d = yf.download(list(symbols), period="5d", interval="1d", auto_adjust=False, progress=False, threads=True)
    close, vol = d["Close"], d["Volume"]
    last, prev = close.ffill().iloc[-1], close.ffill().iloc[-2]
    caps = _yf_mcaps(symbols)
    out = pd.DataFrame({"Symbol": list(symbols)})
    out["key"] = out["Symbol"].map(_norm)
    out["Close"] = out["Symbol"].map(last)
    out["ChangesRatio"] = (out["Symbol"].map(last) / out["Symbol"].map(prev) - 1) * 100
    out["Volume"] = out["Symbol"].map(vol.iloc[-1])
    out["Amount"] = out["Close"] * out["Volume"]
    out["Marcap"] = out["Symbol"].map(caps)
    return out


def us_snapshot() -> tuple:
    """(S&P 500 시세 표, 미국 전체 시세 표 또는 None, 출처)."""
    sp = sp500_list()
    try:
        allq = nasdaq_snapshot()
        m = sp.merge(allq, on="key", how="left")
        src = "나스닥 스크리너 시세 · GICS 분류(S&P 500)"
        if m["Marcap"].notna().sum() < 300:
            raise RuntimeError("시세가 맞지 않습니다")
        return m, allq, src
    except Exception:  # noqa: BLE001
        q = _yf_snapshot(tuple(sp["Symbol"].astype(str).str.replace(".", "-", regex=False)))
        m = sp.merge(q.drop(columns=["Symbol"]), on="key", how="left")
        return m, None, "Yahoo 일봉 · GICS 분류(S&P 500)"
