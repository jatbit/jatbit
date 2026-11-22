"""기업 리서치 판단의 규칙: 비율 방향·기준값, 업종 유형(PROFILES)과 분류, 점수 기준 기간·표시 이름.

outlook 묶음: outlook_rules(규칙) → outlook_data(데이터 준비) → outlook_signals(신호·과거 검증) → outlook_score(점수·게이지·스크리너용)
→ outlook(화면). 뒤 모듈은 앞 모듈을 from … import * 로 이어받는다. 방법 설명은 outlook.py 머리말.
"""
import numpy as np
import pandas as pd
import streamlit as st

from research_data import RATIOS, _split_factor, annual_eps_table, compute_ratios

DIR = {  # +1: 높을수록 좋음, −1: 낮을수록 좋음 (여기 없는 비율은 판단에 쓰지 않음)
    "매출총이익률": 1, "영업이익률": 1, "순이익률": 1, "FCF 마진": 1, "ROE": 1, "ROA": 1,
    "부채비율": -1, "자기자본비율": 1, "유동비율": 1, "이자보상배율": 1, "총자산회전율": 1, "영업현금흐름/순이익": 1,
    "PER": -1, "PBR": -1, "PSR": -1,
}
GROWTH = {"매출 성장률", "영업이익 성장률", "순이익 성장률", "EPS 성장률"}
VALN = {"PER", "PBR", "PSR"}
CUT = 0.25  # 점수 합 ÷ 근거 수 기준
HORIZONS = [("1개월", 30), ("3개월", 91), ("6개월", 182), ("1년", 365)]


def _naive(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def _px_on(h: pd.Series, d) -> float:
    i = h.index.searchsorted(pd.Timestamp(d))
    return float(h.iloc[i]) if i < len(h) else np.nan


def _fwd(h: pd.Series, d, days: int) -> float:
    if h is None or not len(h):
        return np.nan
    t = pd.Timestamp(d) + pd.Timedelta(days=days)
    if t > h.index[-1] or pd.Timestamp(d) < h.index[0]:
        return np.nan
    p0, p1 = _px_on(h, d), _px_on(h, t)
    return p1 / p0 - 1 if p0 and np.isfinite(p0) and np.isfinite(p1) else np.nan


def _avail_dates(sec: pd.DataFrame, cols) -> dict:
    """그 사업연도 숫자를 실제로 쓸 수 있게 된 날 (첫 공시일, 없으면 회계연도 말 + 75일)."""
    row = pd.to_numeric(sec.loc["_avail_희석 EPS"], errors="coerce") if "_avail_희석 EPS" in sec.index else None
    out = {}
    for c in cols:
        c = pd.Timestamp(c)
        d = None
        if row is not None:
            v = row.get(c)
            if v is not None and np.isfinite(v):
                d = pd.Timestamp.fromordinal(int(v))
        if d is None or d < c or (d - c).days > 400:
            d = c + pd.Timedelta(days=75)
        out[c] = d
    return out


def _pctile(vals: np.ndarray, v: float) -> float:
    return float((np.sum(vals < v) + 0.5 * np.sum(vals == v)) / len(vals))


def _val_sig(p: float, direction: int) -> int:
    if p >= 0.8:
        return direction
    if p <= 0.2:
        return -direction
    return 0


def fscore(sec: pd.DataFrame, splits) -> pd.DataFrame:
    """피오트로스키 F-스코어 (9개 항목, 계산 가능한 항목만). 열: 점수·항목 수."""
    g = lambda k: pd.to_numeric(sec.loc[k], errors="coerce") if k in sec.index else pd.Series(np.nan, index=sec.columns)  # noqa: E731
    ta, ni, cfo = g("총자산"), g("순이익"), g("영업현금흐름")
    debt = (g("장기차입금") if g("장기차입금").notna().sum() >= 2 else g("총부채")) / ta
    cr = g("유동자산") / g("유동부채")
    gm = g("매출총이익") / g("매출")
    turn = g("매출") / ta
    roa = ni / ta
    sh = g("희석주식수") * _split_factor(g("_filed_희석주식수"), splits).reindex(sec.columns).fillna(1.0)
    tests = [roa > 0, cfo > 0, roa.diff() > 0, cfo > ni, debt.diff() < 0, cr.diff() > 0, sh <= sh.shift(1) * 1.005,
             gm.diff() > 0, turn.diff() > 0]
    known = [roa, cfo, roa.diff(), cfo + ni, debt.diff(), cr.diff(), sh.diff(), gm.diff(), turn.diff()]
    score = sum(t.astype(float).where(k.notna()) .fillna(0) for t, k in zip(tests, known))
    n = sum(k.notna().astype(int) for k in known)
    return pd.DataFrame({"점수": score, "항목": n})


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _index_hist(sym: str) -> pd.Series:
    import yfinance as yf

    return yf.Ticker(sym).history(period="max", auto_adjust=False)["Close"].dropna()


def _verdict(score: float, n: int):
    if not n:
        return None
    x = score / n
    return "상승 우세" if x >= CUT else ("하락 우세" if x <= -CUT else "중립")


def _sig_txt(s) -> str:
    return "상승 쪽 +1" if s > 0 else ("하락 쪽 −1" if s < 0 else "중립 0")


def _p(x, nd=1) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x * 100:+.{nd}f}%"


# ----------------------------------------------------------------------
# 업종 유형별로 볼 지표 (방법론 해설 문서 '업종별로 봐야 할 지표')
# w: 가중치(없으면 1), skip: 이 업종에서 판단에 쓰지 않는 지표와 이유, earn: 이익 기준 배수(PER 등)를 읽는 방법
#   normal = 자기 과거 대비 비싸면 하락 쪽 / peg = 이익 성장률로 나눈 PER(PEG)로 비교 / cycle = PER 대신 평균이익 PER
# ----------------------------------------------------------------------
EARN_KEYS = {"PER", "12M PER", "이익수익률 (E/P)"}
_FIN_SKIP = "부채가 영업 자산인 업종이라 일반 기업 기준이 맞지 않습니다"
PROFILES = {
    "general": dict(
        name="일반", earn="normal", w={}, skip={},
        why="업종 특성을 따로 반영하지 않고 모든 지표를 같은 무게로 봅니다.",
        rec=["주가", "PER (직전 사업연도 EPS)", "ROA", "영업이익률", "매출 성장률", "부채비율", "영업현금흐름/순이익"]),
    "growth": dict(
        name="성장·기술형", earn="peg",
        w={"매출 성장률": 1.5, "매출총이익률": 1.5, "FCF 마진": 1.5, "부채비율": 0.5, "유동비율": 0.5, "자기자본비율": 0.5,
           "이자보상배율": 0.5},
        skip={"PBR": "연구개발비가 비용으로 처리돼 장부가치가 실제 자산을 반영하지 못합니다"},
        why="이익이 빠르게 늘고 무형자산(연구개발·소프트웨어) 비중이 큰 업종입니다. 이익이 늘면 PER이 그대로여도 주가가 오르므로, "
            "자기 과거 대비 PER 대신 이익 성장률로 나눈 PER(PEG)로 비싼지 보고, 매출총이익률·성장률·현금흐름에 무게를 둡니다."),
    "cyclical": dict(
        name="경기민감형", earn="cycle",
        w={"영업이익률": 1.5, "부채비율": 1.5, "이자보상배율": 1.5, "PBR": 1.5, "EV/EBITDA": 1.5},
        skip={},
        why="이익이 경기에 따라 크게 오르내리는 업종입니다. PER이 낮을 때가 이익 고점(이후 하락)이고 높을 때가 이익 바닥인 경우가 많아, "
            "PER 대신 최근 최대 10년 평균 이익으로 나눈 '평균이익 PER'을 보고, PBR·EV/EBITDA·재무 안정성에 무게를 둡니다."),
    "financial": dict(
        name="금융", earn="normal", w={"ROE": 1.5, "PBR": 1.5},
        skip={k: _FIN_SKIP for k in ("부채비율", "유동비율", "이자보상배율", "매출총이익률", "영업이익률", "FCF 마진", "총자산회전율",
                                     "영업현금흐름/순이익", "PSR", "EV/EBITDA", "EV/EBIT", "EV/매출", "P/FCF", "P/영업현금흐름", "FCF 수익률")},
        why="은행·보험·증권은 부채가 영업 자산이라 부채비율·유동비율·EV 배수가 의미 없습니다. ROE와 PBR을 함께 봅니다 "
            "(ROE가 높은데 PBR이 낮으면 저평가)."),
    "reit": dict(
        name="부동산·리츠", earn="normal",
        w={"P/영업현금흐름": 1.5, "배당수익률 (TTM)": 1.5, "이자보상배율": 1.5, "PER": 0.5, "12M PER": 0.5},
        skip={},
        why="감가상각이 커서 순이익·PER이 실제 현금 창출력을 낮게 보여줍니다. 영업현금흐름 대비 가격(P/FFO 대용)과 배당수익률, 이자 부담을 봅니다."),
    "defensive": dict(
        name="방어·배당형", earn="normal", w={"배당수익률 (TTM)": 1.5, "이자보상배율": 1.5, "매출 성장률": 0.5},
        skip={},
        why="이익이 안정적이라 자기 과거 대비 PER 평균회귀가 비교적 잘 맞는 업종입니다. 배당수익률과 이자 부담을 함께 봅니다."),
}
PROFILES["growth"]["rec"] = ["주가", "12M 선행 PER (컨센서스 추이)", "PER (직전 사업연도 EPS)", "매출 성장률", "매출총이익률",
                             "영업이익률", "FCF 마진", "ROE"]
PROFILES["cyclical"]["rec"] = ["주가", "PER (직전 사업연도 EPS)", "PBR", "EV/EBITDA", "영업이익률", "매출 성장률", "부채비율",
                               "이자보상배율"]
PROFILES["financial"]["rec"] = ["주가", "PBR", "PER (직전 사업연도 EPS)", "ROE", "ROA"]
PROFILES["reit"]["rec"] = ["주가", "P/영업현금흐름", "배당수익률 (TTM)", "부채비율", "이자보상배율"]
PROFILES["defensive"]["rec"] = ["주가", "PER (직전 사업연도 EPS)", "배당수익률 (TTM)", "영업이익률", "ROE", "부채비율", "이자보상배율"]

_KR_WORDS = [  # 국내 업종명(KRX) → 유형
    ("reit", ("부동산", "리츠")),
    ("financial", ("은행", "보험", "증권", "금융", "신탁", "캐피탈", "카드")),
    ("cyclical", ("반도체", "전자부품", "통신 및 방송 장비", "디스플레이", "철강", "금속", "화학", "석유", "정유", "자동차", "조선",
                  "선박", "해운", "건설", "기계", "시멘트", "항공")),
    ("growth", ("소프트웨어", "인터넷", "정보서비스", "게임", "포털", "바이오", "의료용", "연구개발")),
    ("defensive", ("전기", "가스", "통신업", "전기통신", "식료품", "음료", "담배", "의약품")),
]


def classify(sector, industry, kr_sector=None) -> str:
    s, i = (sector or "").lower(), (industry or "").lower()
    if "reit" in i or s == "real estate":
        return "reit"
    if s == "financial services" or any(x in i for x in ("bank", "insurance", "capital markets", "credit services", "asset management")):
        return "financial"
    if any(x in i for x in ("semiconductor", "auto manufacturers", "auto parts", "steel", "chemical", "oil & gas", "marine shipping",
                            "airlines", "aluminum", "copper", "building materials", "construction", "farm & heavy",
                            "industrial machinery", "agricultural inputs", "coal", "industrial metals", "paper")):
        return "cyclical"
    if any(x in i for x in ("software", "internet", "information technology services", "biotechnology", "medical devices",
                            "electronic gaming", "diagnostics")):
        return "growth"
    if s in ("utilities", "consumer defensive") or any(x in i for x in ("telecom", "drug manufacturers", "tobacco")):
        return "defensive"
    if s in ("energy", "basic materials", "industrials", "consumer cyclical"):
        return "cyclical"
    if s in ("technology", "communication services", "healthcare"):
        return "growth"
    k = kr_sector or ""
    for prof, words in _KR_WORDS:
        if any(w in k for w in words):
            return prof
    return "general"


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def sec_sic(ticker: str):
    """SEC 등록 업종 코드(SIC)와 설명 — Yahoo 기업 정보가 막혔을 때 업종 판별용."""
    from research_data import _sec_cik_map, _sec_json

    m = _sec_cik_map()
    cik = m.get(ticker.upper()) or m.get(ticker.upper().replace(".", "-"))
    if not cik:
        return 0, ""
    j = _sec_json(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
    return int(j.get("sic") or 0), str(j.get("sicDescription") or "")


def classify_sic(sic: int) -> str:
    if not sic:
        return "general"
    if sic == 6798 or 6500 <= sic <= 6599:
        return "reit"
    if 6000 <= sic <= 6799:
        return "financial"
    if sic == 3674 or 1000 <= sic <= 1799 or 2800 <= sic <= 2829 or 2900 <= sic <= 3399 or 3500 <= sic <= 3569 \
            or 3710 <= sic <= 3716 or 4400 <= sic <= 4599:
        return "cyclical"
    if 7370 <= sic <= 7379 or sic in (2836, 5961, 7812) or 3570 <= sic <= 3579 or 3841 <= sic <= 3851 or 3661 <= sic <= 3669:
        return "growth"
    if 4900 <= sic <= 4999 or 4810 <= sic <= 4819 or 2000 <= sic <= 2199 or sic == 2834:
        return "defensive"
    return "general"


# ----------------------------------------------------------------------
# 점수 기준
# ----------------------------------------------------------------------
SCORE_H = ("6개월", 182)  # 점수의 기준 기간
RELIABLE_WIDTH = 20  # 80% 구간 폭이 이 점수 이하이면 한 점으로, 넘으면 범위로 보여줌


def score_label(x):
    return "매수" if x >= 65 else ("약한 매수" if x >= 55 else ("중립·보유" if x > 45 else ("약한 매도" if x > 35 else "매도")))


# ----------------------------------------------------------------------
# 스크리너용: 여러 종목 점수를 한꺼번에 (기업 리서치 첫 화면의 기본 설정과 같은 판정)
# ----------------------------------------------------------------------
GICS_YH = {"Information Technology": "Technology", "Health Care": "Healthcare", "Financials": "Financial Services",
           "Consumer Discretionary": "Consumer Cyclical", "Consumer Staples": "Consumer Defensive", "Materials": "Basic Materials",
           "Real Estate": "Real Estate", "Communication Services": "Communication Services", "Energy": "Energy",
           "Industrials": "Industrials", "Utilities": "Utilities"}
_GICS_IND = (("pharmaceuticals", "drug manufacturers"), ("automobile manufacturers", "auto manufacturers"),
             ("automotive parts", "auto parts"))


def screen_profile(domestic: bool, sector: str = "", industry: str = "", ksec: str = "", yinfo: dict | None = None) -> str:
    if yinfo and (yinfo.get("sector") or yinfo.get("industry")):
        return classify(yinfo.get("sector"), yinfo.get("industry"), ksec or None)
    if domestic:
        return classify(None, None, ksec)
    ind = (industry or "").lower()
    for a, b in _GICS_IND:
        if a in ind:
            ind = b
    return classify(GICS_YH.get(sector, sector), ind)


__all__ = [n for n in dir() if not n.startswith("__")]  # 밑줄로 시작하는 도우미도 다음 모듈에서 쓸 수 있게
