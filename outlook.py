"""기업 리서치: 고른 재무비율로 '상승 우세 / 중립 / 하락 우세'를 판단하고, 같은 방법을 과거에 적용했을 때의 결과를 보여준다.

방법 (방법론 해설 문서의 '앱에 넣는다면' 설계)
- 수익성·안정성·효율성·현금흐름 비율: 전년보다 좋아졌으면 +1, 나빠졌으면 −1 (피오트로스키 F-스코어 방식)
- 성장률: 플러스 성장 +1, 마이너스 성장 −1
- 밸류에이션(PER 등): 고른 기간 안에서 그때까지의 값 중 상위 20%면 −1, 하위 20%면 +1 (자기 과거 대비 평균회귀)
- 점수 합 ÷ 근거 수가 +0.25 이상이면 상승 우세, −0.25 이하면 하락 우세
과거 검증: 기간 안의 매달 첫 거래일마다 그때까지 공시된 숫자로 다시 판정하고 그 뒤 1·3·6개월·1년 주가 수익률을 붙인다.
기대수익률: 과거에 지금과 같은 판정이었던 달들의 이후 수익률 평균 (사례가 적으면 기간 전체 평균 쪽으로 당김).
업종 유형(성장·기술 / 경기민감 / 금융 / 리츠 / 방어)에 따라 지표 가중치와 PER을 읽는 방법을 바꾼다 (PROFILES).
추가 신호: 분기 TTM 재무, 모멘텀(12-1개월), 실적 서프라이즈, 동종업계 대비 백분위, 위험 신호 거부권(Altman Z''·Beneish M).
기대수익률 = CAPM(무위험 + 베타 × 시장 위험프리미엄) + 같은 판정이었던 과거의 평균 알파(사례 수만큼만 반영). 워크포워드로 검증.
매수·매도 점수(0~100): CAPM 미포함(6개월 수익률이 플러스였던 비율) · 업종 대비(섹터 ETF를 이긴 비율) · CAPM 대비(CAPM을 이긴 비율) ·
미래 추정치(예상 EPS 성장 + PER 회귀 + 배당 → 6개월 수익률이 플러스일 확률). 80% 구간이 20점보다 넓으면 범위로 표시.
추가 신호(영상 방법론, 업종별 검증 통과분만 EXTRA_ON): 고점 대비 하락 · 현금 품질 · 주주환원 · 외국인 수급(국내).
밸류에이션은 후행이므로 외국인 지분율이 늘고 있으면 PER 등이 비싸도 감점하지 않는다.
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


# ----------------------------------------------------------------------
# 신호 계산
# ----------------------------------------------------------------------
def _ratio_signal(k, row, j, pcols, kind, lag, min_n):
    v = row.iloc[j]
    if not np.isfinite(v):
        return None
    unit = RATIOS[k][0] if k in RATIOS else "배"
    fmt = _fmt(unit)
    if kind in ("level", "peg"):
        past = np.array([row[c] for c in pcols if c <= row.index[j] and np.isfinite(row[c])])
        if len(past) < min_n:
            return None
        p = _pctile(past, v)
        lab = f"PEG {v:.2f}" if kind == "peg" else fmt(v)
        return _val_sig(p, DIR.get(k, -1)), f"{lab} · 기간 내 백분위 {p * 100:.0f}%"
    if kind == "growth":
        return (1 if v > 1 else (-1 if v < -1 else 0)), fmt(v)
    if j - lag < 0:
        return None
    prev = row.iloc[j - lag]
    if not np.isfinite(prev):
        return None
    ch = v - prev
    sig = int(np.sign(ch)) * DIR[k] if abs(ch) > 0.02 * abs(prev) else 0
    return sig, f"1년 전 {fmt(prev)} → {fmt(v)}"


def _daily_signal(s, unit, start, d, label="", direction=None):
    vals = s[(s.index >= start) & (s.index <= pd.Timestamp(d))].dropna()
    if len(vals) < 120:
        return None
    v = float(vals.iloc[-1])
    p = _pctile(vals.values, v)
    direction = direction if direction is not None else (1 if unit == "%" else -1)  # 배당수익률 같은 %는 높을수록 쌈
    lab = f"{label} {v:.2f}" if label else (f"{v:.2f}%" if unit == "%" else f"{v:.1f}배")
    return _val_sig(p, direction), f"{lab} · 기간 내 백분위 {p * 100:.0f}%"


def build_ctx(sec, R, ticker, daily, opts, domestic=False, peers=None):
    """판정에 필요한 데이터를 한 번에 준비 (업종 유형과 무관한 부분)."""
    h = _naive(R["hist"])
    c = dict(h=h, sec=sec, splits=R.get("splits"), opts=opts, domestic=domestic, ticker=ticker)
    period = R.get("_period")
    rt_a = compute_ratios(sec, R["hist"], R.get("splits"))
    rt_a.columns = [pd.Timestamp(x) for x in rt_a.columns]
    c["rt_a"], c["avail_a"] = rt_a, _avail_dates(sec, rt_a.columns)
    c["start"] = pd.Timestamp(period[0]) if period else rt_a.columns[0]
    c["end"] = pd.Timestamp(period[1]) + pd.Timedelta(days=1) if period else h.index[-1]
    qf = None
    if opts.get("quarterly") and not domestic:
        try:
            qf = quarterly_frame(ticker)
        except Exception:  # noqa: BLE001
            qf = None
    if qf is not None and qf.shape[1] >= 8:
        rt = compute_ratios(qf, R["hist"], R.get("splits"), lag=4)
        rt.columns = [pd.Timestamp(x) for x in rt.columns]
        c.update(rt=rt, avail=_avail_dates(qf, rt.columns), lag=4, min_n=8, freq="분기", qf=qf)
    else:
        c.update(rt=rt_a, avail=c["avail_a"], lag=1, min_n=3, freq="연간")
    c["cols"] = list(c["rt"].columns)
    c["pcols"] = [x for x in c["cols"] if c["start"] <= x <= c["end"]]
    c["dser"] = {nm: (_naive(s), u) for nm, (s, u) in daily.items() if isinstance(s, pd.Series) and len(s)}
    c["eps_tb"] = annual_eps_table(sec, R.get("splits"))
    c["g_date"], _ = _eps_growth(sec, R.get("splits"))
    tb = c["eps_tb"]
    if tb is not None and len(tb) >= 5:  # 평균이익(최근 최대 10년, 최소 5년) — 경기민감형 PER용
        e = tb["EPS"].values
        avg = [np.mean(e[max(0, i - 9):i + 1]) if i >= 4 else np.nan for i in range(len(e))]
        step = pd.Series(avg, index=tb.index)
        st_ = _step_at(step, h.index)
        c["norm_per"] = pd.Series(np.where(st_ > 0, h.values / st_.values, np.nan), index=h.index)
    else:
        c["norm_per"] = None
    c["mkt"], c["mname"] = None, None
    sym = "^KS11" if ticker.endswith(".KS") else ("^KQ11" if ticker.endswith(".KQ") else "^GSPC")
    try:
        c["mkt"] = _naive(_index_hist(sym))
        c["mname"] = {"^KS11": "코스피", "^KQ11": "코스닥", "^GSPC": "S&P 500"}[sym]
    except Exception:  # noqa: BLE001
        pass
    c["sect"], c["sname"] = None, None
    try:
        ssym, c["sname"] = sector_bench(ticker, R.get("info") or {}, domestic)
        c["sect"] = c["mkt"] if ssym in ("^GSPC", "^KS11", "^KQ11") and c["mkt"] is not None else _naive(_hist_close(ssym)[0])
    except Exception:  # noqa: BLE001
        c["sect"] = None
    c["sue"] = _surprises(R.get("edates")) if opts.get("surprise") else pd.Series(dtype=float)
    c["peers"] = []
    if opts.get("peers") and peers and not domestic:
        from concurrent.futures import ThreadPoolExecutor

        def one(p):
            try:
                return p, _peer_ratios(p)
            except Exception:  # noqa: BLE001
                return p, None

        with ThreadPoolExecutor(max_workers=4) as ex:
            c["peers"] = [(p, r) for p, r in ex.map(one, [p for p in peers if p.upper() != ticker.upper()][:10]) if r is not None]
    c["frgn"], c["frgn_err"] = None, ""
    if opts.get("extra") and domestic:
        try:
            c["frgn"] = foreign_flow(ticker.split(".")[0])
            if c["frgn"] is None:
                c["frgn_err"] = "자료 없음"
        except Exception as e:  # noqa: BLE001
            c["frgn_err"] = str(e)[:60]
    c["extra"] = {}
    if opts.get("flags") and not domestic:
        try:
            c["extra"] = extra_annual(ticker)
        except Exception:  # noqa: BLE001
            c["extra"] = {}
    # 주간 수익률 (베타 추정용)
    hw = h.resample("W-FRI").last().pct_change()
    c["hw"] = hw
    c["mw"] = c["mkt"].resample("W-FRI").last().pct_change() if c["mkt"] is not None else None
    return c


def _beta(c, d):
    if c["mw"] is None:
        return 1.0
    d = pd.Timestamp(d)
    a = c["hw"][(c["hw"].index > d - pd.Timedelta(days=730)) & (c["hw"].index <= d)]
    b = c["mw"].reindex(a.index)
    m = a.notna() & b.notna()
    if m.sum() < 52 or float(b[m].var()) <= 0:
        return 1.0
    beta = float(np.cov(a[m], b[m])[0, 1] / b[m].var())
    return 0.67 * beta + 0.33  # 블룸 조정 (1 쪽으로 당김)


def make_signals(c, prof, use):
    """업종 유형 prof 기준 신호 함수 signals(d) → {이름: (신호, 설명, 가중치)}, 제외 목록."""
    P = PROFILES[prof]
    o = c["opts"]
    skipped, ann = {}, []
    g_date = c["g_date"]
    rt, cols, lag = c["rt"], c["cols"], c["lag"]
    for k in use:
        if k in P["skip"]:
            skipped[k] = P["skip"][k]
            continue
        if k not in rt.index:
            continue
        if k in VALN:
            if k in EARN_KEYS and P["earn"] == "cycle":
                continue
            if k in EARN_KEYS and P["earn"] == "peg":
                g = _step_at(g_date, [c["avail"][x] for x in cols]) if g_date is not None else pd.Series(np.nan, index=cols)
                g.index = cols
                ann.append((k, rt.loc[k] / g.where(g > 0), "peg"))
            else:
                ann.append((k, rt.loc[k], "level"))
        elif k in GROWTH:
            ann.append((k, rt.loc[k], "growth"))
        else:
            ann.append((k, rt.loc[k], "change"))
    daily = []
    has_earn = any(k in EARN_KEYS for k in use) or any(nm in EARN_KEYS for nm in c["dser"])
    for nm, (s, u) in c["dser"].items():
        if nm in P["skip"]:
            skipped[nm] = P["skip"][nm]
            continue
        if nm in EARN_KEYS and P["earn"] == "cycle":
            continue
        if nm in EARN_KEYS and P["earn"] == "peg" and u != "%":
            g = _step_at(g_date, s.index)
            daily.append((nm + " (성장 대비)", s / g.where(g > 0).values, u, "PEG", None, P["w"].get(nm, 1.0)))
        else:
            daily.append((nm, s, u, "", None, P["w"].get(nm, 1.0)))
    if P["earn"] == "cycle" and has_earn and c["norm_per"] is not None:
        daily.append(("평균이익 PER", c["norm_per"], "배", "", -1, 1.5))
    peer_keys = [k for k in use if k in c["rt_a"].index and k not in P["skip"]] if c["peers"] else []
    h, mkt = c["h"], c["mkt"]

    def fy_at(d):
        ok = [x for x in cols if c["avail"][x] <= d]
        return ok[-1] if ok and (pd.Timestamp(d) - ok[-1]).days <= 550 else None

    def signals(d):
        out = {}
        col = fy_at(d)
        if col is not None:
            j = cols.index(col)
            for k, row, kind in ann:
                r = _ratio_signal(k, row, j, c["pcols"], kind, lag, c["min_n"])
                if r is not None:
                    out[k + (" (성장 대비)" if kind == "peg" else "")] = (r[0], r[1], P["w"].get(k, 1.0))
        for nm, s, u, lab, dirn, w in daily:
            r = _daily_signal(s, u, c["start"], d, lab, dirn)
            if r is not None:
                out[nm] = (r[0], r[1], w)
        if o.get("momentum"):
            p0, p1 = _px_before(h, pd.Timestamp(d) - pd.Timedelta(days=365)), _px_before(h, pd.Timestamp(d) - pd.Timedelta(days=30))
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                r = p1 / p0 - 1
                ex = r
                if mkt is not None:
                    m0, m1 = _px_before(mkt, pd.Timestamp(d) - pd.Timedelta(days=365)), _px_before(mkt, pd.Timestamp(d) - pd.Timedelta(days=30))
                    if np.isfinite(m0) and np.isfinite(m1) and m0 > 0:
                        ex = r - (m1 / m0 - 1)
                sig = 1 if ex > 0.05 else (-1 if ex < -0.05 else 0)
                out["모멘텀 (12-1개월)"] = (sig, f"{r * 100:+.1f}%" + (f" · 지수 대비 {ex * 100:+.1f}%p" if mkt is not None else ""), 1.0)
        sue = c["sue"]
        if len(sue):
            past = sue[sue.index <= pd.Timestamp(d)]
            if len(past) and (pd.Timestamp(d) - past.index[-1]).days <= 91:
                v = float(past.iloc[-1])
                if len(past) >= 6:
                    p = _pctile(past.values, v)
                    sig = 1 if p >= 0.7 else (-1 if p <= 0.3 else 0)
                    extra = f" · 과거 서프라이즈 중 백분위 {p * 100:.0f}%"
                else:
                    sig = 1 if v > 0.05 else (-1 if v < -0.05 else 0)
                    extra = ""
                out["실적 서프라이즈"] = (sig, f"예상 대비 {v * 100:+.1f}% ({past.index[-1]:%Y-%m-%d} 발표){extra}", 1.0)
        if peer_keys:
            col_a = [x for x in c["rt_a"].columns if c["avail_a"][x] <= d]
            if col_a and (pd.Timestamp(d) - col_a[-1]).days <= 550:
                ca = col_a[-1]
                for k in peer_keys:
                    me = c["rt_a"].loc[k, ca]
                    if not np.isfinite(me):
                        continue
                    vals = []
                    for _p_, (prt, pav) in c["peers"]:
                        if k not in prt.index:
                            continue
                        pc = [x for x in prt.columns if pav[x] <= d]
                        if pc and (pd.Timestamp(d) - pc[-1]).days <= 550 and np.isfinite(prt.loc[k, pc[-1]]):
                            vals.append(float(prt.loc[k, pc[-1]]))
                    if len(vals) < 4:
                        continue
                    p = _pctile(np.array(vals + [me]), me)
                    dirn = DIR.get(k, 1)
                    sig = dirn if p >= 0.8 else (-dirn if p <= 0.2 else 0)
                    out[f"{k} (동종업계 대비)"] = (sig, f"{_fmt(RATIOS[k][0])(me)} · {len(vals)}개사 중 백분위 {p * 100:.0f}%", 1.0)
        if o.get("extra"):
            _extra_signals(c, P, prof, d, out, col, val_names)
        return out

    val_names = ({k + (" (성장 대비)" if kind == "peg" else "") for k, _r, kind in ann if kind in ("level", "peg")}
                 | {x[0] for x in daily} | {"PBR ÷ ROE"})
    return signals, skipped


FRGN_CHG = 0.5  # 60거래일 외국인 지분율 변화(%p)가 이 이상이면 매수·매도 신호
# 업종별로 켜는 추가 신호 — 2026-11-21 업종별 대표 종목 검증(신호가 켜진 달의 이후 6개월 CAPM 대비 초과수익)에서
# 적중률 50% 초과 + 평균 초과수익 플러스였던 것만 남김. 결과(적중률, 평균 초과수익):
#   고점 대비 하락: 성장 55% +0.8%p · 경기민감 48% −6.7%p · 금융 56% +0.4%p · 방어 62% +1.5%p · 일반 58% +2.3%p · 리츠 66% +4.8%p
#                   (국내 4종목 합계 43% → 국내 종목은 끔)
#   이익 고점 위치(경기민감): 46% −6.5%p → 뺌        PBR ÷ ROE: 모든 업종 50% 이하 → 뺌
#   현금 품질: 성장 58% +2.9%p만 통과 · 주주환원: 성장 71% +7.1%p · 경기민감 52% +5.8%p만 통과
#   외국인 수급(국내 4종목 합계 52%, 평균 +4.1%p): 국내 종목에 항상 사용
EXTRA_ON = {
    "growth": {"고점 대비 하락", "현금 품질", "주주환원"},
    "cyclical": {"주주환원"},
    "financial": {"고점 대비 하락"},
    "defensive": {"고점 대비 하락"},
    "general": {"고점 대비 하락"},
    "reit": {"고점 대비 하락"},
}


def _extra_signals(c, P, prof, d, out, col, val_names):
    """방법론 문서 '영상에서 정리한 방법론' 신호. 조건에 걸릴 때만 넣는다(아니면 근거 수를 늘리지 않음)."""
    d = pd.Timestamp(d)
    h = c["h"]
    on = EXTRA_ON.get(prof, set())
    # 1) 고점 대비 하락: 52주 고점에서 15% 넘게 빠지고 최근 3개월도 마이너스면 매도 쪽 (고점은 못 맞추니 꺾인 걸 확인하고 판다)
    hh = h[(h.index > d - pd.Timedelta(days=365)) & (h.index <= d)]
    if "고점 대비 하락" in on and not c["domestic"] and len(hh) > 200:
        dd = float(hh.iloc[-1] / hh.max() - 1)
        p3 = _px_before(h, d - pd.Timedelta(days=91))
        r3 = float(hh.iloc[-1] / p3 - 1) if np.isfinite(p3) and p3 > 0 else np.nan
        if dd <= -0.15 and np.isfinite(r3) and r3 < 0:
            out["고점 대비 하락"] = (-1, f"52주 고점 대비 {dd * 100:+.1f}% · 최근 3개월 {r3 * 100:+.1f}%", 1.0)
    # 2) 이익 고점 위치 (경기민감형): 지금 EPS가 최근 최대 10년 평균보다 크게 높으면 이익 고점권, 크게 낮으면 바닥권
    tb = c["eps_tb"]
    if "이익 고점 위치" in on and P["earn"] == "cycle" and tb is not None:
        t = tb[tb.index <= d]
        if len(t) >= 5 and (d - t.index[-1]).days <= 550:
            e = t["EPS"].values[-10:]
            avg, cur = float(np.mean(e)), float(e[-1])
            if avg > 0:
                ratio = cur / avg
                if ratio >= 1.5:
                    out["이익 고점 위치"] = (-1, f"EPS {cur:.2f} = 최근 {len(e)}년 평균의 {ratio:.1f}배 (이익 고점권)", 1.0)
                elif ratio <= 0.6:
                    out["이익 고점 위치"] = (1, f"EPS {cur:.2f} = 최근 {len(e)}년 평균의 {ratio:.1f}배 (이익 바닥권)", 1.0)
    rt = c["rt"]
    if col is not None:
        # 3) 현금 품질: 순이익보다 영업현금흐름이 많으면 좋음, 크게 적으면 이익의 질이 낮음
        if "현금 품질" in on and {"영업현금흐름/순이익", "순이익률"} <= set(rt.index):
            v, nm = rt.loc["영업현금흐름/순이익", col], rt.loc["순이익률", col]
            if np.isfinite(v) and np.isfinite(nm) and nm > 0:
                if v >= 1.1:
                    out["현금 품질"] = (1, f"영업현금흐름 ÷ 순이익 {v:.2f}배", 1.0)
                elif v < 0.8:
                    out["현금 품질"] = (-1, f"영업현금흐름 ÷ 순이익 {v:.2f}배 (이익보다 현금이 적음)", 1.0)
        # 4) PBR ÷ ROE: ROE가 높아지면 PBR이 높아도 비싸지 않다 → 자기 과거 대비 상위 20%면 비쌈, 하위 20%면 쌈
        if "PBR ÷ ROE" in on and {"PBR", "ROE"} <= set(rt.index):
            x = rt.loc["PBR"] / rt.loc["ROE"].where(rt.loc["ROE"] > 0)
            past = np.array([x[cc] for cc in c["pcols"] if cc <= col and np.isfinite(x[cc])])
            v = x[col]
            if np.isfinite(v) and len(past) >= c["min_n"]:
                p = _pctile(past, v)
                s = _val_sig(p, -1)
                if s:
                    out["PBR ÷ ROE"] = (s, f"PBR {rt.loc['PBR', col]:.2f}배 ÷ ROE {rt.loc['ROE', col]:.1f}% · 기간 내 백분위 {p * 100:.0f}%",
                                        P["w"].get("PBR", 1.0))
    # 5) 주주환원: 배당 + 자사주 매입이 잉여현금흐름의 절반 이상이고 시가총액의 3% 이상이면 매수 쪽 (연간 재무제표)
    rt_a, sec = c["rt_a"], c["sec"]
    ok = [x for x in rt_a.columns if c["avail_a"][x] <= d]
    if "주주환원" in on and ok and (d - ok[-1]).days <= 550 and "PBR" in rt_a.index:
        j = list(rt_a.columns).index(ok[-1])

        def val(k):
            if k not in sec.index:
                return np.nan
            return pd.to_numeric(sec.loc[k], errors="coerce").iloc[j]

        ret = np.nansum([abs(val("배당금")), abs(val("자사주매입"))])
        fcf = val("영업현금흐름") - (abs(val("설비투자")) if np.isfinite(val("설비투자")) else 0.0)
        mcap = rt_a.loc["PBR"].iloc[j] * val("자기자본")
        if ret > 0 and np.isfinite(fcf) and fcf > 0 and np.isfinite(mcap) and mcap > 0:
            y, po = ret / mcap, ret / fcf
            if po >= 0.5 and y >= 0.03:
                out["주주환원"] = (1, f"배당+자사주 매입 = 시가총액의 {y * 100:.1f}% (잉여현금흐름의 {po * 100:.0f}%)", 1.0)
    # 6) 외국인 수급 (국내): 60거래일 외국인 지분율 변화
    f = c.get("frgn")
    if f is not None:
        fr = f[f.index <= d]
        if len(fr) >= 61 and (d - fr.index[-1]).days <= 10:
            r0, r1 = float(fr["rate"].iloc[-61]), float(fr["rate"].iloc[-1])
            chg = r1 - r0
            if abs(chg) >= FRGN_CHG:
                out["외국인 수급"] = (1 if chg > 0 else -1, f"60거래일 외국인 지분율 {r0:.2f}% → {r1:.2f}% ({chg:+.2f}%p)", 1.0)
    # 밸류에이션은 후행: 외국인이 사고 있으면 비싸 보여도(PER 등 상위 20%) 감점하지 않는다
    if out.get("외국인 수급", (0,))[0] > 0:
        for k in list(out):
            base = k.split(" (")[0]
            if (k in val_names or base in VALN) and out[k][0] < 0:
                out[k] = (0, out[k][1] + " · 외국인 매수 중이라 감점 안 함", out[k][2])


def _score(sg: dict):
    return sum(v[0] * v[2] for v in sg.values()), sum(v[2] for v in sg.values())


def _flags_at(flags, c, d):
    ok = [x for x in flags if c["avail_a"].get(pd.Timestamp(x), pd.Timestamp.max) <= d]
    return flags[ok[-1]] if ok else []


def evaluate(c, prof, use, rf_now):
    """매달 첫 거래일 판정과 이후 수익률(1·3·6개월·1년), CAPM 기대수익률, 실현 알파."""
    signals, skipped = make_signals(c, prof, use)
    flags = red_flags(c["sec"], c["extra"], prof) if c["opts"].get("flags") else {}
    h, mkt = c["h"], c["mkt"]
    hp = h[(h.index >= c["start"]) & (h.index <= c["end"])]
    dates = list(hp.groupby([hp.index.year, hp.index.month]).head(1).index)
    rows = []
    for d in dates:
        sg = signals(d)
        if not sg:
            continue
        sc, n = _score(sg)
        vd = _verdict(sc, n)
        fl = _flags_at(flags, c, d) if flags else []
        if fl and vd == "상승 우세":
            vd = "중립"
        beta, rf = _beta(c, d), _rf_at(d, c["domestic"], rf_now)
        rec = dict(d=d, sig=sg, score=sc, n=n, verdict=vd, flags=fl, beta=beta, rf=rf)
        for lab, days in HORIZONS:
            r = _fwd(h, d, days)
            m = _fwd(mkt, d, days) if mkt is not None else np.nan
            rfh = (1 + rf) ** (days / 365) - 1
            capm = (1 + rf + beta * MRP) ** (days / 365) - 1
            rec[lab] = r
            rec["capm_" + lab] = capm
            rec["a_" + lab] = (r - (rfh + beta * (m - rfh))) if np.isfinite(r) and np.isfinite(m) else (r - capm)
            sr = _fwd(c["sect"], d, days) if c.get("sect") is not None else np.nan
            rec["i_" + lab] = r - sr if np.isfinite(r) and np.isfinite(sr) else np.nan
        rows.append(rec)
    return rows, signals, skipped, flags


def _alpha_est(sample, days):
    """같은 판정 사례의 평균 알파를 사례 수만큼만 믿고 0(=CAPM) 쪽으로 당김."""
    a = np.array([x for x in sample if np.isfinite(x)])
    if not len(a):
        return 0.0, 0
    n_eff = len(a) * min(1.0, 30 / days)  # 매달 판정하므로 기간이 겹치는 만큼 독립 사례는 적다
    return float(n_eff / (n_eff + SHRINK_K) * a.mean()), len(a)


def walk_forward(rows):
    """각 판정 시점에 '그때까지 결과가 나온 과거 사례'만으로 예측 → 실제와 비교."""
    out = {}
    for lab, days in HORIZONS:
        em, ec, hit, nh = [], [], 0, 0
        for r in rows:
            if not np.isfinite(r[lab]):
                continue
            prior = [k["a_" + lab] for k in rows if k["d"] + pd.Timedelta(days=days) <= r["d"] and k["verdict"] == r["verdict"]]
            a, n = _alpha_est(prior, days)
            pred = r["capm_" + lab] + a
            em.append(abs(r[lab] - pred))
            ec.append(abs(r[lab] - r["capm_" + lab]))
            if a != 0 and np.isfinite(r["a_" + lab]) and r["a_" + lab] != 0:
                nh += 1
                hit += int(np.sign(a) == np.sign(r["a_" + lab]))
        out[lab] = dict(mae_m=np.mean(em) if em else np.nan, mae_c=np.mean(ec) if ec else np.nan, hit=hit, nh=nh, n=len(em))
    return out


def _hit_rate(rows, key="score"):
    dd = [r for r in rows if np.isfinite(r["1년"]) and r.get(key, 0) != 0 and r["1년"] != 0]
    return sum(1 for r in dd if np.sign(r[key]) == np.sign(r["1년"])), len(dd)


def sig_tally(rows, lab="6개월"):
    """신호별: 신호가 켜진(±1) 달 중 이후 6개월 CAPM 대비 초과수익(알파) 방향이 맞은 횟수, 사례 수, 신호 방향을 곱한 평균 알파."""
    out = {}
    for r in rows:
        a = r.get("a_" + lab, np.nan)
        if not np.isfinite(a) or a == 0:
            continue
        for k, (s, _desc, _w) in r["sig"].items():
            if s == 0:
                continue
            t = out.setdefault(k, [0, 0, 0.0])
            t[0] += int(np.sign(s) == np.sign(a))
            t[1] += 1
            t[2] += float(np.sign(s) * a)
    return out


def _w(x) -> str:
    return f"{x:g}"


def _set_rec(key, vals):
    st.session_state[key] = vals


def _split_rec(prof, dv_keys):
    """업종 추천 지표 → (연간 재무비율 목록, 일별 지표 이름 목록)."""
    ann, daily = [], []
    for k in PROFILES[prof]["rec"]:
        if k == "주가":
            continue
        if k == "PER (직전 사업연도 EPS)":
            daily.append("PER")
        elif k == "12M 선행 PER (컨센서스 추이)":
            continue
        elif k in dv_keys:
            daily.append(k)
        elif k in RATIOS:
            ann.append(k)
    return ann, daily


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def validate_profile(prof: str, years: int = 10):
    """업종 유형의 대표 종목들에 같은 규칙(추천 지표·분기 TTM·모멘텀·위험 신호)을 적용한 결과."""
    from concurrent.futures import ThreadPoolExecutor

    import stats
    import valuation as V
    from research_data import annual_per_series, load_sec_financials

    rf_now = stats.risk_free_for(False, pd.Timestamp.now().date())[0]
    end = pd.Timestamp.now().normalize()
    period = (end - pd.Timedelta(days=int(365.25 * years)), end)

    def one(t):
        try:
            sec = load_sec_financials(t)
            h, sp = _hist_close(t)
            daily = {}
            ann = annual_per_series(h, sec, sp)
            if ann is not None:
                daily["PER"] = (ann, "배")
            try:
                dv = V.daily_valuation(V.load_quarterly(t), h, sp)
            except Exception:  # noqa: BLE001
                dv = None
            dv_keys = [k for k in V.VAL if dv is not None and k in dv and dv[k].notna().sum() > 20]
            a_use, d_use = _split_rec(prof, dv_keys)
            d_use_g = d_use
            for k in d_use:
                if k != "PER" and dv is not None:
                    daily[k] = (dv[k], V.VAL[k][0])
            daily = {k: v for k, v in daily.items() if k in d_use_g}
            R = {"hist": h, "splits": sp, "_period": period}
            opts = dict(quarterly=True, momentum=True, surprise=False, peers=False, flags=True, extra=True)
            c = build_ctx(sec, R, t, daily, opts)
            rows_p = evaluate(c, prof, a_use, rf_now)[0]
            rows_g = evaluate(c, "general", a_use, rf_now)[0]
            hp, np_ = _hit_rate(rows_p)
            hg, ng = _hit_rate(rows_g)
            wf = walk_forward(rows_p)["1년"]
            pts = [(float(r["score"] / r["n"]), float(r["a_" + SCORE_H[0]]), float(r[SCORE_H[0]]), float(r["i_" + SCORE_H[0]]))
                   for r in rows_p if r["n"] and np.isfinite(r["a_" + SCORE_H[0]]) and np.isfinite(r[SCORE_H[0]])]
            return dict(종목=t, hit_p=hp, n_p=np_, hit_g=hg, n_g=ng, mae_m=wf["mae_m"], mae_c=wf["mae_c"], wf_hit=wf["hit"], wf_n=wf["nh"],
                        pts=pts, sig=sig_tally(rows_p))
        except Exception as e:  # noqa: BLE001
            return dict(종목=t, err=str(e)[:80])

    with ThreadPoolExecutor(max_workers=3) as ex:
        return list(ex.map(one, VAL_TICKERS.get(prof, [])))


# ----------------------------------------------------------------------
# 매수·매도 점수 (0~100)
# ----------------------------------------------------------------------
SCORE_H = ("6개월", 182)  # 점수의 기준 기간
RELIABLE_WIDTH = 20  # 80% 구간 폭이 이 점수 이하이면 한 점으로, 넘으면 범위로 보여줌


def _nb_score(pts, s_now, days):
    """판정 점수가 지금과 비슷했던 과거 시점들의 결과(값 > 0 이면 성공) → 0~100 점수와 80% 구간."""
    nb, width = [], None
    for width in (0.15, 0.25, 0.4, 2.0):
        nb = [a for s_, a in pts if abs(s_ - s_now) <= width]
        if len(nb) >= 24:
            break
    n = len(nb)
    k = sum(1 for a in nb if a > 0)
    n_eff = n * min(1.0, 30 / days)  # 매달 판정하므로 기간이 겹치는 만큼 독립 사례는 적다
    p_hat = k / n if n else 0.5
    a_, b_ = 2 + n_eff * p_hat, 2 + n_eff * (1 - p_hat)  # 베타(2,2) 사전분포 → 사례가 적으면 50점 쪽으로
    m = a_ / (a_ + b_)
    sd = np.sqrt(m * (1 - m) / (a_ + b_ + 1))
    lo, hi = max(0.0, m - 1.2816 * sd), min(1.0, m + 1.2816 * sd)
    return dict(mid=m * 100, lo=lo * 100, hi=hi * 100, n=n, n_eff=n_eff, k=k, width=width,
                mean=float(np.mean(nb)) if nb else np.nan)


def rec_score(rows, s_now, pool=None, absolute=False, mode=None):
    """mode: "capm" = 이후 6개월 수익률이 CAPM 기대수익률을 넘었나(알파 > 0) · "abs" = 수익률 자체가 플러스였나 ·
    "ind" = 같은 기간 업종 평균(섹터 ETF) 수익률을 넘었나."""
    mode = mode or ("abs" if absolute else "capm")
    lab, days = SCORE_H
    key = {"abs": lab, "capm": "a_" + lab, "ind": "i_" + lab}[mode]
    pi = {"capm": 1, "abs": 2, "ind": 3}[mode]
    own = [(r["score"] / r["n"], r[key]) for r in rows if r["n"] and np.isfinite(r.get(key, np.nan))]
    pl = [(p[0], p[pi]) for p in (pool or []) if len(p) > pi and np.isfinite(p[pi])]
    out = _nb_score(own + pl, s_now, days)
    out.update(own=sum(1 for s_, _a in own if abs(s_ - s_now) <= out["width"]), pooled=bool(pl))
    return out


def _phi(x):
    import math

    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def forward_score(c, R, fwd=None, prof="general"):
    """미래 추정치 기준: 1년 기대수익률 = (1 + EPS 성장률) × (PER가 기준 PER 쪽으로 k년에 걸쳐 돌아감) − 1 + 배당수익률.
    점수 = 6개월 수익률이 플러스일 확률 Φ(μ6 / σ6) × 100 (σ = 최근 2년 주간 변동성). 성장률·회귀 속도를 바꾼 시나리오로 범위를 낸다."""
    from research_data import ntm_eps

    h = c["h"]
    px = float(h.iloc[-1])
    eps_ttm, src = None, ""
    qf = c.get("qf")
    if qf is not None and "희석 EPS" in qf.index:
        e = pd.to_numeric(qf.loc["희석 EPS"], errors="coerce").dropna()
        if len(e):
            fl = pd.to_numeric(qf.loc["_filed_희석 EPS"], errors="coerce") if "_filed_희석 EPS" in qf.index else None
            fac = _split_factor(fl.reindex(e.index), c["splits"]).reindex(e.index).fillna(1.0) if fl is not None else 1.0
            eps_ttm, src = float((e / fac).iloc[-1]), "최근 4개 분기 합"
    tb = c["eps_tb"]
    notes = []
    eps_ann = float(tb["EPS"].iloc[-1]) if tb is not None and len(tb) else None
    if eps_ttm is not None and eps_ann and eps_ann > 0 and not (0.5 <= eps_ttm / eps_ann <= 2.0):
        notes.append(f"분기 합계 EPS({eps_ttm:.2f})가 직전 사업연도 EPS({eps_ann:.2f})와 너무 달라 사업연도 값을 썼습니다.")
        eps_ttm = None
    if eps_ttm is None and eps_ann:
        eps_ttm, src = eps_ann, "직전 사업연도"
    if not eps_ttm or eps_ttm <= 0:
        return dict(err="최근 12개월 이익이 적자라 PER 기반 미래 수익률을 계산할 수 없습니다.")
    per_now = px / eps_ttm
    target = np.nan
    if tb is not None and len(tb):
        step = _step_at(tb["EPS"], h.index)
        ph = (h / step.where(step > 0)).dropna()
        ph = ph[(ph.index >= c["start"]) & (ph.index <= c["end"]) & (ph > 0)]
        if len(ph) > 250:
            target = float(ph.median())
    gs = {}
    n_eps, _ = ntm_eps(R.get("info") or {}, R.get("eps_est"))
    if not (n_eps and n_eps > 0) and isinstance(fwd, pd.Series):
        f = _naive(fwd).dropna()
        if len(f) and f.iloc[-1] > 0:
            n_eps = px / float(f.iloc[-1])
    g_hist = float(c["g_date"].dropna().iloc[-1] / 100) if c["g_date"] is not None and c["g_date"].notna().any() else None
    if n_eps and n_eps > 0:
        g_c = n_eps / eps_ttm - 1
        if abs(g_c) > 0.6 and not (g_hist is not None and abs(g_hist) > 0.4 and np.sign(g_hist) == np.sign(g_c)):
            notes.append(f"예상 EPS({n_eps:,.2f})가 최근 EPS({eps_ttm:,.2f})보다 {g_c * 100:+.0f}% 달라 자료 오류 가능성이 있어 뺐습니다.")
            n_eps = None
        else:
            gs["컨센서스 (12개월 예상 EPS)"] = float(np.clip(g_c, -0.6, 1.0))
    if g_hist is not None:
        gs["과거 3년 EPS 성장률"] = float(np.clip(g_hist, -0.6, 1.0))
    if not gs:
        return dict(err="예상 EPS와 과거 성장률이 모두 없어 미래 수익률을 계산할 수 없습니다.")
    dy = 0.0
    sec = c["sec"]
    try:
        dv_ = pd.to_numeric(sec.loc["배당금"], errors="coerce").dropna()
        shr = pd.to_numeric(sec.loc["희석주식수"], errors="coerce").dropna()
        fac = _split_factor(pd.to_numeric(sec.loc["_filed_희석주식수"], errors="coerce"), c["splits"]).reindex(sec.columns).fillna(1.0) \
            if "_filed_희석주식수" in sec.index else pd.Series(1.0, index=sec.columns)
        if len(dv_) and len(shr):
            dy = float(abs(dv_.iloc[-1]) / (px * float(shr.iloc[-1] * fac[shr.index[-1]])))
            dy = dy if 0 <= dy < 0.2 else 0.0
    except Exception:  # noqa: BLE001
        dy = 0.0
    hw = c["hw"][c["hw"].index > h.index[-1] - pd.Timedelta(days=730)].dropna()
    sig = float(hw.std() * np.sqrt(52)) if len(hw) > 40 else 0.3
    sig = max(sig, 0.05)
    days = SCORE_H[1]

    def to_score(mu):
        mu_h = (1 + mu) ** (days / 365) - 1
        return 100 * _phi(mu_h / (sig * np.sqrt(days / 365)))

    if prof in ("cyclical", "reit"):  # 경기민감형은 PER 평균이 의미 없고, 리츠는 감가상각 때문에 PER 자체가 왜곡됨
        notes.append("이 업종은 PER이 기준값으로 돌아간다고 보기 어려워 PER 회귀를 빼고 계산했습니다.")
        target = np.nan
    ratio = float(np.clip(target / per_now, 0.67, 1.5)) if np.isfinite(target) else 1.0  # 극단적인 PER 차이는 줄여서 반영
    scen = []
    for gname, g in gs.items():
        for k in (2, 3, 5, None):
            rev = 1.0 if (k is None or not np.isfinite(target)) else ratio ** (1 / k)
            mu = (1 + g) * rev - 1 + dy
            scen.append(dict(g=gname, gv=g, k=k, mu=mu, score=to_score(mu)))
    base_g = "컨센서스 (12개월 예상 EPS)" if "컨센서스 (12개월 예상 EPS)" in gs else next(iter(gs))
    base = next(x for x in scen if x["g"] == base_g and x["k"] == (3 if np.isfinite(target) else None))
    lo, hi = min(x["score"] for x in scen), max(x["score"] for x in scen)
    return dict(mid=base["score"], lo=lo, hi=hi, mu=base["mu"], scen=scen, per_now=per_now, target=target, eps_ttm=eps_ttm,
                eps_src=src, n_eps=n_eps, dy=dy, sig=sig, consensus="컨센서스 (12개월 예상 EPS)" in gs, notes=notes,
                reliable=("컨센서스 (12개월 예상 EPS)" in gs) and (hi - lo) <= RELIABLE_WIDTH)


def _show_score(col, title, d, reliable, note=""):
    if reliable:
        col.metric(title, f"{d['mid']:.0f}점", score_label(d["mid"]), delta_color="normal" if d["mid"] >= 50 else "inverse")
        col.caption(f"80% 구간 {d['lo']:.0f}–{d['hi']:.0f}점. " + note)
    else:
        side = "매수 쪽" if d["lo"] >= 55 else ("매도 쪽" if d["hi"] <= 45 else "판단 보류")
        col.metric(title, f"{d['lo']:.0f}–{d['hi']:.0f}점", side, delta_color="off")
        col.caption(f"근거가 부족해 범위로 표시 (가운데 {d['mid']:.0f}점). " + note)


def _gauge_svg(title, d, reliable, sub=""):
    """0~100 원형 점수. 믿을 만하면 한 점(채운 호), 아니면 범위(옅은 호 + 가운데 표시)."""
    import html as _h

    r, sw, C = 44, 10, 2 * np.pi * 44
    lo, hi, mid = (max(0.0, min(100.0, d[k])) for k in ("lo", "hi", "mid"))
    if reliable:
        col = "#1B9E5A" if mid >= 55 else ("#D9534F" if mid <= 45 else "#8A8F98")
        arc = (f'<circle cx="60" cy="60" r="{r}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round" '
               f'stroke-dasharray="{mid / 100 * C:.1f} {C:.1f}" transform="rotate(-90 60 60)"/>')
        big, lab = f"{mid:.0f}", score_label(mid)
    else:
        col = "#1B9E5A" if lo >= 55 else ("#D9534F" if hi <= 45 else "#8A8F98")
        a = (max(hi - lo, 1.5)) / 100 * C
        arc = (f'<circle cx="60" cy="60" r="{r}" fill="none" stroke="{col}" stroke-opacity=".45" stroke-width="{sw}" '
               f'stroke-dasharray="{a:.1f} {C:.1f}" stroke-dashoffset="{-lo / 100 * C:.1f}" transform="rotate(-90 60 60)"/>'
               f'<circle cx="60" cy="60" r="{r}" fill="none" stroke="{col}" stroke-width="{sw + 4}" '
               f'stroke-dasharray="2 {C:.1f}" stroke-dashoffset="{-mid / 100 * C + 1:.1f}" transform="rotate(-90 60 60)"/>')
        big, lab = f"{lo:.0f}–{hi:.0f}", ("매수 쪽" if lo >= 55 else ("매도 쪽" if hi <= 45 else "판단 보류"))
    fs = 26 if reliable else 19
    return (f'<div style="flex:1 1 150px;max-width:220px;text-align:center">'
            f'<svg viewBox="0 0 120 120" width="132" height="132" role="img" aria-label="{_h.escape(title)} {big}점">'
            f'<circle cx="60" cy="60" r="{r}" fill="none" stroke="rgba(128,128,128,.22)" stroke-width="{sw}"/>{arc}'
            f'<text x="60" y="62" text-anchor="middle" font-size="{fs}" font-weight="700" fill="currentColor">{big}</text>'
            f'<text x="60" y="82" text-anchor="middle" font-size="11" fill="currentColor" opacity=".7">점</text></svg>'
            f'<div style="font-weight:600;font-size:14px;margin-top:-4px">{_h.escape(title)}</div>'
            f'<div style="font-size:13px;color:{col};font-weight:600">{lab}</div>'
            f'<div style="font-size:11px;opacity:.65;line-height:1.35;margin-top:2px">{_h.escape(sub)}</div></div>')


def gauges_html(items):
    return ('<div style="display:flex;flex-wrap:wrap;gap:10px;justify-content:space-around;margin:4px 0 2px">'
            + "".join(_gauge_svg(*it) for it in items) + "</div>")


def summary_scores(sec, R, ticker, sel, daily_names, build_daily, opts, prof, domestic=False):
    """맨 위 원형 점수 3개용: CAPM 미포함 · 업종 평균 대비 · CAPM 대비. 같은 입력이면 세션 안에서 다시 계산하지 않는다.
    build_daily() → {표시 이름: (일별 시계열, 단위)} (필요할 때만 계산)."""
    import stats

    use = [k for k in sel if k in DIR or k in GROWTH]
    pooled = not domestic  # 같은 업종 대표 종목 사례를 기본으로 합친다 (국내는 대표 종목 목록이 없어 이 종목만)
    per = R.get("_period")
    key = (ticker, prof, tuple(use), tuple(sorted(daily_names)), tuple(sorted(opts.items())), str(per), pooled,
           str(R["hist"].index[-1]) if isinstance(R.get("hist"), pd.Series) and len(R["hist"]) else "")
    memo = st.session_state.setdefault("_rs_summary", {})
    if key in memo:
        return memo[key]
    if not use and not daily_names and not opts.get("momentum"):
        return None
    daily = build_daily()
    rf_now = stats.risk_free_for(domestic, pd.Timestamp.now().date())[0]
    c = build_ctx(sec, R, ticker, daily, dict(opts, peers=False), domestic=domestic)
    rows, signals, _sk, flags = evaluate(c, prof, use, rf_now)
    if not rows:
        return None
    now_d = min(c["end"], c["h"].index[-1])
    sg = signals(now_d)
    sc, n = _score(sg)
    if not n:
        return None
    pool, pool_n = None, 0
    if pooled:
        try:
            res = [r for r in validate_profile(prof) if "pts" in r and r["종목"].upper() != ticker.upper()]
            pool = [tuple(p) for r in res for p in r["pts"]]
            pool_n = len(res)
        except Exception:  # noqa: BLE001
            pool = None
    fl = _flags_at(flags, c, now_d) if flags else []
    out = {}
    for m in ("abs", "ind", "capm"):
        d = rec_score(rows, sc / n, pool, mode=m)
        rel = d["hi"] - d["lo"] <= RELIABLE_WIDTH
        if fl:
            d.update(mid=min(d["mid"], 50), lo=min(d["lo"], 50), hi=min(d["hi"], 50))
        out[m] = (d, rel)
    res_ = dict(scores=out, verdict=_verdict(sc, n), flags=fl, sname=c["sname"], pooled=bool(pool), pool_n=pool_n, n_sig=len(sg))
    memo.clear()  # 최근 결과 하나만 보관
    memo[key] = res_
    return res_


def score_label(x):
    return "매수" if x >= 65 else ("약한 매수" if x >= 55 else ("중립·보유" if x > 45 else ("약한 매도" if x > 35 else "매도")))


def _val_on(prof):
    st.session_state[f"rs_val_{prof}"] = True


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
def render_outlook(sec, R: dict, ticker: str, sel: list, daily: dict, options=None, ms_key=None, domestic=False, fwd=None) -> None:
    """sel: 고른 연간 재무비율 이름, daily: {표시 이름: (일별 시계열, 단위)} — 고른 일별 밸류에이션."""
    from glossary import table_html
    from options import mgrid

    st.markdown("##### 상승·하락 판단")
    domestic = bool(domestic or "_ksec" in R)
    hist = R.get("hist")
    if sec is None or not isinstance(hist, pd.Series) or not len(hist):
        st.caption("재무제표나 주가가 없어 판단할 수 없습니다.")
        return

    # ---- 업종 유형
    info = R.get("info") or {}
    det = classify(info.get("sector"), info.get("industry"), R.get("_ksec"))
    sic_txt = ""
    if det == "general" and not R.get("_ksec") and not info.get("sector"):  # Yahoo가 막혔으면 SEC 업종 코드로
        try:
            sic, sic_txt = sec_sic(ticker)
            det = classify_sic(sic)
            sic_txt = f"SEC 업종 코드 {sic} {sic_txt}".strip() if sic else ""
        except Exception:  # noqa: BLE001
            sic_txt = ""
    keys = list(PROFILES)
    c1, c2 = st.columns([2, 3])
    prof = c1.selectbox("업종 유형", keys, index=keys.index(det), key=f"rs_prof_{ticker}",
                        format_func=lambda k: PROFILES[k]["name"] + (" (자동 판별)" if k == det else ""))
    P = PROFILES[prof]
    src = " · ".join(x for x in (info.get("sector"), info.get("industry"), R.get("_ksec"), sic_txt) if x)
    c2.caption((f"업종: {src}. " if src else "") + P["why"])
    if options is not None and ms_key:
        rec = [k for k in P["rec"] if k in options]
        if rec:
            names = {"PER (직전 사업연도 EPS)": "PER", "12M 선행 PER (컨센서스 추이)": "12M PER"}
            st.caption("이 업종에서 볼 지표: " + ", ".join(names.get(k, k) for k in rec if k != "주가"))
            st.button("이 지표들로 바꾸기", key=f"rs_rec_{ticker}_{prof}", on_click=_set_rec, args=(ms_key, rec))

    with st.expander("판정에 넣을 신호"):
        o1, o2 = st.columns(2)
        opts = dict(
            quarterly=o1.toggle("분기 실적으로 갱신 (최근 12개월)", value=not domestic, key=f"rs_o_q_{ticker}", disabled=domestic,
                                help="연간 재무제표 대신 분기마다 최근 12개월 합계로 재무비율을 다시 계산합니다 (미국 SEC 공시 종목)."),
            momentum=o1.toggle("모멘텀 (12-1개월 수익률)", value=True, key=f"rs_o_m_{ticker}",
                               help="최근 12개월 수익률(직전 1개월 제외)이 지수보다 5%p 넘게 높으면 상승 쪽, 낮으면 하락 쪽."),
            surprise=o1.toggle("실적 서프라이즈", value=True, key=f"rs_o_s_{ticker}",
                               help="최근 분기 실적이 발표 직전 컨센서스보다 얼마나 좋았나 (발표 후 91일 동안). 이 회사의 과거 서프라이즈와 비교합니다."),
            peers=o2.toggle("동종업계 비교", value=False, key=f"rs_o_p_{ticker}", disabled=domestic,
                            help="고른 재무비율이 동종업계 비교 탭의 회사들 가운데 상위·하위 20%인지. 처음 켜면 회사들 재무제표를 받느라 30초쯤 걸립니다."),
            flags=o2.toggle("위험 신호 거부권 (Altman Z'' · Beneish M)", value=True, key=f"rs_o_f_{ticker}",
                            help="부실 위험(Z'' 1.1 미만)이나 이익 조작 의심(M −1.78 초과)이면 '상승 우세'를 '중립'으로 낮춥니다."),
            extra=o2.toggle("추가 신호 (고점 대비 하락 · 현금 품질 · 주주환원 · 외국인 수급)", value=True,
                            key=f"rs_o_x_{ticker}",
                            help="업종별 대표 종목 검증에서 효과가 있었던 업종에만 씁니다. "
                                 "고점 대비 하락(경기민감형·국내 제외): 52주 고점보다 15% 넘게 빠지고 3개월 마이너스면 매도 쪽 · "
                                 "현금 품질(성장·기술형): 영업현금흐름 ÷ 순이익 1.1배 이상 매수 쪽, 0.8배 미만 매도 쪽 · "
                                 "주주환원(성장·기술형, 경기민감형): 배당+자사주 매입이 잉여현금흐름의 50% 이상이고 시가총액의 3% 이상이면 매수 쪽 · "
                                 "외국인 수급(국내): 60거래일 외국인 지분율이 0.5%p 넘게 늘면 매수 쪽, 줄면 매도 쪽이고, 늘 때는 PER 등이 비싸도 감점 안 함. "
                                 "이익 고점 위치와 PBR ÷ ROE는 검증에서 효과가 없어 뺐습니다. 조건에 걸릴 때만 근거에 들어갑니다."),
        )
        if domestic:
            st.caption("국내 종목은 분기 재무와 동종업계 비교를 아직 지원하지 않습니다 (DART 연간 재무제표 기준).")

    use = [k for k in sel if k in DIR or k in GROWTH]
    no_dir = [k for k in sel if k not in use]
    if not use and not daily and not opts["momentum"]:
        st.info("위 '그래프 항목'에서 재무비율이나 PER 같은 밸류에이션을 하나 이상 고르면 판단합니다.")
        return
    peers = None
    if opts["peers"]:
        try:
            from peers import us_peer_list

            peers, _desc = us_peer_list(ticker, info)
        except Exception:  # noqa: BLE001
            peers = None
    import stats

    rf_now, rf_lab = stats.risk_free_for(domestic, pd.Timestamp.now().date())
    with st.spinner("과거 시점마다 다시 판정하는 중..."):
        c = build_ctx(sec, R, ticker, daily, opts, domestic=domestic, peers=peers)
        rows, signals, skipped, flags = evaluate(c, prof, use, rf_now)
    if not rows:
        st.caption("고른 기간 안에 판정할 수 있는 시점이 없습니다.")
        return

    # ---- 지금 판단
    h = c["h"]
    now_d = min(c["end"], h.index[-1])
    now_sig = signals(now_d)
    sc, n = _score(now_sig)
    vd = _verdict(sc, n)
    if vd is None:
        st.caption("고른 항목의 값이 부족해 판단할 수 없습니다 (전년 값이 없거나 비교할 과거 값이 적음).")
        return
    fl_now = _flags_at(flags, c, now_d) if flags else []
    capped = bool(fl_now) and vd == "상승 우세"
    if capped:
        vd = "중립"
    msg = (f"**{vd}**{' (위험 신호로 낮춤)' if capped else ''} — 점수 {sc:+.1f} / {n:.1f} "
           f"({P['name']} 기준 · {c['freq']} 재무 · {now_d:%Y-%m-%d})")
    (st.success if vd == "상승 우세" else st.error if vd == "하락 우세" else st.info)(msg)
    if fl_now:
        st.warning("위험 신호: " + ", ".join(fl_now))

    # ---- 미래 추정치 점수 (CAPM 미포함·업종 대비·CAPM 대비 원형 점수는 이 섹션 위에 표시)
    try:
        rf_ = forward_score(c, R, fwd, prof)
    except Exception as e:  # noqa: BLE001
        rf_ = dict(err=f"계산 실패: {e}")
    k1, _k2 = st.columns([1, 2])
    if "err" in rf_:
        k1.metric("미래 추정치 점수 (6개월)", "계산 불가", delta_color="off")
        k1.caption(rf_["err"])
    else:
        _show_score(k1, "미래 추정치 점수 (6개월)", rf_, rf_["reliable"],
                    f"1년 기대수익률 {_p(rf_['mu'])}" + ("" if rf_["consensus"] else " (컨센서스 없음 → 과거 성장률 사용)") + ". 과거 검증 없음.")
    _k2.caption("예상 EPS 성장과 PER의 기준값 회귀로 낸 1년 기대수익률을 변동성으로 나눠, 6개월 수익률이 플러스일 확률로 바꾼 값입니다. "
                "과거 데이터로 계산하는 세 가지 점수(CAPM 미포함·업종 대비·CAPM 대비)는 이 그래프 섹션 바로 위 원형 점수에 있습니다.")
    if "err" not in rf_:
        with st.expander("미래 추정치 계산 내역"):
            mgrid([("현재 PER", f"{rf_['per_now']:.1f}배", f"주가 ÷ EPS({rf_['eps_src']}) {rf_['eps_ttm']:.2f}"),
                   ("기준 PER", f"{rf_['target']:.1f}배" if np.isfinite(rf_["target"]) else "-", "고른 기간의 PER 중앙값"),
                   ("12개월 예상 EPS", f"{rf_['n_eps']:.2f}" if rf_["n_eps"] else "-", "컨센서스"),
                   ("배당수익률", _p(rf_["dy"]).lstrip("+"), "최근 사업연도 배당금 ÷ 시가총액"),
                   ("변동성 (연)", _p(rf_["sig"]).lstrip("+"), "최근 2년 주간 수익률")])
            tb_ = [[x["g"], _p(x["gv"]), "회귀 안 함" if x["k"] is None else f"{x['k']}년", _p(x["mu"]), f"{x['score']:.0f}점"]
                   for x in rf_["scen"]]
            st.markdown(table_html(tb_, ["EPS 성장 가정", "성장률", "PER 회귀 기간", "1년 기대수익률", "점수"]), unsafe_allow_html=True)
            for t_ in rf_.get("notes", []):
                st.caption(t_)
            st.caption("1년 기대수익률 = (1 + EPS 성장률) × (기준 PER ÷ 현재 PER)^(1/회귀 기간) − 1 + 배당수익률 "
                       "(기준 PER ÷ 현재 PER은 0.67~1.5배로 제한). 과거 시점의 컨센서스 기록이 없어 이 점수는 과거 검증을 하지 못했습니다.")
    if c["peers"]:
        st.caption("동종업계: " + ", ".join(p for p, _ in c["peers"]))

    # ---- 향후 기대수익률 = CAPM 기대수익률 + 판정에 따른 보정(과거 같은 판정의 평균 알파, 사례 수만큼만 반영)
    beta_now = _beta(c, now_d)
    st.markdown("**향후 기대수익률**")
    ftb = []
    for lab, days in HORIZONS:
        capm = (1 + rf_now + beta_now * MRP) ** (days / 365) - 1
        same = [r for r in rows if r["verdict"] == vd and np.isfinite(r[lab])]
        a, ns = _alpha_est([r["a_" + lab] for r in same], days)
        base = [r["a_" + lab] for r in same] if len(same) >= 3 else [r["a_" + lab] for r in rows if np.isfinite(r[lab])]
        base = [x for x in base if np.isfinite(x)]
        rng = f"{_p(capm + np.percentile(base, 25))} ~ {_p(capm + np.percentile(base, 75))}" if base else "-"
        up = f"{np.mean([r[lab] > 0 for r in same]) * 100:.0f}%" if same else "-"
        ftb.append([lab, _p(capm + a), _p(capm), _p(a), up, rng, f"{ns}"])
    st.markdown(table_html(ftb, ["기간", "예측 기대수익률", "CAPM 기준", "판정 보정", "상승 확률", "예상 범위", "같은 판정 사례"]),
                unsafe_allow_html=True)
    st.caption(f"CAPM 기준 = 무위험수익률 {rf_now * 100:.2f}% + 베타 {beta_now:.2f} × 시장 위험프리미엄 {MRP * 100:.0f}% (가정). "
               f"판정 보정 = 과거 판정이 '{vd}'인 달들의 시장 대비 초과수익(알파) 평균을, 사례가 적을수록 0 쪽으로 당긴 값. "
               "예상 범위 = 그 알파의 하위 25%~상위 25%를 CAPM 기준에 더한 값. 무위험수익률: " + rf_lab)

    # ---- 워크포워드 성적
    wf = walk_forward(rows)
    wtb = []
    for lab, _days in HORIZONS:
        w = wf[lab]
        if not w["n"]:
            continue
        wtb.append([lab, _p(w["mae_m"]).lstrip("+"), _p(w["mae_c"]).lstrip("+"),
                    f"{w['hit']}/{w['nh']} ({w['hit'] / w['nh'] * 100:.0f}%)" if w["nh"] else "-", f"{w['n']}"])
    if wtb:
        st.markdown("**워크포워드 성적** — 각 시점에 그때까지 결과가 나온 사례만으로 예측했을 때")
        st.markdown(table_html(wtb, ["기간", "예측 오차 (이 방법)", "예측 오차 (CAPM만)", "초과수익 방향 적중", "시점 수"]),
                    unsafe_allow_html=True)
        st.caption("예측 오차 = 예측 기대수익률과 실제 수익률 차이의 평균 (작을수록 좋음). 이 방법의 오차가 CAPM만 쓴 오차보다 작아야 판정이 쓸모 있습니다.")

    past = [r for r in rows if np.isfinite(r["1년"])]
    hit, nh = _hit_rate(rows)
    items = [("점수 방향 적중률", f"{hit}/{nh} ({hit / nh * 100:.0f}%)" if nh else "-",
              f"{P['name']} 기준 · 점수 부호와 이후 1년 방향이 같았던 달의 비율")]
    if prof != "general":
        rows_g = evaluate(c, "general", use, rf_now)[0]
        bh, bn_ = _hit_rate(rows_g)
        items.append(("업종 반영 없이 적중률", f"{bh}/{bn_} ({bh / bn_ * 100:.0f}%)" if bn_ else "-", "같은 지표를 일반 기준으로 판정했을 때"))
    items.append(("베타", f"{beta_now:.2f}", f"최근 2년 주간 수익률 · {c['mname'] or '지수'} 기준 (블룸 조정)"))
    fs = fscore(sec, R.get("splits"))
    fs.index = [pd.Timestamp(x) for x in fs.index]
    if len(fs) and fs["항목"].iloc[-1] >= 5:
        items.append(("F-스코어", f"{int(fs['점수'].iloc[-1])}/{int(fs['항목'].iloc[-1])}", "재무 상태 개선 항목 수 (참고)"))
    mgrid(items)

    # ---- 근거별 신호
    trs = []
    for k, (s, desc, wt) in now_sig.items():
        hp_ = [r for r in past if k in r["sig"] and r["sig"][k][0] != 0 and r["1년"] != 0]
        hk = sum(1 for r in hp_ if np.sign(r["sig"][k][0]) == np.sign(r["1년"]))
        trs.append([k, desc, _sig_txt(s), _w(wt), f"{hk}/{len(hp_)}" if hp_ else "-"])
    st.markdown(table_html(trs, ["근거", "값", "신호", "가중치", "과거 적중 (1년)"]), unsafe_allow_html=True)
    notes = []
    if any("(성장 대비)" in k for k in now_sig):
        notes.append("(성장 대비) = PER ÷ 최근 3년 EPS 연평균 성장률(%). 이익이 늘지 않았으면 계산하지 않습니다.")
    if "평균이익 PER" in now_sig:
        notes.append("경기민감형이라 PER 대신 '평균이익 PER'(주가 ÷ 최근 최대 10년 평균 EPS)로 판단했습니다. 이익 고점의 낮은 PER에 속지 않기 위해서입니다.")
    for k, why in skipped.items():
        notes.append(f"{k}: 판단에서 뺐습니다 — {why}.")
    if no_dir:
        notes.append(f"{', '.join(no_dir)}: 높고 낮음의 좋고 나쁨이 정해져 있지 않아 판단에서 뺐습니다.")
    tr = R.get("eps_trend")
    if isinstance(tr, pd.DataFrame) and {"current", "90daysAgo"} <= set(tr.columns):
        parts = []
        for per, nm in (("0y", "올해"), ("+1y", "내년")):
            try:
                a_, b_ = float(tr.loc[per, "current"]), float(tr.loc[per, "90daysAgo"])
                if b_:
                    parts.append(f"{nm} {(a_ / b_ - 1) * 100:+.1f}%")
            except Exception:  # noqa: BLE001
                pass
        if parts:
            notes.append("실적 추정치 변화 (최근 90일 컨센서스): " + ", ".join(parts)
                         + ". 과거 시점의 추정치 기록이 없어 점수와 검증에는 넣지 않고 참고로만 보여줍니다.")
    if any(k.split(" (")[0] in VALN or "PER" in k or "PBR" in k or k in daily for k in now_sig):
        notes.append("밸류에이션(PER·PBR 등)은 후행 지표입니다. 이익이 바닥일 때는 비싸 보이고 고점일 때는 싸 보입니다."
                     + (" 그래서 외국인 지분율이 늘고 있으면 비싸 보여도 감점하지 않습니다." if domestic and opts.get("extra") else ""))
    if opts.get("extra") and domestic and c.get("frgn") is None:
        notes.append(f"외국인 수급 자료를 받지 못해 이 신호는 뺐습니다 ({c.get('frgn_err') or '자료 없음'}).")
    for t in notes:
        st.caption(t)

    # ---- 판정별 평균과 과거 판정 내역
    with st.expander("과거 판정과 이후 수익률"):
        sm = []
        for v_ in ("상승 우세", "중립", "하락 우세"):
            g_ = [r for r in rows if r["verdict"] == v_]
            line = [v_, f"{len(g_)}"]
            for lab, _d in HORIZONS:
                x = [r[lab] for r in g_ if np.isfinite(r[lab])]
                line.append(f"{_p(np.mean(x))} ({sum(1 for y in x if y > 0)}/{len(x)})" if x else "-")
            sm.append(line)
        st.markdown(table_html(sm, ["판정", "횟수"] + [f"{lab} 평균" for lab, _ in HORIZONS]), unsafe_allow_html=True)
        st.caption("괄호는 오른 횟수/전체 횟수입니다.")
        df = pd.DataFrame([[f"{r['d']:%Y-%m-%d}", f"{r['score']:+.1f}/{r['n']:.1f}", r["verdict"] + (" ⚠" if r["flags"] else "")]
                           + [_p(r[lab]) for lab, _ in HORIZONS] for r in reversed(rows)],
                          columns=["판정일", "점수", "판정"] + [lab for lab, _ in HORIZONS])
        st.dataframe(df, hide_index=True, use_container_width=True, height=320)
        st.caption(f"매달 첫 거래일에 그때까지 공시된 숫자({c['freq']})와 그날의 주가로 다시 판정했습니다. ⚠ = 위험 신호. "
                   "수익률은 판정일 종가부터 1·3·6개월, 1년 뒤 종가까지입니다 (배당 제외).")
        own = sig_tally(rows)
        if own:
            stb = [[k, f"{v[0]}/{v[1]}", f"{v[0] / v[1] * 100:.0f}%", _p(v[2] / v[1])]
                   for k, v in sorted(own.items(), key=lambda kv: -kv[1][1]) if v[1]]
            st.markdown("**이 종목의 신호별 성적** — 신호가 켜진 달의 이후 6개월 CAPM 대비 초과수익 방향")
            st.markdown(table_html(stb, ["신호", "적중/사례", "적중률", "평균 초과수익 (신호 방향)"]), unsafe_allow_html=True)

    # ---- 여러 종목 검증
    with st.expander(f"여러 종목으로 '{P['name']}' 규칙 검증"):
        st.caption(f"대표 종목 {', '.join(VAL_TICKERS[prof])}에 이 업종의 추천 지표·분기 재무·모멘텀·위험 신호를 똑같이 적용합니다 "
                   "(최근 10년, 동종업계 비교·실적 서프라이즈 제외). 처음 실행은 1~2분 걸리고 결과는 1주일 저장됩니다.")
        run_key = f"rs_val_{prof}"
        st.button("검증 실행", key=f"rs_valbtn_{ticker}_{prof}", on_click=_val_on, args=(prof,))
        if st.session_state.get(run_key):
            with st.spinner("대표 종목들을 다시 판정하는 중..."):
                res = validate_profile(prof)
            ok = [r for r in res if "err" not in r]
            tb = [[r["종목"], f"{r['hit_p']}/{r['n_p']}", f"{r['hit_g']}/{r['n_g']}", _p(r["mae_m"]).lstrip("+"),
                   _p(r["mae_c"]).lstrip("+")] for r in ok]
            if ok:
                sp = sum(r["hit_p"] for r in ok), sum(r["n_p"] for r in ok)
                sg_ = sum(r["hit_g"] for r in ok), sum(r["n_g"] for r in ok)
                mm = np.nanmean([r["mae_m"] for r in ok])
                mc = np.nanmean([r["mae_c"] for r in ok])
                tb.append(["합계·평균", f"{sp[0]}/{sp[1]} ({sp[0] / max(sp[1], 1) * 100:.0f}%)",
                           f"{sg_[0]}/{sg_[1]} ({sg_[0] / max(sg_[1], 1) * 100:.0f}%)", _p(mm).lstrip("+"), _p(mc).lstrip("+")])
                st.markdown(table_html(tb, ["종목", f"적중 ({P['name']})", "적중 (일반 기준)", "1년 예측 오차 (이 방법)",
                                            "1년 예측 오차 (CAPM만)"]), unsafe_allow_html=True)
            agg = {}
            for r in ok:
                for k, (hk, nk, sa) in (r.get("sig") or {}).items():
                    t = agg.setdefault(k, [0, 0, 0.0])
                    t[0], t[1], t[2] = t[0] + hk, t[1] + nk, t[2] + sa
            if agg:
                stb = [[k, f"{v[0]}/{v[1]}", f"{v[0] / v[1] * 100:.0f}%", _p(v[2] / v[1])]
                       for k, v in sorted(agg.items(), key=lambda kv: -kv[1][1]) if v[1]]
                st.markdown("**신호별 성적** — 신호가 켜진 달의 이후 6개월 CAPM 대비 초과수익 방향")
                st.markdown(table_html(stb, ["신호", "적중/사례", "적중률", "평균 초과수익 (신호 방향)"]), unsafe_allow_html=True)
                st.caption("적중률이 50% 근처이고 평균 초과수익이 0 근처면 그 신호는 이 업종에서 효과가 없다는 뜻입니다. "
                           "매달 판정하므로 이웃한 달끼리 기간이 겹쳐 실제 독립 사례는 더 적습니다.")
            bad = [r for r in res if "err" in r]
            if bad:
                st.caption("불러오지 못한 종목: " + ", ".join(f"{r['종목']}({r['err']})" for r in bad))

    st.caption("과거 통계로 낸 추정이며 예측을 보장하지 않습니다. 매달 판정하므로 이웃한 달의 수익률은 기간이 겹쳐, "
               "실제로 독립된 사례 수는 표의 횟수보다 적습니다.")
