"""기업 리서치: 고른 재무비율로 '상승 우세 / 중립 / 하락 우세'를 판단하고, 같은 방법을 과거에 적용했을 때의 결과를 보여준다.

방법 (방법론 해설 문서의 '앱에 넣는다면' 설계)
- 수익성·안정성·효율성·현금흐름 비율: 전년보다 좋아졌으면 +1, 나빠졌으면 −1 (피오트로스키 F-스코어 방식)
- 성장률: 플러스 성장 +1, 마이너스 성장 −1
- 밸류에이션(PER 등): 고른 기간 안에서 그때까지의 값 중 상위 20%면 −1, 하위 20%면 +1 (자기 과거 대비 평균회귀)
- 점수 합 ÷ 근거 수가 +0.25 이상이면 상승 우세, −0.25 이하면 하락 우세
과거 검증: 기간 안의 매달 첫 거래일마다 그때까지 공시된 숫자로 다시 판정하고 그 뒤 1·3·6개월·1년 주가 수익률을 붙인다.
기대수익률: 과거에 지금과 같은 판정이었던 달들의 이후 수익률 평균 (사례가 적으면 기간 전체 평균 쪽으로 당김).
업종 유형(성장·기술 / 경기민감 / 금융 / 리츠 / 방어)에 따라 지표 가중치와 PER을 읽는 방법을 바꾼다 (PROFILES).
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
#   normal = 자기 과거 대비 비싸면 하락 쪽 / peg = 이익 성장률로 나눈 PER(PEG)로 비교 / cycle = PER 대신 이익 사이클 위치
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
            "PER 대신 '이익이 과거 대비 어디쯤인가(이익 사이클)'를 거꾸로 읽고, PBR·EV/EBITDA·재무 안정성에 무게를 둡니다."),
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


def _annual_signal(k, row, j, pcols, kind):
    """연간 재무비율의 j번째 사업연도 신호 → (신호, 설명) 또는 None."""
    v = row.iloc[j]
    if not np.isfinite(v):
        return None
    unit = RATIOS[k][0] if k in RATIOS else "배"
    fmt = _fmt(unit)
    if kind in ("level", "peg"):
        past = np.array([row[c] for c in pcols if c <= row.index[j] and np.isfinite(row[c])])
        if len(past) < 3:
            return None
        p = _pctile(past, v)
        lab = f"PEG {v:.2f}" if kind == "peg" else fmt(v)
        return _val_sig(p, DIR.get(k, -1)), f"{lab} · 기간 내 백분위 {p * 100:.0f}%"
    if kind == "growth":
        return (1 if v > 1 else (-1 if v < -1 else 0)), fmt(v)
    if j == 0:
        return None
    prev = row.iloc[j - 1]
    if not np.isfinite(prev):
        return None
    ch = v - prev
    sig = int(np.sign(ch)) * DIR[k] if abs(ch) > 0.02 * abs(prev) else 0
    return sig, f"{fmt(prev)} → {fmt(v)}"


def _daily_signal(s, unit, start, d, peg=False):
    vals = s[(s.index >= start) & (s.index <= pd.Timestamp(d))].dropna()
    if len(vals) < 120:
        return None
    v = float(vals.iloc[-1])
    p = _pctile(vals.values, v)
    direction = 1 if unit == "%" else -1  # 이익수익률·배당수익률 같은 %는 높을수록 쌈
    lab = f"PEG {v:.2f}" if peg else (f"{v:.2f}%" if unit == "%" else f"{v:.1f}배")
    return _val_sig(p, direction), f"{lab} · 기간 내 백분위 {p * 100:.0f}%"


def _cycle_signal(eps_tb, d):
    """이익 사이클 위치: 그날까지 공시된 연간 EPS 중 최근 값의 백분위. 고점권이면 하락 쪽, 바닥권이면 상승 쪽."""
    if eps_tb is None:
        return None
    e = eps_tb["EPS"][eps_tb.index <= pd.Timestamp(d)].dropna()
    if len(e) < 5:
        return None
    v = float(e.iloc[-1])
    p = _pctile(e.values, v)
    where = "고점권" if p >= 0.8 else ("바닥권" if p <= 0.2 else "중간")
    return (-1 if p >= 0.8 else (1 if p <= 0.2 else 0)), f"EPS {v:.2f} · 과거 대비 백분위 {p * 100:.0f}% ({where})"


def _make_signals(prof, use, rt, cols, pcols, dser, start, sec, splits):
    """업종 유형 prof 기준 신호 함수 → signals(j, d) = {이름: (신호, 설명, 가중치)}, 제외 목록."""
    P = PROFILES[prof]
    skipped = {}
    ann = []  # (이름, 행, 종류)
    g_date, g_fy = _eps_growth(sec, splits) if P["earn"] == "peg" else (None, None)
    for k in use:
        if k in P["skip"]:
            skipped[k] = P["skip"][k]
            continue
        if k in VALN:
            if k in EARN_KEYS and P["earn"] == "cycle":
                continue
            if k in EARN_KEYS and P["earn"] == "peg":
                g = g_fy.reindex(cols) if g_fy is not None else pd.Series(np.nan, index=cols)
                ann.append((k, (rt.loc[k] / g.where(g > 0)), "peg"))
            else:
                ann.append((k, rt.loc[k], "level"))
        elif k in GROWTH:
            ann.append((k, rt.loc[k], "growth"))
        else:
            ann.append((k, rt.loc[k], "change"))
    daily = []  # (이름, 시계열, 단위, peg)
    has_earn = any(k in EARN_KEYS for k in use) or any(nm in EARN_KEYS for nm in dser)
    for nm, (s, u) in dser.items():
        if nm in P["skip"]:
            skipped[nm] = P["skip"][nm]
            continue
        if nm in EARN_KEYS and P["earn"] == "cycle":
            continue
        if nm in EARN_KEYS and P["earn"] == "peg" and u != "%":
            if g_date is None:
                g_date, _ = _eps_growth(sec, splits)
            g = _step_at(g_date, s.index)
            daily.append((nm, s / g.where(g > 0).values, u, True))
        else:
            daily.append((nm, s, u, False))
    eps_tb = annual_eps_table(sec, splits) if (P["earn"] == "cycle" and has_earn) else None

    def signals(j, d):
        out = {}
        for k, row, kind in ann:
            r = _annual_signal(k, row, j, pcols, kind)
            if r is not None:
                out[k + (" (성장 대비)" if kind == "peg" else "")] = (r[0], r[1], P["w"].get(k, 1.0))
        for nm, s, u, peg in daily:
            r = _daily_signal(s, u, start, d, peg)
            if r is not None:
                out[nm + (" (성장 대비)" if peg else "")] = (r[0], r[1], P["w"].get(nm, 1.0))
        if eps_tb is not None:
            r = _cycle_signal(eps_tb, d)
            if r is not None:
                out["이익 사이클 위치"] = (r[0], r[1], 1.5)
        return out

    return signals, skipped


def _score(sg: dict):
    sc = sum(v[0] * v[2] for v in sg.values())
    n = sum(v[2] for v in sg.values())
    return sc, n


def _w(x) -> str:
    return f"{x:g}"


def _set_rec(key, vals):
    st.session_state[key] = vals


def render_outlook(sec, R: dict, ticker: str, sel: list, daily: dict, options=None, ms_key=None) -> None:
    """sel: 고른 연간 재무비율 이름, daily: {표시 이름: (일별 시계열, 단위)} — 고른 일별 밸류에이션."""
    from glossary import table_html
    from options import mgrid

    st.markdown("##### 상승·하락 판단")
    hist = R.get("hist")
    if sec is None or not isinstance(hist, pd.Series) or not len(hist):
        st.caption("재무제표나 주가가 없어 판단할 수 없습니다.")
        return

    # ---- 업종 유형
    info = R.get("info") or {}
    det = classify(info.get("sector"), info.get("industry"), R.get("_ksec"))
    keys = list(PROFILES)
    c1, c2 = st.columns([2, 3])
    prof = c1.selectbox("업종 유형", keys, index=keys.index(det), key=f"rs_prof_{ticker}",
                        format_func=lambda k: PROFILES[k]["name"] + (" (자동 판별)" if k == det else ""))
    P = PROFILES[prof]
    src = " · ".join(x for x in (info.get("sector"), info.get("industry"), R.get("_ksec")) if x)
    c2.caption((f"업종: {src}. " if src else "") + P["why"])
    if options is not None and ms_key:
        rec = [k for k in P["rec"] if k in options]
        if rec:
            names = {"PER (직전 사업연도 EPS)": "PER", "12M 선행 PER (컨센서스 추이)": "12M PER"}
            st.caption("이 업종에서 볼 지표: " + ", ".join(names.get(k, k) for k in rec if k != "주가"))
            st.button("이 지표들로 바꾸기", key=f"rs_rec_{ticker}_{prof}", on_click=_set_rec, args=(ms_key, rec))

    use = [k for k in sel if k in DIR or k in GROWTH]
    no_dir = [k for k in sel if k not in use]
    if not use and not daily:
        st.info("위 '그래프 항목'에서 재무비율이나 PER 같은 밸류에이션을 하나 이상 고르면 판단합니다.")
        return
    h = _naive(hist)
    rt = compute_ratios(sec, hist, R.get("splits"))
    cols = [pd.Timestamp(c) for c in rt.columns]
    rt.columns = cols
    period = R.get("_period")
    start = pd.Timestamp(period[0]) if period else cols[0]
    end = pd.Timestamp(period[1]) + pd.Timedelta(days=1) if period else h.index[-1]
    pcols = [c for c in cols if start <= c <= end]
    if not pcols:
        st.caption("고른 기간 안에 사업연도가 없어 판단할 수 없습니다.")
        return
    avail = _avail_dates(sec, cols)
    dser = {nm: (_naive(s), u) for nm, (s, u) in daily.items() if isinstance(s, pd.Series) and len(s)}
    fs = fscore(sec, R.get("splits"))
    fs.index = [pd.Timestamp(c) for c in fs.index]
    signals, skipped = _make_signals(prof, use, rt, cols, pcols, dser, start, sec, R.get("splits"))
    base_sig = signals if prof == "general" else _make_signals("general", use, rt, cols, pcols, dser, start, sec, R.get("splits"))[0]

    mkt, mname = None, None
    sym = "^KS11" if ticker.endswith(".KS") else ("^KQ11" if ticker.endswith(".KQ") else "^GSPC")
    try:
        mkt = _naive(_index_hist(sym))
        mname = {"^KS11": "코스피", "^KQ11": "코스닥", "^GSPC": "S&P 500"}[sym]
    except Exception:  # noqa: BLE001
        mkt = None

    # ---- 과거 판정: 기간 안의 매달 첫 거래일마다 그때까지 공시된 최근 사업연도 + 그날 밸류에이션으로 다시 판정
    def fy_at(d):
        ok = [c for c in cols if avail[c] <= d]
        return ok[-1] if ok else None

    hp = h[(h.index >= start) & (h.index <= end)]
    dates = list(hp.groupby([hp.index.year, hp.index.month]).head(1).index)
    rows = []
    for d in dates:
        c = fy_at(d)
        if c is None:
            continue
        j = cols.index(c)
        sg = signals(j, d)
        sc, n = _score(sg)
        rec_ = dict(d=d, fy=c, sig=sg, score=sc, n=n, verdict=_verdict(sc, n))
        if base_sig is not signals:
            bsc, bn = _score(base_sig(j, d))
            rec_["bscore"] = bsc
        for lab, days in HORIZONS:
            rec_[lab] = _fwd(h, d, days)
        m = _fwd(mkt, d, 365) if mkt is not None else np.nan
        rec_["ex"] = rec_["1년"] - m if np.isfinite(rec_["1년"]) and np.isfinite(m) else np.nan
        rows.append(rec_)
    if not rows:
        st.caption("고른 기간 안에 판정할 수 있는 시점이 없습니다.")
        return

    # ---- 지금 판단: 가장 최근 공시 사업연도 재무비율 + 오늘 기준 밸류에이션
    now_d = min(end, h.index[-1])
    fy_now = fy_at(now_d)
    now_sig = signals(cols.index(fy_now), now_d)
    sc, n = _score(now_sig)
    vd = _verdict(sc, n)
    if vd is None:
        st.caption("고른 항목의 값이 부족해 판단할 수 없습니다 (전년 값이 없거나 비교할 과거 값이 적음).")
        for k, why in skipped.items():
            st.caption(f"{k}: {why}")
        return
    msg = (f"**{vd}** — 점수 {sc:+.1f} / {n:.1f} ({P['name']} 기준, {fy_now:%Y}년 사업연도 공시 {avail[fy_now]:%Y-%m-%d}, "
           f"밸류에이션은 {now_d:%Y-%m-%d} 기준)")
    (st.success if vd == "상승 우세" else st.error if vd == "하락 우세" else st.info)(msg)

    # ---- 향후 기대수익률: 과거에 같은 판정이었던 때의 수익률 (사례가 적으면 전체 평균 쪽으로 당김)
    st.markdown("**향후 기대수익률**")
    ftb = []
    for lab, days in HORIZONS:
        allr = np.array([r[lab] for r in rows if np.isfinite(r[lab])])
        same = np.array([r[lab] for r in rows if np.isfinite(r[lab]) and r["verdict"] == vd])
        if not len(allr):
            ftb.append([lab, "-", "-", "-", "-", "0"])
            continue
        n_eff = len(same) * min(1.0, 30 / days)  # 매달 판정하므로 기간이 겹치는 만큼 독립 사례는 적다
        w = n_eff / (n_eff + 3) if len(same) else 0.0
        exp_r = w * (same.mean() if len(same) else 0.0) + (1 - w) * allr.mean()
        base = same if len(same) >= 3 else allr
        lo, hi = np.percentile(base, [25, 75])
        ftb.append([lab, _p(exp_r), f"{(same > 0).mean() * 100:.0f}%" if len(same) else "-",
                    f"{_p(lo)} ~ {_p(hi)}", _p(allr.mean()), f"{len(same)}"])
    st.markdown(table_html(ftb, ["기간", "예측 기대수익률", "상승 확률", "예상 범위", "기간 전체 평균", "같은 판정 사례"]),
                unsafe_allow_html=True)
    st.caption(f"과거 판정이 '{vd}'인 달들의 이후 수익률로 계산합니다. 사례가 적을수록 기간 전체 평균 쪽으로 당겨서 계산하고, "
               "예상 범위는 그 수익률의 하위 25%~상위 25%입니다 (같은 판정 사례가 3개 미만이면 전체 기준).")

    past = [r for r in rows if np.isfinite(r["1년"])]

    def hit_rate(key):
        dd = [r for r in past if r.get(key, 0) != 0 and r["1년"] != 0]
        k_ = sum(1 for r in dd if np.sign(r[key]) == np.sign(r["1년"]))
        return k_, len(dd)

    hit, nh = hit_rate("score")
    exs = [r["ex"] for r in past if r["verdict"] == vd and np.isfinite(r["ex"])]
    items = [("점수 방향 적중률", f"{hit}/{nh} ({hit / nh * 100:.0f}%)" if nh else "-",
              f"{P['name']} 기준 · 점수 부호와 이후 1년 방향이 같았던 달의 비율")]
    if base_sig is not signals:
        bh, bn_ = hit_rate("bscore")
        items.append(("업종 반영 없이 적중률", f"{bh}/{bn_} ({bh / bn_ * 100:.0f}%)" if bn_ else "-", "같은 지표를 일반 기준으로 판정했을 때"))
    items.append((f"{mname} 대비 초과수익" if mname else "시장 대비 초과수익", _p(np.mean(exs)) if exs else "-",
                  f"과거 판정이 '{vd}'일 때 1년 평균"))
    fl = fs.loc[fy_now] if fy_now in fs.index else None
    if fl is not None and fl["항목"] >= 5:
        items.append(("F-스코어", f"{int(fl['점수'])}/{int(fl['항목'])}", "재무 상태 개선 항목 수 (참고)"))
    mgrid(items)

    # ---- 근거별 신호
    trs = []
    for k, (s, desc, wt) in now_sig.items():
        hp_ = [r for r in past if k in r["sig"] and r["sig"][k][0] != 0 and r["1년"] != 0]
        hk = sum(1 for r in hp_ if np.sign(r["sig"][k][0]) == np.sign(r["1년"]))
        trs.append([k, desc, _sig_txt(s), _w(wt), f"{hk}/{len(hp_)}" if hp_ else "-"])
    st.markdown(table_html(trs, ["근거", "값", "신호", "가중치", "과거 적중 (1년)"]), unsafe_allow_html=True)
    notes = []
    if P["earn"] == "peg" and any("(성장 대비)" in k for k in now_sig):
        notes.append("(성장 대비) = PER ÷ 최근 3년 EPS 연평균 성장률(%). 이익이 늘지 않았으면 계산하지 않습니다.")
    if P["earn"] == "cycle" and (any(k in EARN_KEYS for k in use) or any(k in EARN_KEYS for k in dser)):
        notes.append("경기민감형이라 PER은 '이익 사이클 위치'로 바꿔 판단했습니다 (이익이 고점권이면 하락 쪽, 바닥권이면 상승 쪽).")
    for k, why in skipped.items():
        notes.append(f"{k}: 판단에서 뺐습니다 — {why}.")
    if no_dir:
        notes.append(f"{', '.join(no_dir)}: 높고 낮음의 좋고 나쁨이 정해져 있지 않아 판단에서 뺐습니다.")
    for t in notes:
        st.caption(t)

    # ---- 판정별 평균과 과거 판정 내역
    with st.expander("과거 판정과 이후 수익률", expanded=True):
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
        df = pd.DataFrame([[f"{r['d']:%Y-%m-%d}", f"{r['fy']:%Y}", f"{r['score']:+.1f}/{r['n']:.1f}", r["verdict"] or "-"]
                           + [_p(r[lab]) for lab, _ in HORIZONS] for r in reversed(rows)],
                          columns=["판정일", "사업연도", "점수", "판정"] + [lab for lab, _ in HORIZONS])
        st.dataframe(df, hide_index=True, use_container_width=True, height=320)
        st.caption("매달 첫 거래일에 그때까지 공시된 숫자와 그날의 밸류에이션으로 다시 판정했습니다. 수익률은 판정일 종가부터 "
                   "1·3·6개월, 1년 뒤 종가까지입니다 (배당 제외). 아직 기간이 안 지난 칸은 비어 있습니다.")
    st.caption("과거 통계로 낸 추정이며 예측을 보장하지 않습니다. 매달 판정하므로 이웃한 달의 1년 수익률은 기간이 겹쳐, "
               "실제로 독립된 사례 수는 표의 횟수보다 적습니다.")
