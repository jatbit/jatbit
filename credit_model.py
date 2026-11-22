"""신용위험 계산 (화면 없음): 등급 표·누적 부도율 표, 추정 신용등급, 스프레드 내재 부도확률, 머튼(KMV) 구조모형,
CDS 가격(축약형 모형), Altman Z. 화면은 credit.py.
"""
import math

import numpy as np

from bsm import ncdf

# Damodaran "Ratings, Interest Coverage Ratios and Default Spread" (data as of January 2026)
# (이자보상배율 하한, 등급, 스프레드) — 대형 비금융 기업
DAMO_LARGE = [(8.5, "AAA", 0.0040), (6.5, "AA", 0.0055), (5.5, "A+", 0.0070), (4.25, "A", 0.0078), (3.0, "A-", 0.0089),
              (2.5, "BBB", 0.0111), (2.25, "BB+", 0.0138), (2.0, "BB", 0.0184), (1.75, "B+", 0.0275), (1.5, "B", 0.0321),
              (1.25, "B-", 0.0509), (0.8, "CCC", 0.0885), (0.65, "CC", 0.1261), (0.2, "C", 0.1600), (-1e9, "D", 0.1900)]
# 금융 서비스 기업 (장기 이자보상배율 기준)
DAMO_FIN = [(3.0, "AAA", 0.0040), (2.5, "AA", 0.0055), (2.0, "A+", 0.0070), (1.5, "A", 0.0078), (1.2, "A-", 0.0089),
            (0.9, "BBB", 0.0111), (0.75, "BB+", 0.0138), (0.6, "BB", 0.0184), (0.5, "B+", 0.0275), (0.4, "B", 0.0321),
            (0.3, "B-", 0.0509), (0.2, "CCC", 0.0885), (0.1, "CC", 0.1261), (0.05, "C", 0.1600), (-1e9, "D", 0.1900)]

# S&P Global Ratings, 2024 Annual Global Corporate Default And Rating Transition Study, Table 24
# 전 세계 기업 평균 누적 부도율(%) 1981–2024, 1~15년
SP_CUM = {
    "AAA": [0.00, 0.03, 0.13, 0.23, 0.34, 0.44, 0.49, 0.57, 0.62, 0.67, 0.70, 0.73, 0.75, 0.81, 0.86],
    "AA": [0.02, 0.05, 0.11, 0.19, 0.28, 0.37, 0.45, 0.52, 0.59, 0.65, 0.71, 0.76, 0.81, 0.86, 0.90],
    "A": [0.05, 0.11, 0.19, 0.29, 0.39, 0.51, 0.65, 0.78, 0.90, 1.03, 1.14, 1.25, 1.35, 1.45, 1.56],
    "BBB": [0.14, 0.38, 0.67, 1.01, 1.36, 1.71, 2.00, 2.30, 2.58, 2.86, 3.13, 3.35, 3.56, 3.78, 4.01],
    "BB": [0.56, 1.76, 3.12, 4.48, 5.75, 6.93, 7.94, 8.86, 9.68, 10.44, 11.06, 11.65, 12.17, 12.60, 13.05],
    "B": [2.93, 6.93, 10.46, 13.31, 15.60, 17.45, 18.90, 20.06, 21.08, 22.02, 22.82, 23.43, 24.02, 24.57, 25.11],
    "CCC/C": [26.12, 35.92, 41.32, 44.35, 46.53, 47.57, 48.61, 49.29, 49.89, 50.43, 50.85, 51.32, 51.86, 52.26, 52.30],
}

# FRED: ICE BofA 미국 회사채 지수 (OAS: 국채 대비 옵션조정 스프레드, EY: 실효수익률), 단위 %
ICE = {"AAA": ("BAMLC0A1CAAA", "BAMLC0A1CAAAEY"), "AA": ("BAMLC0A2CAA", "BAMLC0A2CAAEY"), "A": ("BAMLC0A3CA", "BAMLC0A3CAEY"),
       "BBB": ("BAMLC0A4CBBB", "BAMLC0A4CBBBEY"), "BB": ("BAMLH0A1HYBB", "BAMLH0A1HYBBEY"), "B": ("BAMLH0A2HYB", "BAMLH0A2HYBEY"),
       "CCC 이하": ("BAMLH0A3HYC", "BAMLH0A3HYCEY")}
OTHER = {"DGS1": "미 국채 1년", "DGS5": "미 국채 5년", "DGS10": "미 국채 10년", "DAAA": "무디스 Aaa 회사채", "DBAA": "무디스 Baa 회사채",
         "BAMLC0A0CM": "투자등급 전체 OAS", "BAMLH0A0HYM2": "하이일드 전체 OAS"}
COLORS = {"AAA": "#1B7F3B", "AA": "#3FA35B", "A": "#4C9BE8", "BBB": "#9B6FD6", "BB": "#F2A93B", "B": "#E5534B", "CCC 이하": "#8B1E1E"}
FIN_SECTORS = ("Financial Services",)


def bucket(rating: str) -> str:
    """세부 등급 → 큰 등급 (부도율 표·시장 스프레드 조회용)."""
    r = rating.rstrip("+-")
    if r in ("AAA", "AA", "A", "BBB", "BB", "B"):
        return r
    return "CCC/C"


def synthetic_rating(coverage: float, financial: bool = False):
    tbl = DAMO_FIN if financial else DAMO_LARGE
    if coverage is None or np.isnan(coverage):
        return None
    for lo, rt, sp in tbl:
        if coverage > lo:
            return rt, sp
    return tbl[-1][1], tbl[-1][2]


def spread_pd(spread: float, T: float, recovery: float = 0.4) -> float:
    """스프레드 ≈ 부도강도 × (1 − 회수율) → T년 내 부도확률 = 1 − exp(−λT) (위험중립)."""
    lam = spread / (1 - recovery)
    return 1 - math.exp(-lam * T)


def merton(E: float, sigE: float, D: float, r: float, T: float = 1.0, mu: float | None = None) -> dict:
    """E = V·N(d1) − D·e^(−rT)·N(d2),  σE·E = N(d1)·σV·V 를 반복법으로 풀이."""
    if not (E > 0 and sigE > 0 and D > 0):
        return {"ok": False}
    sv = sigE * E / (E + D)
    V = E + D * math.exp(-r * T)
    for _ in range(200):
        def eq(v):
            d1 = (math.log(v / D) + (r + 0.5 * sv * sv) * T) / (sv * math.sqrt(T))
            return v * ncdf(d1) - D * math.exp(-r * T) * ncdf(d1 - sv * math.sqrt(T)) - E
        lo, hi = E, E + D * 2 + 1
        while eq(hi) < 0:
            hi *= 2
        for _ in range(100):
            m = (lo + hi) / 2
            if eq(m) > 0:
                hi = m
            else:
                lo = m
        V = (lo + hi) / 2
        d1 = (math.log(V / D) + (r + 0.5 * sv * sv) * T) / (sv * math.sqrt(T))
        sv_new = sigE * E / (ncdf(d1) * V)
        if abs(sv_new - sv) < 1e-9:
            sv = sv_new
            break
        sv = 0.5 * sv + 0.5 * sv_new
    d1 = (math.log(V / D) + (r + 0.5 * sv * sv) * T) / (sv * math.sqrt(T))
    d2 = d1 - sv * math.sqrt(T)
    debt_val = V - E  # 부채의 시장가치 = 무위험 채권 − 풋옵션
    y = -math.log(debt_val / (D * math.exp(-r * T))) / T + r if debt_val > 0 else float("nan")  # 부채의 만기수익률(연속복리)
    out = {"ok": True, "V": V, "sigV": sv, "d1": d1, "d2": d2, "DD": d2, "PD": ncdf(-d2), "lev": D / V,
           "debt_val": debt_val, "spread": max(y - r, 0.0)}
    if mu is not None:
        dd_p = (math.log(V / D) + (mu - 0.5 * sv * sv) * T) / (sv * math.sqrt(T))
        out.update({"DD_p": dd_p, "PD_p": ncdf(-dd_p)})
    return out


# ----------------------------------------------------------------------
# CDS (축약형 모형: 구간별 일정한 부도강도, 분기 지급, 부도 시 경과 프리미엄 지급)
# ----------------------------------------------------------------------
def surv_from_cum(cum_pd: dict):
    """{연도: 누적 부도확률} → 생존확률 Q(t) 함수 (로그 선형 보간 = 구간별 일정한 부도강도)."""
    ts = np.array([0.0] + sorted(cum_pd))
    qs = np.array([1.0] + [max(1 - cum_pd[t], 1e-12) for t in sorted(cum_pd)])
    lq = np.log(qs)
    lam_last = (lq[-2] - lq[-1]) / (ts[-1] - ts[-2]) if len(ts) > 2 else 0.0

    def Q(t):
        t = np.asarray(t, float)
        out = np.interp(t, ts, lq)
        out = np.where(t > ts[-1], lq[-1] - lam_last * (t - ts[-1]), out)
        return np.exp(out)
    return Q


def surv_flat(lam: float):
    return lambda t: np.exp(-lam * np.asarray(t, float))


def cds_price(Q, T: float, r: float, R: float = 0.4, freq: int = 4, steps_per_period: int = 3) -> dict:
    """적정 CDS 스프레드 = 보호 레그 현재가치 ÷ RPV01 (프리미엄 1당 현재가치)."""
    n = max(int(round(T * freq)), 1)
    dt = T / n
    t = np.arange(1, n + 1) * dt
    df = lambda x: np.exp(-r * np.asarray(x, float))  # noqa: E731
    q = Q(t)
    q_prev = Q(t - dt)
    rpv01 = float(np.sum(dt * df(t) * q) + np.sum(0.5 * dt * df(t - dt / 2) * (q_prev - q)))  # 정기 지급 + 부도 시 경과분
    g = np.linspace(0, T, n * steps_per_period + 1)
    qg = Q(g)
    prot = float((1 - R) * np.sum(df((g[1:] + g[:-1]) / 2) * (qg[:-1] - qg[1:])))
    return {"par": prot / rpv01 if rpv01 > 0 else float("nan"), "rpv01": rpv01, "prot": prot, "pd": float(1 - Q(T))}


def implied_hazard(spread: float, T: float, r: float, R: float = 0.4) -> float:
    """시장 CDS 스프레드와 맞는 일정한 부도강도 (이분법)."""
    lo, hi = 0.0, 5.0
    for _ in range(100):
        m = (lo + hi) / 2
        if cds_price(surv_flat(m), T, r, R)["par"] > spread:
            hi = m
        else:
            lo = m
    return (lo + hi) / 2


def merton_cum_pd(V: float, sigV: float, D: float, r: float, years) -> dict:
    """머튼 모형 만기별 위험중립 누적 부도확률 N(−d2(t))."""
    out = {}
    for t in years:
        d2 = (math.log(V / D) + (r - 0.5 * sigV * sigV) * t) / (sigV * math.sqrt(t))
        out[float(t)] = min(max(ncdf(-d2), 0.0), 0.999999)
    return out


def altman(wc, re_, ebit, mve, sales, ta, tl, bve):
    x1, x2, x3, x4, x5 = wc / ta, re_ / ta, ebit / ta, mve / tl, sales / ta
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    z2 = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * (bve / tl)
    return {"Z": z, "Z2": z2, "X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5, "X4b": bve / tl}


def z_zone(z, kind="Z"):
    if z is None or not np.isfinite(z):
        return "-"
    hi, lo = (2.99, 1.81) if kind == "Z" else (2.60, 1.10)
    return "안전" if z > hi else ("회색지대" if z >= lo else "부실 위험")


__all__ = [n for n in dir() if not n.startswith("__")]
