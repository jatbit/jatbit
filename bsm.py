"""블랙-숄즈-머튼(연속 배당수익률 q) 가격·그릭스·내재변동성. numpy만 사용 (scipy 불필요)."""
import math

import numpy as np

_erf = np.vectorize(math.erf, otypes=[float])
SQRT2 = math.sqrt(2.0)
INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def ncdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / SQRT2))


def npdf(x):
    x = np.asarray(x, dtype=float)
    return INV_SQRT2PI * np.exp(-0.5 * x * x)


def _prep(S, K, T, r, q, sig, is_call):
    S, K, T, sig = (np.asarray(v, dtype=float) for v in (S, K, T, sig))
    is_call = np.asarray(is_call, dtype=bool)
    S, K, T, sig, is_call = np.broadcast_arrays(S, K, T, sig, is_call)
    T = np.maximum(T, 1e-6)
    sig = np.maximum(sig, 1e-6)
    return S, K, T, float(r), float(q), sig, is_call


def price(S, K, T, r, q, sig, is_call):
    S, K, T, r, q, sig, c = _prep(S, K, T, r, q, sig, is_call)
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sq)
    d2 = d1 - sig * sq
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    call = S * dq * ncdf(d1) - K * dr * ncdf(d2)
    put = K * dr * ncdf(-d2) - S * dq * ncdf(-d1)
    return np.where(c, call, put)


def greeks(S, K, T, r, q, sig, is_call) -> dict:
    """델타(주당), 감마(주당/$1), 베가($/변동성 1%p), 세타($/1일, 달력일), 로($/금리 1%p)."""
    S, K, T, r, q, sig, c = _prep(S, K, T, r, q, sig, is_call)
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sq)
    d2 = d1 - sig * sq
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    n1 = npdf(d1)
    N1, N2, Nm1, Nm2 = ncdf(d1), ncdf(d2), ncdf(-d1), ncdf(-d2)
    delta = np.where(c, dq * N1, dq * (N1 - 1.0))
    gamma = dq * n1 / (S * sig * sq)
    vega = S * dq * n1 * sq / 100.0
    common = -S * dq * n1 * sig / (2.0 * sq)
    theta = np.where(c, common - r * K * dr * N2 + q * S * dq * N1,
                     common + r * K * dr * Nm2 - q * S * dq * Nm1) / 365.0
    rho = np.where(c, K * T * dr * N2, -K * T * dr * Nm2) / 100.0
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(mkt, S, K, T, r, q, is_call, lo=1e-4, hi=5.0, iters=100):
    """이분법으로 시장가에 맞는 변동성을 찾는다. 무차익 범위를 벗어나거나 가격이 없으면 NaN."""
    mkt = np.asarray(mkt, dtype=float)
    S, K, T, r, q, _, c = _prep(S, K, T, r, q, np.ones_like(mkt), is_call)
    mkt = np.broadcast_to(mkt, S.shape).astype(float)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    lower = np.where(c, np.maximum(S * dq - K * dr, 0.0), np.maximum(K * dr - S * dq, 0.0))
    upper = np.where(c, S * dq, K * dr)
    ok = np.isfinite(mkt) & (mkt > 0) & (mkt > lower + 1e-10) & (mkt < upper)
    a = np.full(S.shape, lo)
    b = np.full(S.shape, hi)
    ok &= price(S, K, T, r, q, b, c) >= mkt  # 상한 변동성으로도 못 맞추면 제외
    for _ in range(iters):
        m = 0.5 * (a + b)
        pm = price(S, K, T, r, q, m, c)
        go_up = pm < mkt
        a = np.where(go_up, m, a)
        b = np.where(go_up, b, m)
    out = 0.5 * (a + b)
    return np.where(ok, out, np.nan)


def hist_vol(close, window: int, periods: int = 252) -> float:
    """일별 로그수익률 표준편차 × √252 (최근 window일)."""
    import pandas as pd

    s = pd.Series(close).dropna()
    lr = np.log(s / s.shift(1)).dropna()
    if len(lr) < max(5, window // 2):
        return float("nan")
    return float(lr.iloc[-window:].std(ddof=1) * math.sqrt(periods))


def solve_hedge_two(port_delta, port_gamma, d1, g1, d2, g2):
    """수단 1·2의 수량 n1, n2 (단위당 델타 d, 감마 g)로 포트폴리오 델타·감마를 0으로. 해가 없으면 None."""
    det = d1 * g2 - d2 * g1
    if not np.isfinite(det) or abs(det) < 1e-12:
        return None
    n1 = (-port_delta * g2 + port_gamma * d2) / det
    n2 = (-port_gamma * d1 + port_delta * g1) / det
    return float(n1), float(n2)


# ----------------------------------------------------------------------
# 미국식(조기행사 가능) 옵션: CRR 이항트리 (연속 배당수익률 q)
# ----------------------------------------------------------------------
def american_tree(S, K, T, r, q, sig, is_call, steps: int = 300, boundary: bool = False) -> dict:
    """미국식 옵션 가격과 트리에서 바로 읽는 델타·감마·세타. boundary=True면 시점별 조기행사 경계 주가도 돌려준다."""
    S, K, T, sig = float(S), float(K), max(float(T), 1e-6), max(float(sig), 1e-6)
    n = int(max(steps, 3))
    dt = T / n
    u = math.exp(sig * math.sqrt(dt))
    d = 1.0 / u
    a = math.exp((r - q) * dt)
    p = (a - d) / (u - d)
    p = min(max(p, 0.0), 1.0)
    disc = math.exp(-r * dt)
    sgn = 1.0 if is_call else -1.0
    j = np.arange(n + 1)
    ST = S * u ** (2 * j - n)
    V = np.maximum(sgn * (ST - K), 0.0)
    VE = V.copy()  # 같은 트리의 유럽식 값 (트리 오차 보정용)
    keep, bnd = {}, []
    for i in range(n - 1, -1, -1):
        Si = S * u ** (2 * np.arange(i + 1) - i)
        VE = disc * (p * VE[1:i + 2] + (1 - p) * VE[0:i + 1])
        cont = disc * (p * V[1:i + 2] + (1 - p) * V[0:i + 1])
        ex = np.maximum(sgn * (Si - K), 0.0)
        early = (ex > cont + 1e-12) & (ex > 0)
        V = np.where(early, ex, cont)
        if boundary:
            if early.any():
                bnd.append((i * dt, float(Si[early].max() if not is_call else Si[early].min())))
            else:
                bnd.append((i * dt, float("nan")))
        if i <= 2:
            keep[i] = (Si.copy(), V.copy())
    S1, V1 = keep[1]
    S2, V2 = keep[2]
    delta = (V1[1] - V1[0]) / (S1[1] - S1[0])
    d_up = (V2[2] - V2[1]) / (S2[2] - S2[1])
    d_dn = (V2[1] - V2[0]) / (S2[1] - S2[0])
    gamma = (d_up - d_dn) / (0.5 * (S2[2] - S2[0]))
    theta = (V2[1] - V[0]) / (2 * dt) / 365.0
    eu_exact = float(price(S, K, T, r, q, sig, is_call))
    out = {"price": float(V[0]), "price_cv": float(V[0] + eu_exact - VE[0]), "delta": float(delta), "gamma": float(gamma),
           "theta": float(theta)}
    if boundary:
        out["boundary"] = sorted(bnd)
    return out


def american_price(S, K, T, r, q, sig, is_call, steps: int = 300) -> float:
    """트리 가격에 '정확한 유럽식 − 트리 유럽식'을 더해 트리 오차를 줄인 값 (제어변량)."""
    t = american_tree(S, K, T, r, q, sig, is_call, steps)
    return max(t["price_cv"], max((S - K) if is_call else (K - S), 0.0))


def american_greeks(S, K, T, r, q, sig, is_call, steps: int = 300) -> dict:
    """트리 델타·감마·세타 + 변동성·금리를 조금씩 바꿔 다시 계산한 베가(변동성 1%p당)·로(금리 1%p당)."""
    base = american_tree(S, K, T, r, q, sig, is_call, steps)
    base["price"] = american_price(S, K, T, r, q, sig, is_call, steps)
    hv = 0.01
    vega = (american_price(S, K, T, r, q, sig + hv, is_call, steps) - american_price(S, K, T, r, q, max(sig - hv, 1e-4), is_call, steps)) / 2
    rho = (american_price(S, K, T, r + 0.01, q, sig, is_call, steps) - american_price(S, K, T, r - 0.01, q, sig, is_call, steps)) / 2
    base.update({"vega": vega, "rho": rho})
    return base


def american_iv(mkt, S, K, T, r, q, is_call, steps: int = 150, lo=1e-3, hi=4.0, iters=45) -> float:
    """미국식 트리 가격이 시장가와 같아지는 변동성(이분법). 내재가치보다 싸거나 상한을 넘으면 NaN."""
    try:
        mkt = float(mkt)
    except Exception:  # noqa: BLE001
        return float("nan")
    if not np.isfinite(mkt) or mkt <= 0:
        return float("nan")
    intrinsic = max((S - K) if is_call else (K - S), 0.0)
    if mkt < intrinsic - 1e-9:
        return float("nan")
    if american_price(S, K, T, r, q, hi, is_call, steps) < mkt:
        return float("nan")
    if american_price(S, K, T, r, q, lo, is_call, steps) > mkt:
        return float("nan")
    a, b = lo, hi
    for _ in range(iters):
        m = 0.5 * (a + b)
        if american_price(S, K, T, r, q, m, is_call, steps) < mkt:
            a = m
        else:
            b = m
        if b - a < 1e-5:
            break
    return 0.5 * (a + b)
