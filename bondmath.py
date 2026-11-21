"""채권 수학 (곡선 기반)
- 국채 만기수익률(액면가 채권 금리) → 할인계수·제로금리 부트스트랩 → 넬슨-시겔-스벤슨(NSS) 적합 → 선도금리
- 곡선 기준 가격, 키 레이트 듀레이션(KRD), 금리 곡선 시나리오(평행·스티프닝·플래트닝·버터플라이)
- 금리 변화 주성분(PCA: 수준·기울기·곡률), 면역화(두 채권으로 부채의 현재가치·듀레이션 맞추기)
- 회사채: 신용스프레드 → 일정 부도강도(회수율 가정) → 부도확률, 위험 반영 가격
금리는 모두 소수(0.04 = 4%), 제로금리는 연속복리.
"""
import math

import numpy as np
import pandas as pd

KEY_T = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]  # 키 레이트 지점 (년)


# ----------------------------------------------------------------------
# 현금흐름
# ----------------------------------------------------------------------
def cashflows(coupon: float, T: float, freq: int = 2, face: float = 100.0):
    """(시점 배열, 현금흐름 배열, 경과이자). 첫 이자까지 남은 기간은 T에서 거꾸로 맞춘다."""
    n = max(int(math.ceil(T * freq - 1e-9)), 1)
    first = T - (n - 1) / freq
    acc = min(max(1.0 - first * freq, 0.0), 1.0)
    t = first + np.arange(n) / freq
    cf = np.full(n, face * coupon / freq)
    cf[-1] += face
    return t, cf, face * coupon / freq * acc


# ----------------------------------------------------------------------
# 곡선
# ----------------------------------------------------------------------
def bootstrap_par(tenors, par, freq: int = 2, t_max: float | None = None):
    """액면가 채권 금리(채권 등가 수익률, 연 freq회) → 반기 격자의 (시점, 할인계수, 연속복리 제로금리).
    격자 사이의 액면 금리는 직선 보간, 1년 미만은 할인채로 본다."""
    tenors = np.asarray(tenors, float)
    par = np.asarray(par, float)
    o = np.argsort(tenors)
    tenors, par = tenors[o], par[o]
    t_max = float(t_max or tenors.max())
    step = 1.0 / freq
    grid = np.arange(step, t_max + 1e-9, step)
    y = np.interp(grid, tenors, par)
    df = np.zeros(len(grid))
    for i, (t, c) in enumerate(zip(grid, y)):
        if t < 1.0 - 1e-9:
            df[i] = (1 + c / freq) ** (-freq * t)
        else:
            df[i] = (1 - c / freq * df[:i].sum()) / (1 + c / freq)
    df = np.clip(df, 1e-8, None)
    # 1년 미만 짧은 지점(1·3개월)도 넣어 곡선 앞쪽을 잡는다
    short = tenors[tenors < step - 1e-9]
    if len(short):
        sdf = (1 + par[: len(short)] / freq) ** (-freq * short)
        grid = np.concatenate([short, grid])
        df = np.concatenate([sdf, df])
    z = -np.log(df) / grid
    return grid, df, z


def _nss_basis(t, l1, l2):
    t = np.maximum(np.asarray(t, float), 1e-6)
    x1, x2 = t / l1, t / l2
    f1 = (1 - np.exp(-x1)) / x1
    f2 = f1 - np.exp(-x1)
    f3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return np.vstack([np.ones_like(t), f1, f2, f3]).T


def nss(t, p):
    b0, b1, b2, b3, l1, l2 = p
    return _nss_basis(t, l1, l2) @ np.array([b0, b1, b2, b3])


def nss_fwd(t, p):
    b0, b1, b2, b3, l1, l2 = p
    t = np.maximum(np.asarray(t, float), 1e-6)
    return b0 + b1 * np.exp(-t / l1) + b2 * (t / l1) * np.exp(-t / l1) + b3 * (t / l2) * np.exp(-t / l2)


def fit_nss(t, z):
    """람다 두 개는 격자 탐색, 베타 네 개는 선형 최소제곱 (값이 튀지 않고 빠름). 반환: (p, rmse)."""
    t = np.asarray(t, float)
    z = np.asarray(z, float)
    best = None
    for l1 in np.geomspace(0.3, 8.0, 18):
        for l2 in np.geomspace(1.0, 30.0, 18):
            if l2 <= l1 * 1.2:
                continue
            X = _nss_basis(t, l1, l2)
            b, *_ = np.linalg.lstsq(X, z, rcond=None)
            e = float(np.sqrt(np.mean((X @ b - z) ** 2)))
            if best is None or e < best[1]:
                best = ((*b, l1, l2), e)
    return best


class Curve:
    """제로금리 곡선. 가격 계산은 부트스트랩 지점 직선 보간(액면가 채권을 정확히 재현), 그림·선도금리는 NSS."""

    def __init__(self, tenors, par, label: str = "", freq: int = 2):
        self.tenors = np.asarray(tenors, float)
        self.par = np.asarray(par, float)
        self.label = label
        self.t, self.df, self.z = bootstrap_par(self.tenors, self.par, freq)
        self.p, self.rmse = fit_nss(self.t, self.z)

    def zero(self, t, dz=None):
        t = np.asarray(t, float)
        z = np.interp(t, self.t, self.z, left=self.z[0], right=self.z[-1])
        if dz is not None:
            z = z + dz(t)
        return z

    def disc(self, t, dz=None, spread=0.0):
        t = np.asarray(t, float)
        return np.exp(-(self.zero(t, dz) + spread) * t)

    def fwd(self, t):
        return nss_fwd(t, self.p)

    def smooth_zero(self, t):
        return nss(t, self.p)

    def par_yield(self, T, freq: int = 2):
        tt = np.arange(1, int(round(T * freq)) + 1) / freq
        d = self.disc(tt)
        return float(freq * (1 - d[-1]) / d.sum())


def flat_curve(rate: float, label="평탄 곡선"):
    return Curve([0.25, 1, 2, 5, 10, 30], [rate] * 6, label)


# ----------------------------------------------------------------------
# 가격·위험
# ----------------------------------------------------------------------
def price(curve: Curve, t, cf, spread=0.0, dz=None) -> float:
    return float((np.asarray(cf) * curve.disc(t, dz, spread)).sum())


def z_spread(curve: Curve, t, cf, dirty: float) -> float:
    lo, hi = -0.05, 0.5
    for _ in range(100):
        m = (lo + hi) / 2
        if price(curve, t, cf, m) > dirty:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def _bump(k_i: int, bp: float = 1e-4):
    """키 지점 KEY_T[k_i]에서 1, 이웃 지점에서 0인 삼각형 충격."""
    pts = KEY_T
    lo = pts[k_i - 1] if k_i > 0 else 0.0
    hi = pts[k_i + 1] if k_i < len(pts) - 1 else 100.0
    c = pts[k_i]

    def f(t):
        t = np.asarray(t, float)
        w = np.where(t <= c, (t - lo) / (c - lo) if c > lo else 1.0, (hi - t) / (hi - c))
        if k_i == 0:
            w = np.where(t <= c, 1.0, w)
        if k_i == len(pts) - 1:
            w = np.where(t >= c, 1.0, w)
        return bp * np.clip(w, 0.0, 1.0)

    return f


def risk(curve: Curve, t, cf, spread=0.0) -> dict:
    """곡선 기준 가격, 유효 듀레이션·볼록성(평행 ±1bp), DV01, 키 레이트 듀레이션."""
    P = price(curve, t, cf, spread)
    up = price(curve, t, cf, spread + 1e-4)
    dn = price(curve, t, cf, spread - 1e-4)
    dur = (dn - up) / (2 * P * 1e-4)
    conv = (dn + up - 2 * P) / (P * 1e-8)
    krd = []
    for i in range(len(KEY_T)):
        b = _bump(i)
        pu = price(curve, t, cf, spread, dz=b)
        pd_ = price(curve, t, cf, spread, dz=lambda x, b=b: -b(x))
        krd.append((pd_ - pu) / (2 * P * 1e-4))
    fw = float((np.asarray(t) * np.asarray(cf) * curve.disc(t, spread=spread)).sum() / P)
    return {"price": P, "dur": dur, "conv": conv, "dv01": P * dur * 1e-4, "krd": np.array(krd), "fw_dur": fw}


SCENARIOS = {
    "평행 +100bp": lambda t: np.full_like(np.asarray(t, float), 0.01),
    "평행 −100bp": lambda t: np.full_like(np.asarray(t, float), -0.01),
    "스티프닝 (단기 −50 · 장기 +50bp)": lambda t: np.interp(t, [2, 10], [-0.005, 0.005]),
    "플래트닝 (단기 +50 · 장기 −50bp)": lambda t: np.interp(t, [2, 10], [0.005, -0.005]),
    "버터플라이 (중기 +50 · 양끝 −25bp)": lambda t: np.interp(t, [1, 5, 20], [-0.0025, 0.005, -0.0025]),
    "단기만 +100bp (2년 이하)": lambda t: np.interp(t, [2, 5], [0.01, 0.0]),
}


# ----------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------
def curve_pca(ts: pd.DataFrame, years: int = 5, n: int = 3):
    """일별 금리 변화(bp)의 공분산 → 고유벡터 (수준·기울기·곡률). 반환: (loadings DataFrame, 설명 비율, 일별 표준편차 bp)."""
    d = ts[ts.index >= ts.index.max() - pd.Timedelta(days=365 * years)].ffill().diff().dropna() * 1e4
    d = d.dropna(axis=1, how="any")
    if len(d) < 60 or d.shape[1] < 3:
        return None
    C = np.cov(d.values.T)
    w, v = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    w, v = w[o], v[:, o]
    for i in range(min(n, v.shape[1])):  # 부호 맞춤: 1번(수준)은 양수, 2번(기울기)은 장기 쪽이 양수
        if i == 0 and v[:, i].sum() < 0:
            v[:, i] *= -1
        if i == 1 and v[-1, i] < v[0, i]:
            v[:, i] *= -1
        if i == 2 and v[len(v) // 2, i] < 0:
            v[:, i] *= -1
    L = pd.DataFrame(v[:, :n], index=d.columns, columns=["수준", "기울기", "곡률"][:n])
    return L, w[:n] / w.sum(), np.sqrt(w[:n])


# ----------------------------------------------------------------------
# 면역화
# ----------------------------------------------------------------------
def immunize(curve: Curve, L: float, H: float, bond_a, bond_b):
    """부채 L(H년 뒤 한 번 지급)을 두 채권(bond = (t, cf), 액면 100 기준)으로 현재가치·듀레이션 맞춤.
    반환: dict(n_a, n_b 액면 100 단위 수, pv_l, …) 또는 None."""
    pv_l = L * float(curve.disc(H))
    ra = risk(curve, *bond_a)
    rb = risk(curve, *bond_b)
    Da, Db = ra["fw_dur"], rb["fw_dur"]
    if abs(Da - Db) < 1e-6 or not (min(Da, Db) <= H <= max(Da, Db)):
        return None
    w = (H - Db) / (Da - Db)
    n_a = pv_l * w / ra["price"]
    n_b = pv_l * (1 - w) / rb["price"]
    rows = []
    for nm, sh in [("−200bp", -0.02), ("−100bp", -0.01), ("변화 없음", 0.0), ("+100bp", 0.01), ("+200bp", 0.02)]:
        dz = (lambda t, s=sh: np.full_like(np.asarray(t, float), s))
        a = n_a * price(curve, *bond_a, dz=dz) + n_b * price(curve, *bond_b, dz=dz)
        l = L * float(curve.disc(H, dz=dz))
        rows.append({"평행 이동": nm, "채권 가치": a, "부채 현재가치": l, "잉여": a - l})
    conv_a = (n_a * ra["price"] * ra["conv"] + n_b * rb["price"] * rb["conv"]) / pv_l
    return {"n_a": n_a, "n_b": n_b, "w": w, "pv_l": pv_l, "Da": Da, "Db": Db, "conv_assets": conv_a, "conv_liab": H * H,
            "table": pd.DataFrame(rows)}


# ----------------------------------------------------------------------
# 회사채 신용
# ----------------------------------------------------------------------
def hazard_from_spread(spread: float, recovery: float) -> float:
    """신용 삼각형: 스프레드 ≈ 부도강도 × (1 − 회수율)."""
    return max(spread, 0.0) / max(1 - recovery, 1e-6)


def risky_price(curve: Curve, t, cf, lam: float, recovery: float, face: float = 100.0) -> float:
    """생존 시 현금흐름 + 부도 시 액면 × 회수율 (각 이자 기간 중간에 부도 난다고 가정)."""
    t = np.asarray(t, float)
    Q = np.exp(-lam * t)
    t_prev = np.concatenate([[0.0], t[:-1]])
    Qp = np.exp(-lam * t_prev)
    surv = float((np.asarray(cf) * curve.disc(t) * Q).sum())
    dflt = float((recovery * face * curve.disc((t + t_prev) / 2) * (Qp - Q)).sum())
    return surv + dflt


__all__ = [n for n in dir() if not n.startswith("__")]
