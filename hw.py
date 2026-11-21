"""헐-화이트 1요인 삼항 트리 (Hull 교재 방식): dr = (θ(t) − a·r)dt + σ dW
오늘의 제로 곡선에 정확히 맞춘 뒤 콜옵션부 채권(발행사가 정해진 날 정해진 가격에 되살 수 있는 채권)을 계산하고,
시장가와 맞는 옵션 조정 스프레드(OAS)를 구한다.
"""
import math

import numpy as np


class HWTree:
    def __init__(self, curve, T: float, a: float = 0.05, sigma: float = 0.01, steps_per_year: int = 24):
        self.a, self.sigma = max(a, 1e-4), max(sigma, 1e-5)
        self.N = max(int(math.ceil(T * steps_per_year)), 1)
        self.dt = T / self.N
        dt = self.dt
        self.dx = self.sigma * math.sqrt(3 * dt)
        M = math.exp(-self.a * dt) - 1
        self.jmax = max(int(math.ceil(0.184 / (self.a * dt))), 1)
        J = np.arange(-self.jmax, self.jmax + 1)
        self.J = J
        # 가지 확률과 다음 칸 (보통 / 맨 위는 아래로 꺾임 / 맨 아래는 위로 꺾임)
        jm = J * M
        pu = 1 / 6 + (jm * jm + jm) / 2
        pm = 2 / 3 - jm * jm
        pd = 1 / 6 + (jm * jm - jm) / 2
        mid = J.copy()
        top, bot = J == self.jmax, J == -self.jmax
        pu[top] = 7 / 6 + (jm[top] ** 2 + 3 * jm[top]) / 2
        pm[top] = -1 / 3 - jm[top] ** 2 - 2 * jm[top]
        pd[top] = 1 / 6 + (jm[top] ** 2 + jm[top]) / 2
        mid[top] = J[top] - 1
        pu[bot] = 1 / 6 + (jm[bot] ** 2 - jm[bot]) / 2
        pm[bot] = -1 / 3 - jm[bot] ** 2 + 2 * jm[bot]
        pd[bot] = 7 / 6 + (jm[bot] ** 2 - 3 * jm[bot]) / 2
        mid[bot] = J[bot] + 1
        self.p = np.vstack([pu, pm, pd])            # (3, nodes)
        self.k = np.vstack([mid + 1, mid, mid - 1]) + self.jmax  # 다음 칸 인덱스
        self.fit(curve)

    def fit(self, curve):
        """각 단계의 α를 오늘 할인계수에 맞춘다 (애로-드브뢰 가격 전진)."""
        n, dt, dx = len(self.J), self.dt, self.dx
        times = np.arange(1, self.N + 1) * dt
        P = curve.disc(times)
        self.alpha = np.zeros(self.N)
        Q = np.zeros(n)
        Q[self.jmax] = 1.0
        self.width = []
        for m in range(self.N):
            jm_ = min(m, self.jmax)
            idx = np.arange(self.jmax - jm_, self.jmax + jm_ + 1)
            s = float((Q[idx] * np.exp(-self.J[idx] * dx * dt)).sum())
            self.alpha[m] = (math.log(s) - math.log(P[m])) / dt
            if m == self.N - 1:
                break
            disc = Q[idx] * np.exp(-(self.alpha[m] + self.J[idx] * dx) * dt)
            Qn = np.zeros(n)
            for b in range(3):
                np.add.at(Qn, self.k[b, idx], disc * self.p[b, idx])
            Q = Qn
        return self

    def rate(self, m):
        return self.alpha[m] + self.J * self.dx

    def value(self, cf_steps: dict, call_steps: dict | None = None, put_steps: dict | None = None, spread: float = 0.0) -> float:
        """cf_steps: {단계: 현금흐름}, call_steps: {단계: 콜 가격(이자 지급 직후 기준)}. 반환: 오늘 가치(경과이자 포함 가격)."""
        call_steps = call_steps or {}
        put_steps = put_steps or {}
        V = np.zeros(len(self.J))
        for m in range(self.N, 0, -1):
            if m in call_steps:
                V = np.minimum(V, call_steps[m])
            if m in put_steps:
                V = np.maximum(V, put_steps[m])
            V = V + cf_steps.get(m, 0.0)
            jm_ = min(m - 1, self.jmax)
            idx = np.arange(self.jmax - jm_, self.jmax + jm_ + 1)
            cont = (self.p[:, idx] * V[self.k[:, idx]]).sum(axis=0)
            Vn = np.zeros(len(self.J))
            Vn[idx] = cont * np.exp(-(self.rate(m - 1)[idx] + spread) * self.dt)
            V = Vn
        return float(V[self.jmax])


def _steps(tree: HWTree, times):
    return [max(1, min(tree.N, int(round(t / tree.dt)))) for t in times]


def callable_bond(curve, coupon: float, T: float, freq: int, call_price: float, first_call: float,
                  a: float = 0.05, sigma: float = 0.01, spread: float = 0.0, face: float = 100.0, kind: str = "call",
                  tree: HWTree | None = None) -> dict:
    """반환: straight(옵션 없는 가치), callable(옵션부 가치), option(차이) — 모두 경과이자 포함, 액면 100 기준."""
    from bondmath import cashflows

    t, cf, acc = cashflows(coupon, T, freq, face)
    tree = tree or HWTree(curve, T, a, sigma, steps_per_year=max(24, freq * 6))
    st_ = _steps(tree, t)
    cfs = {}
    for s, c in zip(st_, cf):
        cfs[s] = cfs.get(s, 0.0) + c
    ex = {s: call_price for s, tt in zip(st_[:-1], t[:-1]) if tt >= first_call - 1e-9}
    straight = tree.value(cfs, spread=spread)
    opt = tree.value(cfs, call_steps=ex if kind == "call" else None, put_steps=ex if kind == "put" else None, spread=spread)
    return {"straight": straight, "callable": opt, "option": straight - opt, "accrued": acc, "tree": tree, "n_ex": len(ex)}


def oas(curve, dirty_mkt: float, coupon, T, freq, call_price, first_call, a=0.05, sigma=0.01, kind="call") -> float:
    from bondmath import cashflows

    t, cf, _ = cashflows(coupon, T, freq)
    tree = HWTree(curve, T, a, sigma, steps_per_year=max(24, freq * 6))
    lo, hi = -0.05, 0.30
    f = lambda s: callable_bond(curve, coupon, T, freq, call_price, first_call, a, sigma, s, kind=kind, tree=tree)["callable"] - dirty_mkt  # noqa: E731
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return float("nan")
    for _ in range(60):
        m = (lo + hi) / 2
        fm = f(m)
        if fm > 0:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def eff_risk(curve, coupon, T, freq, call_price, first_call, a, sigma, spread, kind="call", bp=0.0025):
    """곡선 평행 ±25bp 이동 후 트리를 다시 맞춰 유효 듀레이션·볼록성 (OAS는 고정)."""
    import copy

    def shifted(s):
        c = copy.copy(curve)
        c.z = curve.z + s
        return c

    v0 = callable_bond(curve, coupon, T, freq, call_price, first_call, a, sigma, spread, kind=kind)
    vu = callable_bond(shifted(bp), coupon, T, freq, call_price, first_call, a, sigma, spread, kind=kind)
    vd = callable_bond(shifted(-bp), coupon, T, freq, call_price, first_call, a, sigma, spread, kind=kind)
    out = {}
    for k in ("straight", "callable"):
        P0, Pu, Pd = v0[k], vu[k], vd[k]
        out[k] = {"dur": (Pd - Pu) / (2 * P0 * bp), "conv": (Pd + Pu - 2 * P0) / (P0 * bp * bp)}
    return out


__all__ = [n for n in dir() if not n.startswith("__")]
