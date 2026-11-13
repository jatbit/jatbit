"""변동성 모형 보정 — SVI(만기별 스마일), SABR(만기별), 헤스턴(곡면 전체)
시장 옵션의 내재변동성에 모형 파라미터를 최소제곱으로 맞춘다 (scipy.optimize.least_squares).
"""
import math

import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:  # noqa: BLE001
    least_squares = None

import bsm


# ----------------------------------------------------------------------
# SVI (raw): 총분산 w(k) = a + b(ρ(k−m) + √((k−m)² + σ²)),  k = ln(K/F)
# ----------------------------------------------------------------------
def svi_w(k, a, b, rho, m, s):
    k = np.asarray(k, float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s * s))


def fit_svi(k, iv, T, weights=None) -> dict:
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    ok = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[ok], iv[ok]
    if len(k) < 5 or least_squares is None:
        return {"ok": False}
    w = iv * iv * T
    wt = np.ones_like(w) if weights is None else np.asarray(weights, float)[ok]
    atm = float(np.interp(0.0, np.sort(k), w[np.argsort(k)]))
    x0 = [max(atm * 0.8, 1e-6), 0.1, -0.3, 0.0, 0.1]
    lo = [-1.0, 1e-6, -0.999, -2.0, 1e-4]
    hi = [max(w.max() * 2, 1e-3), 5.0, 0.999, 2.0, 3.0]

    def res(p):
        return (svi_w(k, *p) - w) * wt / max(T, 1e-4)

    try:
        r = least_squares(res, x0, bounds=(lo, hi), max_nfev=4000)
    except Exception:  # noqa: BLE001
        return {"ok": False}
    a, b, rho, m, s = r.x
    fit_iv = np.sqrt(np.maximum(svi_w(k, *r.x), 1e-12) / T)
    rmse = float(np.sqrt(np.mean((fit_iv - iv) ** 2)))
    # 나비 스프레드 차익(밀도 음수) 점검: g(k) ≥ 0
    kk = np.linspace(k.min() - 0.1, k.max() + 0.1, 400)
    ww = svi_w(kk, *r.x)
    dw = b * (rho + (kk - m) / np.sqrt((kk - m) ** 2 + s * s))
    d2w = b * s * s / ((kk - m) ** 2 + s * s) ** 1.5
    g = (1 - kk * dw / (2 * ww)) ** 2 - dw * dw / 4 * (1 / ww + 0.25) + d2w / 2
    return {"ok": True, "a": a, "b": b, "rho": rho, "m": m, "s": s, "rmse": rmse, "arb_free": bool((g >= -1e-8).all() and (ww > 0).all())}


# ----------------------------------------------------------------------
# SABR (Hagan 2002 로그정규 근사)
# ----------------------------------------------------------------------
def sabr_vol(F, K, T, alpha, beta, rho, nu):
    F = float(F)
    K = np.asarray(K, float)
    with np.errstate(all="ignore"):
        fk = F * K
        lfk = np.log(F / K)
        ob = 1 - beta
        fkb = fk ** (ob / 2)
        z = nu / alpha * fkb * lfk
        xz = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
        zx = np.where(np.abs(z) < 1e-7, 1.0, z / np.where(xz == 0, 1.0, xz))
        den = fkb * (1 + ob ** 2 / 24 * lfk ** 2 + ob ** 4 / 1920 * lfk ** 4)
        corr = 1 + (ob ** 2 / 24 * alpha * alpha / fk ** ob + rho * beta * nu * alpha / (4 * fkb) + (2 - 3 * rho * rho) / 24 * nu * nu) * T
        out = alpha / den * zx * corr
    return np.where(K > 0, out, np.nan)


def fit_sabr(F, K, iv, T, beta=0.5) -> dict:
    K, iv = np.asarray(K, float), np.asarray(iv, float)
    ok = np.isfinite(K) & np.isfinite(iv) & (iv > 0)
    K, iv = K[ok], iv[ok]
    if len(K) < 4 or least_squares is None:
        return {"ok": False}
    atm = float(np.interp(F, np.sort(K), iv[np.argsort(K)]))
    x0 = [atm * F ** (1 - beta), -0.3, 0.8]

    def res(p):
        v = sabr_vol(F, K, T, p[0], beta, p[1], p[2])
        return np.where(np.isfinite(v), v - iv, 1.0)

    try:
        r = least_squares(res, x0, bounds=([1e-6, -0.999, 1e-4], [10 * F ** (1 - beta) + 10, 0.999, 10.0]), max_nfev=3000)
    except Exception:  # noqa: BLE001
        return {"ok": False}
    v = sabr_vol(F, K, T, r.x[0], beta, r.x[1], r.x[2])
    return {"ok": True, "alpha": r.x[0], "beta": beta, "rho": r.x[1], "nu": r.x[2], "rmse": float(np.sqrt(np.nanmean((v - iv) ** 2)))}


# ----------------------------------------------------------------------
# 헤스턴: 확률 변동성 dv = κ(θ − v)dt + σv√v dW₂, corr(dW₁, dW₂) = ρ
# ----------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)


def _heston_cf(u, T, r, q, S, v0, kappa, theta, sigma, rho):
    """ln S_T의 특성함수 (Albrecher 'little trap' 형태). u: 복소 배열."""
    iu = 1j * u
    d = np.sqrt((rho * sigma * iu - kappa) ** 2 + sigma * sigma * (iu + u * u))
    g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)
    e = np.exp(-d * T)
    C = (kappa * theta / sigma ** 2) * ((kappa - rho * sigma * iu - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (kappa - rho * sigma * iu - d) / sigma ** 2 * (1 - e) / (1 - g * e)
    return np.exp(iu * (math.log(S) + (r - q) * T) + C + D * v0)


def heston_call(S, K, T, r, q, v0, kappa, theta, sigma, rho, umax: float = 200.0):
    """헤스턴 콜 가격 (행사가 배열 가능). 가우스-르장드르 적분."""
    K = np.atleast_1d(np.asarray(K, float))
    u = 0.5 * umax * (_GL_X + 1) + 1e-8
    w = 0.5 * umax * _GL_W
    phi2 = _heston_cf(u, T, r, q, S, v0, kappa, theta, sigma, rho)
    phi1 = _heston_cf(u - 1j, T, r, q, S, v0, kappa, theta, sigma, rho) / (S * math.exp((r - q) * T))
    lk = np.log(K)[:, None]
    e = np.exp(-1j * u[None, :] * lk) / (1j * u[None, :])
    P1 = 0.5 + (np.real(e * phi1[None, :]) @ w) / math.pi
    P2 = 0.5 + (np.real(e * phi2[None, :]) @ w) / math.pi
    return S * math.exp(-q * T) * P1 - K * math.exp(-r * T) * P2


def heston_iv(S, K, T, r, q, p) -> np.ndarray:
    c = heston_call(S, K, T, r, q, *p)
    return bsm.implied_vol(c, S, K, T, r, q, True)


def fit_heston(S, r, q, data: list) -> dict:
    """data: [(T, K 배열, 시장 IV 배열)]. 모든 만기를 한 번에 맞춘다(오차 = 가격 차이 ÷ 베가 ≈ IV 차이)."""
    if least_squares is None:
        return {"ok": False, "msg": "scipy가 없습니다"}
    rows = []
    for T, K, iv in data:
        K, iv = np.asarray(K, float), np.asarray(iv, float)
        ok = np.isfinite(K) & np.isfinite(iv) & (iv > 0)
        if ok.sum() >= 3:
            mkt = bsm.price(S, K[ok], T, r, q, iv[ok], True)
            vega = np.maximum(np.asarray(bsm.greeks(S, K[ok], T, r, q, iv[ok], True)["vega"]) * 100, 1e-3 * S)
            rows.append((T, K[ok], iv[ok], mkt, vega))
    if not rows:
        return {"ok": False, "msg": "보정할 옵션이 부족합니다"}
    atm_near = float(np.interp(S, np.sort(rows[0][1]), rows[0][2][np.argsort(rows[0][1])]))
    atm_far = float(np.interp(S, np.sort(rows[-1][1]), rows[-1][2][np.argsort(rows[-1][1])]))
    x0 = [atm_near ** 2, 2.0, atm_far ** 2, 0.6, -0.6]
    lo = [1e-4, 0.05, 1e-4, 0.01, -0.99]
    hi = [2.0, 15.0, 2.0, 3.0, 0.99]

    def res(p):
        v0, kappa, theta, sigma, rho = p
        out = []
        for T, K, iv, mkt, vega in rows:
            c = heston_call(S, K, T, r, q, v0, kappa, theta, sigma, rho)
            out.append((c - mkt) / vega)
        return np.concatenate(out)

    try:
        r_ = least_squares(res, x0, bounds=(lo, hi), max_nfev=300, x_scale=[0.05, 1.0, 0.05, 0.3, 0.3])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "msg": str(e)}
    v0, kappa, theta, sigma, rho = r_.x
    errs, fits = [], []
    for T, K, iv, _m, _v in rows:
        fv = heston_iv(S, K, T, r, q, r_.x)
        fits.append((T, K, iv, fv))
        errs.append(fv - iv)
    e = np.concatenate(errs)
    return {"ok": True, "v0": v0, "kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho,
            "rmse": float(np.sqrt(np.nanmean(e ** 2))), "feller": 2 * kappa * theta > sigma * sigma, "fits": fits, "n": int(len(e))}
