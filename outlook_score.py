"""매수·매도 점수(0~100): 과거 사례 기반 점수, 미래 추정치 점수, 게이지 그림, 스크리너용 여러 종목 점수."""
import numpy as np
import pandas as pd
import streamlit as st

from research_data import RATIOS, _split_factor, annual_eps_table, compute_ratios

from outlook_signals import *  # noqa: F401,F403

# ----------------------------------------------------------------------
# 매수·매도 점수 (0~100)
# ----------------------------------------------------------------------
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


def _gauge_svg(title, d, reliable, sub="", px=132):
    """0~100 원형 점수. 호는 언제나 0에서 점수까지 채운다(100점 = 한 바퀴).
    80% 구간이 넓으면 바깥 얇은 띠로 범위를 함께 그린다."""
    import html as _h

    r, sw, C = 44, 10, 2 * np.pi * 44
    lo, hi, mid = (max(0.0, min(100.0, d[k])) for k in ("lo", "hi", "mid"))
    if reliable:
        col = "#1B9E5A" if mid >= 55 else ("#D9534F" if mid <= 45 else "#8A8F98")
        lab, small = score_label(mid), "점"
    else:
        col = "#1B9E5A" if lo >= 55 else ("#D9534F" if hi <= 45 else "#8A8F98")
        lab, small = ("매수 쪽" if lo >= 55 else ("매도 쪽" if hi <= 45 else "판단 보류")), f"{lo:.0f}–{hi:.0f}"
    arc = ""
    if mid > 0.5:
        arc = (f'<circle cx="60" cy="60" r="{r}" fill="none" stroke="{col}" stroke-width="{sw}" '
               + ('' if mid >= 99.5 else 'stroke-linecap="round" ')
               + f'stroke-dasharray="{mid / 100 * C:.2f} {C:.2f}" transform="rotate(-90 60 60)"/>')
    band = ""
    if not reliable:
        r2 = 55
        C2 = 2 * np.pi * r2
        band = (f'<circle cx="60" cy="60" r="{r2}" fill="none" stroke="{col}" stroke-opacity=".5" stroke-width="3" '
                f'stroke-dasharray="{max(hi - lo, 1.0) / 100 * C2:.2f} {C2:.2f}" stroke-dashoffset="{-lo / 100 * C2:.2f}" '
                f'transform="rotate(-90 60 60)"/>')
    big = px >= 150
    return (f'<div style="flex:0 1 {px + 30}px;text-align:center;min-width:{px}px">'
            f'<svg viewBox="0 0 120 120" width="{px}" height="{px}" role="img" aria-label="{_h.escape(title)} {mid:.0f}점">'
            f'<circle cx="60" cy="60" r="{r}" fill="none" stroke="rgba(128,128,128,.22)" stroke-width="{sw}"/>{band}{arc}'
            f'<text x="60" y="66" text-anchor="middle" font-size="30" font-weight="700" fill="currentColor">{mid:.0f}</text>'
            f'<text x="60" y="84" text-anchor="middle" font-size="{11 if reliable else 10}" fill="currentColor" opacity=".7">{small}</text></svg>'
            f'<div style="font-weight:{700 if big else 600};font-size:{16 if big else 12}px;margin-top:-4px">{_h.escape(title)}</div>'
            f'<div style="font-size:{15 if big else 12}px;color:{col};font-weight:600">{lab}</div>'
            + (f'<div style="font-size:{11 if big else 10}px;opacity:.65;line-height:1.3;margin-top:2px">{_h.escape(sub)}</div>' if sub else "")
            + '</div>')


def total_score(ds):
    """여러 점수(dict lo·mid·hi)를 같은 비중으로 평균 → (dict, 80% 구간이 좁은지)."""
    d = {k: float(np.mean([x[k] for x in ds])) for k in ("lo", "mid", "hi")}
    return d, d["hi"] - d["lo"] <= RELIABLE_WIDTH


def gauges_html(items, total=None):
    """total = (제목, d, reliable, 설명) 이면 크게, items는 작은 원으로."""
    small = "".join(_gauge_svg(*it, px=92 if total else 132) for it in items)
    if not total:
        return ('<div style="display:flex;flex-wrap:wrap;gap:10px;justify-content:space-around;margin:4px 0 2px">' + small + "</div>")
    return ('<div style="display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;justify-content:center;margin:4px 0 2px">'
            + _gauge_svg(*total, px=176)
            + '<div style="display:flex;flex-wrap:wrap;gap:6px;justify-content:space-around;flex:1 1 300px;max-width:460px">'
            + small + "</div></div>")


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


# ----------------------------------------------------------------------
# 스크리너용: 여러 종목 점수를 한꺼번에 (기업 리서치 첫 화면의 기본 설정과 같은 판정)
# ----------------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def screen_score(sym: str, domestic: bool, sector: str = "", industry: str = "", ksec: str = "", dart_key: str | None = None):
    """한 종목의 6개월 점수 3개 (CAPM 미포함 · 업종 평균 대비 · CAPM 대비).
    기업 리서치 첫 화면의 기본값(직전 사업연도 EPS 기준 PER · 분기 TTM · 모멘텀 · 위험 신호 · 추가 신호, 최근 10년)과 같은 규칙이며,
    종목마다 Yahoo 요청을 줄이려고 컨센서스(12M 선행 PER)와 실적 서프라이즈는 뺀다."""
    import yfinance as yf

    import stats
    from research_data import annual_per_series, load_sec_financials

    h, sp = _hist_close(sym)
    if len(h) < 300:
        raise RuntimeError("주가 이력이 짧음")
    yinfo = None
    if domestic:
        t = yf.Ticker(sym)
        try:
            yinfo = t.info or None
        except Exception:  # noqa: BLE001
            yinfo = None
        sec = None
        if dart_key:
            try:
                import kr

                sec = kr.load_dart_financials(sym.split(".")[0], dart_key)
                sec = sec if sec.shape[1] >= 2 else None
            except Exception:  # noqa: BLE001
                sec = None
        if sec is None:
            import kr

            sec = kr.yahoo_frame({"income": t.income_stmt, "balance": t.balance_sheet, "cashflow": t.cashflow})
    else:
        sec = load_sec_financials(sym)
    if sec is None or sec.shape[1] < 2:
        raise RuntimeError("재무제표 없음")
    prof = screen_profile(domestic, sector, industry, ksec, yinfo)
    info = {"sector": (yinfo or {}).get("sector") or (None if domestic else GICS_YH.get(sector, sector))}
    end = pd.Timestamp.now().normalize()
    R = {"hist": h, "splits": sp, "_period": (end - pd.Timedelta(days=3652), end), "info": info}
    daily = {}
    ann = annual_per_series(h, sec, sp)
    if ann is not None:
        daily["PER"] = (ann, "배")
    opts = dict(quarterly=not domestic, momentum=True, surprise=False, peers=False, flags=True, extra=True)
    rf_now = stats.risk_free_for(domestic, pd.Timestamp.now().date())[0]
    c = build_ctx(sec, R, sym, daily, opts, domestic=domestic)
    rows, signals, _sk, flags = evaluate(c, prof, [], rf_now)
    if not rows:
        raise RuntimeError("판정할 수 있는 시점이 없음")
    now_d = min(c["end"], c["h"].index[-1])
    sg = signals(now_d)
    sc, n = _score(sg)
    if not n:
        raise RuntimeError("지금 켜진 신호가 없음")
    pool = None
    if not domestic:
        try:
            pool = [tuple(p) for r in validate_profile(prof) if "pts" in r and r["종목"].upper() != sym.upper() for p in r["pts"]]
        except Exception:  # noqa: BLE001
            pool = None
    fl = _flags_at(flags, c, now_d) if flags else []
    out = {"prof": prof, "flags": list(fl)}
    for m in ("abs", "ind", "capm"):
        d = rec_score(rows, sc / n, pool, mode=m)
        lo, mid, hi = d["lo"], d["mid"], d["hi"]
        if fl:
            lo, mid, hi = min(lo, 50), min(mid, 50), min(hi, 50)
        out[m] = (float(mid), float(lo), float(hi), bool(d["hi"] - d["lo"] <= RELIABLE_WIDTH))
    return out


def _val_on(prof):
    st.session_state[f"rs_val_{prof}"] = True


__all__ = [n for n in dir() if not n.startswith("__")]  # 밑줄로 시작하는 도우미도 다음 모듈에서 쓸 수 있게
