"""데이터 점검: 재무·주가 데이터에서 계산 결과를 틀리게 만들 수 있는 통계적 이상을 찾아낸다.

각 점검은 {"level": "error"|"warn"|"info", "title", "detail", "fix"} 사전을 돌려준다.
- error: 계산이 틀렸을 가능성이 높음 (그래프에서 해당 구간을 비우는 등 자동 보정이 들어감)
- warn : 데이터끼리 맞지 않음 (확인 필요)
- info : 실제로 일어난 일일 수 있지만 해석에 주의
"""
import numpy as np
import pandas as pd
import streamlit as st

from research_data import STALE_EPS_DAYS, annual_eps_table

SPLIT_RATIOS = (2, 3, 4, 5, 8, 10, 15, 20, 25, 50)
LEVEL_ORDER = {"error": 0, "warn": 1, "info": 2}
LEVEL_ICON = {"error": "🔴", "warn": "🟠", "info": "🔵"}


def _naive(s):
    if s is None:
        return None
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def _d(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d")


def _near_split_ratio(r: float, tol: float = 0.06):
    """r(또는 1/r)이 흔한 액면분할·병합 비율과 tol 안으로 같으면 그 비율."""
    if not np.isfinite(r) or r <= 0:
        return None
    x = r if r >= 1 else 1 / r
    k = min(SPLIT_RATIOS, key=lambda k: abs(np.log(x / k)))  # 가장 가까운 비율
    return k if abs(x / k - 1) <= tol else None


def _runs(mask: pd.Series):
    """True가 이어지는 구간들의 (시작, 끝, 길이)."""
    out, start, n, prev = [], None, 0, None
    for t, v in mask.items():
        if v:
            if start is None:
                start, n = t, 0
            n += 1
            prev = t
        elif start is not None:
            out.append((start, prev, n))
            start = None
    if start is not None:
        out.append((start, prev, n))
    return out


# ----------------------------------------------------------------------
def check_prices(hist, splits) -> list:
    """주가: 하루 ±40% 넘는 변동. 분할 비율과 맞는데 분할 기록이 없으면 '분할 미반영' 의심."""
    h = _naive(hist)
    if h is None or len(h) < 3:
        return []
    sp = _naive(splits) if splits is not None and len(splits) else pd.Series(dtype=float)
    r = (h / h.shift(1)).dropna()
    out = []
    for t, x in r[(r > 1.4) | (r < 1 / 1.4)].items():
        near_split = len(sp) and (abs((sp.index - t).days) <= 5).any()
        k = _near_split_ratio(float(x))
        if near_split:
            out.append(dict(level="error", title=f"{_d(t)} 액면분할일 주가가 조정되지 않음",
                            detail=f"분할 기록이 있는 날 주가가 하루에 {x:.2f}배가 됐습니다. 수정주가가 아니면 그 이전 PER·수익률이 분할 비율만큼 틀립니다.",
                            fix="수정주가(분할 반영) 데이터로 다시 받아야 합니다."))
        elif k:
            out.append(dict(level="warn", title=f"{_d(t)} 주가가 하루에 {x:.2f}배 — 기록 없는 분할·병합 의심",
                            detail=f"변동 폭이 {k}:1 분할·병합 비율과 거의 같습니다. 분할 기록이 빠졌다면 이 날 전후의 주당 지표가 {k}배 어긋납니다.",
                            fix="공시에서 분할·병합 여부를 확인하세요."))
        else:
            out.append(dict(level="info", title=f"{_d(t)} 주가 하루 변동 {(x - 1) * 100:+.0f}%",
                            detail="실제 급등락일 수 있지만, 이 날을 끼는 평균·변동성 통계는 크게 흔들립니다.", fix=""))
    return out[:6]


def check_annual_eps(sec, splits) -> list:
    """연간 EPS: 공시일 이상, 오래된 EPS, 분할 보정 누락 의심."""
    tb = annual_eps_table(sec, splits)
    if tb is None:
        return []
    out = []
    lag = (pd.Series(tb.index, index=tb.index) - tb["회계연도말"]).dt.days.values
    for (avail, row), g in zip(tb.iterrows(), lag):
        fy = _d(row["회계연도말"])
        if g < 0:
            out.append(dict(level="error", title=f"{fy} 회계연도 EPS가 회계연도가 끝나기 전에 공시된 것으로 나옴",
                            detail=f"공시일 {_d(avail)}가 회계연도 말보다 {-g}일 앞섭니다. 미래 정보를 미리 쓰는 셈이라 과거 PER이 틀립니다.",
                            fix="공시일 데이터 오류입니다."))
        elif g > 400:
            out.append(dict(level="error", title=f"{fy} 회계연도 EPS 공시일이 {g}일 뒤로 잡힘",
                            detail=f"연간 실적은 보통 회계연도 말 뒤 3개월 안에 나옵니다. 공시일이 {_d(avail)}로 잡혀 그 사이 PER이 더 오래된 EPS로 계산됩니다 "
                                   "(나중 보고서에 다시 실린 날을 공시일로 쓰면 생기는 오류).",
                            fix=f"회계연도 말에서 {STALE_EPS_DAYS}일 넘게 지난 EPS 구간은 PER 그래프에서 비웠습니다."))
    # 연속한 해의 EPS 비율이 그 사이 분할 비율과 같으면 보정 누락 의심
    sp = _naive(splits) if splits is not None and len(splits) else pd.Series(dtype=float)
    e = tb["EPS"].values
    for i in range(1, len(tb)):
        a, b = e[i - 1], e[i]
        if not (a > 0 and b > 0):
            continue
        between = sp[(sp.index > tb.index[i - 1]) & (sp.index <= tb.index[i])] if len(sp) else sp
        r = b / a
        for t, f in between.items():
            if f > 1 and (abs(r * f - 1) <= 0.06 or abs(r / f - 1) <= 0.06):
                out.append(dict(level="warn", title=f"{_d(tb['회계연도말'].iloc[i])} EPS가 직전 해의 {r:.3f}배 — 분할 보정 누락 의심",
                                detail=f"{_d(t)}에 {f:g}:1 액면분할이 있었고 두 해 EPS 비율({r:.3f})이 분할 비율과 거의 같습니다.",
                                fix="이익이 정말 그만큼 변했는지 손익계산서로 확인하세요."))
    # 지금 쓰는 EPS가 너무 오래됨
    last_fy = tb["회계연도말"].iloc[-1]
    age = (pd.Timestamp.today() - last_fy).days
    if age > STALE_EPS_DAYS:
        out.append(dict(level="warn", title=f"최근 연간 EPS가 {age}일 전 회계연도({_d(last_fy)}) 것",
                        detail="최신 사업보고서가 아직 반영되지 않았습니다. '직전 사업연도 PER'은 비우고 12M PER을 대신 보여줍니다.",
                        fix=""))
    return out


def check_quarterly(edates, since=None) -> list:
    """분기 실적: 발표 간격이 너무 벌어지면(분기 누락) 12M EPS가 4개 분기가 아닐 수 있다."""
    if not isinstance(edates, pd.DataFrame) or edates.empty or "Reported EPS" not in edates:
        return []
    e = _naive(edates["Reported EPS"].dropna().astype(float))
    e = e[~e.index.duplicated()]
    if len(e) < 5:
        return []
    out = []
    gaps = pd.Series(e.index[1:] - e.index[:-1], index=e.index[1:]).dt.days
    if since is not None:  # 그래프에 그리는 기간만 본다
        gaps = gaps[gaps.index >= pd.Timestamp(since)]
    miss = gaps[gaps > 135]
    if len(miss):
        ds = ", ".join(f"{_d(t)}({g}일)" for t, g in list(miss.items())[:4]) + (" 외" if len(miss) > 4 else "")
        out.append(dict(level="warn", title=f"분기 실적 누락 {len(miss)}곳",
                        detail=f"발표 간격이 4.5개월을 넘는 곳: {ds}. 그 뒤 1년은 최근 4개 분기가 아닌 값을 더하게 됩니다.",
                        fix="4개 분기가 400일 넘게 흩어진 구간의 12M PER은 비웠습니다."))
    for t, g in gaps[gaps < 45].items():
        out.append(dict(level="warn", title=f"{_d(t)} 분기 실적 발표가 {g}일 만에 또 있음",
                        detail="같은 분기가 두 번 들어갔거나(정정 발표) 날짜가 잘못됐을 수 있습니다. 12M EPS에 같은 분기가 두 번 더해질 수 있습니다.",
                        fix=""))
    # 한 분기 EPS가 앞뒤와 동떨어진 값 (부호가 같고 중앙값의 8배 이상)
    med = e.abs().rolling(5, center=True, min_periods=3).median()
    for t, v in e.items():
        m = med.get(t)
        if m and m > 0 and abs(v) > 8 * m:
            out.append(dict(level="info", title=f"{_d(t)} 분기 EPS {v:.2f}가 주변 분기의 {abs(v) / m:.0f}배",
                            detail="일회성 이익·손실이거나 단위 오류일 수 있습니다. 이 분기가 들어간 1년 동안 12M PER이 크게 흔들립니다.",
                            fix=""))
    return out[:8]


def check_eps_consistency(sec, splits, edates) -> list:
    """연간 EPS ≈ 그 해 4개 분기 EPS 합 인지 대조. 비율이 분할 비율과 같으면 분할 보정 오류."""
    sp = _naive(splits) if splits is not None and len(splits) else pd.Series(dtype=float)
    tb = annual_eps_table(sec, splits)
    if tb is None or not isinstance(edates, pd.DataFrame) or edates.empty or "Reported EPS" not in edates:
        return []
    e = _naive(edates["Reported EPS"].dropna().astype(float))
    e = e[~e.index.duplicated()]
    if len(e) < 4:
        return []
    ttm = e.rolling(4).sum()
    out = []
    for _, row in tb.iterrows():
        fy = row["회계연도말"]
        after = ttm[(ttm.index > fy) & (ttm.index <= fy + pd.Timedelta(days=100))].dropna()
        if not len(after) or not row["EPS"] or row["EPS"] <= 0 or after.iloc[0] <= 0:
            continue
        r = float(row["EPS"]) / float(after.iloc[0])
        k = _near_split_ratio(r, tol=0.15)  # 분기 합은 조정 EPS라 조금 다를 수 있어 여유를 둔다
        # 그 회계연도 뒤에 실제로 있었던 분할 비율만 '분할 오류'로 본다 (2배 차이는 GAAP·조정 EPS 차이로도 흔함)
        later = sp[sp.index > fy - pd.Timedelta(days=30)]
        known = {int(round(f if f >= 1 else 1 / f)) for f in later.values if np.isfinite(f) and f > 0}
        if k and k > 1 and k in known:
            out.append(dict(level="error", title=f"{_d(fy)} 연간 EPS와 4개 분기 합이 {k}배 차이",
                            detail=f"연간 {row['EPS']:.3f} vs 분기 합 {after.iloc[0]:.3f}. 차이가 {k}:1 분할 비율과 같아 한쪽이 분할 보정이 안 된 것으로 보입니다.",
                            fix="직전 사업연도 PER과 12M PER 중 한쪽이 그만큼 틀립니다."))
        elif r > 1.6 or r < 1 / 1.6:
            out.append(dict(level="info", title=f"{_d(fy)} 연간 EPS와 4개 분기 합이 {r:.2f}배 차이",
                            detail=f"연간(GAAP 희석) {row['EPS']:.3f} vs 분기 발표 합 {after.iloc[0]:.3f}. "
                                   "분기 발표치는 일회성 항목을 뺀 조정 EPS인 경우가 많아 차이가 날 수 있습니다.",
                            fix=""))
    return out


def check_per_gap(ann, trail, hist=None, ann_dates=None, trail_dates=None) -> list:
    """직전 사업연도 PER과 12M PER 비교.
    - 하루 사이 PER이 크게 뛰었는데 주가는 그대로면 계산 오류(EPS가 엉뚱하게 바뀜).
    - 두 PER이 오래 3배 넘게 벌어지면 실적 급변 구간(정상일 수 있음) 또는 EPS 시점 오류."""
    out = []
    a = _naive(ann).dropna() if ann is not None else None
    t = _naive(trail).dropna() if trail is not None else None
    if a is not None and len(a) > 20 and t is not None and len(t) > 20:
        j = pd.concat([a.rename("a"), t.rename("t")], axis=1, join="inner").dropna()
        j = j[(j["a"] > 0) & (j["t"] > 0)]
        if len(j) > 20:
            ratio = j["a"] / j["t"]
            big = (ratio > 3) | (ratio < 1 / 3)
            for s, e_, n in _runs(big):
                if n < 40:
                    continue
                med = float(ratio[s:e_].median())
                level = "warn" if n > 260 else "info"
                out.append(dict(level=level, title=f"{_d(s)} ~ {_d(e_)} 두 PER이 {med:.1f}배 벌어짐 ({n}거래일)",
                                detail="직전 사업연도 PER은 1년에 한 번만 EPS가 바뀌어서, 이익이 빠르게 늘거나 줄 때는 12M PER과 몇 달간 크게 벌어집니다. "
                                       + ("1년 넘게 이어지면 EPS 공시일·분할 보정 오류일 가능성이 큽니다." if n > 260 else "이 구간은 12M PER이 더 현실에 가깝습니다."),
                                fix=""))
    # PER = 주가 ÷ EPS 이고 EPS는 공시일에만 바뀐다 → 공시일이 아닌 날 PER 변동은 주가 변동과 같아야 한다 (자기 일관성 검사)
    h = _naive(hist).dropna() if hist is not None else None
    for name, s, dates in (("직전 사업연도 PER", a, ann_dates), ("12M PER", t, trail_dates)):
        if s is None or len(s) < 20 or h is None or not len(h):
            continue
        full = _naive(ann if name.startswith("직전") else trail).reindex(h.index)  # 거래일 기준으로 맞춰 하루 변동만 비교
        r = (full / full.shift(1)).dropna()
        pr = (h / h.shift(1)).reindex(r.index)
        ev = pd.DatetimeIndex(pd.to_datetime(list(dates))).tz_localize(None) if dates is not None and len(dates) else pd.DatetimeIndex([])
        bad = []
        for d, x in r.items():
            p = pr.get(d)
            if p is None or not np.isfinite(p) or abs(np.log(x) - np.log(p)) < 0.05:
                continue
            if len(ev) and (abs((ev - d).days) <= 4).any():
                continue  # 실적 공시일 근처 → EPS가 바뀐 정상 변동
            bad.append((d, x, p))
        if bad:
            d, x, p = bad[0]
            out.append(dict(level="error", title=f"{name}: 실적 공시가 없는 날 EPS가 바뀜 ({len(bad)}일, 처음 {_d(d)})",
                            detail=f"{_d(d)} PER은 {x:.2f}배가 됐는데 주가는 {p:.2f}배였습니다. PER 계산에 쓰인 EPS가 공시일이 아닌 날 바뀐 것이라 계산 오류입니다.",
                            fix="EPS 공시일 데이터를 확인해야 합니다."))
    return out[:10]


def robust_outliers(s: pd.Series, z: float = 8.0) -> pd.Series:
    """중앙값·MAD 기준 로버스트 z-점수가 z를 넘는 값 (평균·표준편차는 이상치 자체에 끌려가므로 쓰지 않음)."""
    v = _naive(s).dropna()
    if len(v) < 30:
        return pd.Series(dtype=float)
    lv = np.log(v[v > 0])
    med = lv.median()
    mad = (lv - med).abs().median() * 1.4826
    if not mad or not np.isfinite(mad):
        return pd.Series(dtype=float)
    zs = (lv - med) / mad
    return v[zs.abs() > z]


def check_outliers(series: dict) -> list:
    out = []
    for name, s in series.items():
        if s is None:
            continue
        o = robust_outliers(s)
        if len(o):
            out.append(dict(level="info", title=f"{name}: 극단값 {len(o)}일 ({_d(o.index[0])} ~ {_d(o.index[-1])})",
                            detail=f"평소 수준(중앙값 {float(_naive(s).dropna().median()):.1f})에서 크게 벗어난 값입니다 (최대 {float(o.max()):.1f}). "
                                   "적자 직후처럼 EPS가 0에 가까울 때 생기며, 평균·±1σ 밴드를 왜곡하므로 축은 98% 분위까지만 그립니다.",
                            fix=""))
    return out


def run_checks(hist, sec, splits, edates, ann=None, trail=None, ann_dates=None, trail_dates=None) -> list:
    issues = []
    for fn in (lambda: check_prices(hist, splits), lambda: check_annual_eps(sec, splits), lambda: check_quarterly(edates, since=_naive(hist).index[0] if hist is not None and len(hist) else None),
               lambda: check_eps_consistency(sec, splits, edates), lambda: check_per_gap(ann, trail, hist, ann_dates, trail_dates),
               lambda: check_outliers({"직전 사업연도 PER": ann, "12M PER": trail})):
        try:
            issues += fn()
        except Exception as e:  # noqa: BLE001  점검 하나가 실패해도 나머지는 보여준다
            issues.append(dict(level="info", title="점검 하나를 실행하지 못함", detail=str(e), fix=""))
    return sorted(issues, key=lambda x: LEVEL_ORDER.get(x["level"], 9))


def render_checks(issues: list, key: str = "") -> None:
    issues = [i for i in issues if i.get("level") in ("error", "warn")]  # 해석용 참고(PER 괴리 등)는 표시하지 않음
    n_err = sum(i["level"] == "error" for i in issues)
    n_warn = sum(i["level"] == "warn" for i in issues)
    if not issues:
        st.caption("✅ 데이터 점검: 공시일·액면분할·분기 누락·극단값에서 이상을 찾지 못했습니다.")
        return
    head = f"데이터 점검 — 오류 {n_err} · 주의 {n_warn}"
    with st.expander(head, expanded=bool(n_err)):
        for i in issues:
            body = f"{LEVEL_ICON.get(i['level'], '')} **{i['title']}**  \n{i['detail']}"
            if i.get("fix"):
                body += f"  \n→ {i['fix']}"
            st.markdown(body)
        st.caption("점검 항목: 주가 급변·분할 미반영, 연간 EPS 공시일·오래된 EPS·분할 보정, 분기 실적 누락·중복, "
                   "연간 EPS vs 4개 분기 합, 두 PER 괴리, 로버스트 z-점수(중앙값·MAD) 극단값.")


__all__ = [n for n in dir() if not n.startswith("__")]
