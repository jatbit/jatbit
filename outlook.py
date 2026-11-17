"""기업 리서치: 고른 재무비율로 '상승 우세 / 중립 / 하락 우세'를 판단하고, 같은 방법을 과거에 적용했을 때의 결과를 보여준다.

방법 (방법론 해설 문서의 '앱에 넣는다면' 설계)
- 수익성·안정성·효율성·현금흐름 비율: 전년보다 좋아졌으면 +1, 나빠졌으면 −1 (피오트로스키 F-스코어 방식)
- 성장률: 플러스 성장 +1, 마이너스 성장 −1
- 밸류에이션(PER 등): 고른 기간 안에서 그때까지의 값 중 상위 20%면 −1, 하위 20%면 +1 (자기 과거 대비 평균회귀)
- 점수 합 ÷ 근거 수가 +0.25 이상이면 상승 우세, −0.25 이하면 하락 우세
과거 검증: 기간 안의 매달 첫 거래일마다 그때까지 공시된 숫자로 다시 판정하고 그 뒤 1·3·6개월·1년 주가 수익률을 붙인다.
기대수익률: 과거에 지금과 같은 판정이었던 달들의 이후 수익률 평균 (사례가 적으면 기간 전체 평균 쪽으로 당김).
"""
import numpy as np
import pandas as pd
import streamlit as st

from research_data import RATIOS, _split_factor, compute_ratios

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


def _ratio_signal(k: str, row: pd.Series, j: int, pcols: list):
    """연간 재무비율 k의 j번째 사업연도 신호 → (신호, 설명) 또는 None(값 없음)."""
    v = row.iloc[j]
    if not np.isfinite(v):
        return None
    unit = RATIOS[k][0]
    fmt = (lambda x: f"{x:.1f}%") if unit == "%" else (lambda x: f"{x:.2f}{unit}")
    if k in VALN:
        past = np.array([row[c] for c in pcols if c <= row.index[j] and np.isfinite(row[c])])
        if len(past) < 3:
            return 0, f"{fmt(v)} (비교할 과거 값 부족)"
        p = _pctile(past, v)
        return _val_sig(p, DIR[k]), f"{fmt(v)} · 기간 내 백분위 {p * 100:.0f}%"
    if k in GROWTH:
        return (1 if v > 1 else (-1 if v < -1 else 0)), fmt(v)
    if j == 0:
        return None
    prev = row.iloc[j - 1]
    if not np.isfinite(prev):
        return None
    ch = v - prev
    sig = int(np.sign(ch)) * DIR[k] if abs(ch) > 0.02 * abs(prev) else 0
    return sig, f"{fmt(prev)} → {fmt(v)}"


def _daily_signal(s: pd.Series, unit: str, start, d):
    vals = s[(s.index >= start) & (s.index <= pd.Timestamp(d))].dropna()
    if len(vals) < 120:
        return None
    v = float(vals.iloc[-1])
    p = _pctile(vals.values, v)
    direction = 1 if unit == "%" else -1  # 이익수익률·배당수익률 같은 %는 높을수록 쌈
    return _val_sig(p, direction), (f"{v:.2f}%" if unit == "%" else f"{v:.1f}배") + f" · 기간 내 백분위 {p * 100:.0f}%"


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


def render_outlook(sec, R: dict, ticker: str, sel: list, daily: dict) -> None:
    """sel: 고른 연간 재무비율 이름, daily: {표시 이름: (일별 시계열, 단위)} — 고른 일별 밸류에이션."""
    from glossary import table_html
    from options import mgrid

    st.markdown("##### 상승·하락 판단")
    hist = R.get("hist")
    if sec is None or not isinstance(hist, pd.Series) or not len(hist):
        st.caption("재무제표나 주가가 없어 판단할 수 없습니다.")
        return
    use = [k for k in sel if k in DIR or k in GROWTH]
    skipped = [k for k in sel if k not in use]
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

    mkt, mname = None, None
    sym = "^KS11" if ticker.endswith(".KS") else ("^KQ11" if ticker.endswith(".KQ") else "^GSPC")
    try:
        mkt = _naive(_index_hist(sym))
        mname = {"^KS11": "코스피", "^KQ11": "코스닥", "^GSPC": "S&P 500"}[sym]
    except Exception:  # noqa: BLE001
        mkt = None

    def signals(j: int, d_daily) -> dict:
        out = {}
        for k in use:
            s = _ratio_signal(k, rt.loc[k], j, pcols)
            if s is not None:
                out[k] = s
        for nm, (s, u) in dser.items():
            r = _daily_signal(s, u, start, d_daily)
            if r is not None:
                out[nm] = r
        return out

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
        sg = signals(cols.index(c), d)
        sc, n = sum(v[0] for v in sg.values()), len(sg)
        rec = dict(d=d, fy=c, sig=sg, score=sc, n=n, verdict=_verdict(sc, n))
        for lab, days in HORIZONS:
            rec[lab] = _fwd(h, d, days)
        m = _fwd(mkt, d, 365) if mkt is not None else np.nan
        rec["ex"] = rec["1년"] - m if np.isfinite(rec["1년"]) and np.isfinite(m) else np.nan
        rows.append(rec)
    if not rows:
        st.caption("고른 기간 안에 판정할 수 있는 시점이 없습니다.")
        return

    # ---- 지금 판단: 가장 최근 공시 사업연도 재무비율 + 오늘 기준 밸류에이션
    now_d = min(end, h.index[-1])
    fy_now = fy_at(now_d)
    now_sig = signals(cols.index(fy_now), now_d)
    sc, n = sum(v[0] for v in now_sig.values()), len(now_sig)
    vd = _verdict(sc, n)
    if vd is None:
        st.caption("고른 항목의 값이 부족해 판단할 수 없습니다 (전년 값이 없거나 비교할 과거 값이 적음).")
        return
    msg = (f"**{vd}** — 점수 {sc:+d} / 근거 {n}개 ({fy_now:%Y}년 사업연도 공시 {avail[fy_now]:%Y-%m-%d} 기준, "
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
    st.caption(f"과거 판정이 '{vd}'였던 달들의 이후 수익률로 계산합니다. 사례가 적을수록 기간 전체 평균 쪽으로 당겨서 계산하고, "
               "예상 범위는 그 수익률의 하위 25%~상위 25%입니다 (같은 판정 사례가 3개 미만이면 전체 기준).")

    past = [r for r in rows if np.isfinite(r["1년"])]
    dirn = [r for r in past if r["score"] != 0 and r["1년"] != 0]
    hit = sum(1 for r in dirn if np.sign(r["score"]) == np.sign(r["1년"]))
    exs = [r["ex"] for r in past if r["verdict"] == vd and np.isfinite(r["ex"])]
    items = [("점수 방향 적중률", f"{hit}/{len(dirn)} ({hit / len(dirn) * 100:.0f}%)" if dirn else "-",
              "점수 부호와 이후 1년 방향이 같았던 달의 비율"),
             (f"{mname} 대비 초과수익" if mname else "시장 대비 초과수익", _p(np.mean(exs)) if exs else "-",
              f"과거 판정이 '{vd}'일 때 1년 평균")]
    fl = fs.loc[fy_now] if fy_now in fs.index else None
    if fl is not None and fl["항목"] >= 5:
        items.append(("F-스코어", f"{int(fl['점수'])}/{int(fl['항목'])}", "재무 상태 개선 항목 수 (참고)"))
    mgrid(items)

    # ---- 근거별 신호
    trs = []
    for k, (s, desc) in now_sig.items():
        hp_ = [r for r in past if k in r["sig"] and r["sig"][k][0] != 0 and r["1년"] != 0]
        hk = sum(1 for r in hp_ if np.sign(r["sig"][k][0]) == np.sign(r["1년"]))
        trs.append([k, desc, _sig_txt(s), f"{hk}/{len(hp_)}" if hp_ else "-"])
    st.markdown(table_html(trs, ["근거", "값", "신호", "과거 적중 (1년)"]), unsafe_allow_html=True)
    notes = []
    if skipped:
        notes.append(f"{', '.join(skipped)}: 높고 낮음의 좋고 나쁨이 정해져 있지 않아 판단에서 뺐습니다.")
    missing = [k for k in use if k not in now_sig] + [k for k in dser if k not in now_sig]
    if missing:
        notes.append(f"{', '.join(missing)}: 값이 부족해 이번 판단에서 빠졌습니다.")
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
        df = pd.DataFrame([[f"{r['d']:%Y-%m-%d}", f"{r['fy']:%Y}", f"{r['score']:+d}/{r['n']}", r["verdict"] or "-"]
                           + [_p(r[lab]) for lab, _ in HORIZONS] for r in reversed(rows)],
                          columns=["판정일", "사업연도", "점수", "판정"] + [lab for lab, _ in HORIZONS])
        st.dataframe(df, hide_index=True, use_container_width=True, height=320)
        st.caption("매달 첫 거래일에 그때까지 공시된 숫자와 그날의 밸류에이션으로 다시 판정했습니다. 수익률은 판정일 종가부터 "
                   "1·3·6개월, 1년 뒤 종가까지입니다 (배당 제외). 아직 기간이 안 지난 칸은 비어 있습니다.")
    st.caption("과거 통계로 낸 추정이며 예측을 보장하지 않습니다. 매달 판정하므로 이웃한 달의 1년 수익률은 기간이 겹쳐, "
               "실제로 독립된 사례 수는 표의 횟수보다 적습니다.")
