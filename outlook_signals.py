"""기업 리서치 판단의 신호 계산·판정·과거 검증(워크포워드, 업종별 검증)."""
import numpy as np
import pandas as pd
import streamlit as st

from research_data import RATIOS, _split_factor, annual_eps_table, compute_ratios

from outlook_data import *  # noqa: F401,F403

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


__all__ = [n for n in dir() if not n.startswith("__")]  # 밑줄로 시작하는 도우미도 다음 모듈에서 쓸 수 있게
