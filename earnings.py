"""실적·추정치 (기업 리서치 탭)
- 실적 서프라이즈: 분기별 예상 EPS 대비 실제 EPS
- 실적 발표 때 주가 반응: 발표 시각(장 전·장 후)에 맞춘 다음날·5일·20일 수익률과 시장 대비 초과 반응, 발표 전후 실현변동성
- 다음 실적 발표의 옵션 내재 예상 변동폭: 발표를 포함하는 만기와 아닌 만기의 ATM 내재변동성 차이로 '발표 하루'의 변동을 분리하고,
  발표 뒤 남을 변동성(IV 급락 크기)을 추정
- 애널리스트 추정치 변화: EPS 컨센서스 추이(7·30·60·90일 전), 상향·하향 조정 수, 투자의견·목표주가 변경 이력
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from glossary import col_help
from options import mgrid

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None


def _naive(idx):
    idx = pd.to_datetime(idx)
    return idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx


def _safe(fn, default=None):
    try:
        v = fn()
        return default if v is None else v
    except Exception:  # noqa: BLE001
        return default


@st.cache_data(ttl=60 * 60, show_spinner="애널리스트 추정치를 불러오는 중...")
def load_estimates(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    return {"trend": _safe(lambda: t.eps_trend, pd.DataFrame()), "rev": _safe(lambda: t.eps_revisions, pd.DataFrame()),
            "eps": _safe(lambda: t.earnings_estimate, pd.DataFrame()), "sales": _safe(lambda: t.revenue_estimate, pd.DataFrame()),
            "ud": _safe(lambda: t.upgrades_downgrades, pd.DataFrame()), "hist": _safe(lambda: t.earnings_history, pd.DataFrame())}


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_bench(sym: str) -> pd.Series:
    h = yf.Ticker(sym).history(period="10y", auto_adjust=False)["Close"].dropna()
    h.index = _naive(h.index).normalize()
    return h


# ----------------------------------------------------------------------
# 계산
# ----------------------------------------------------------------------
def surprise_table(edates: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(edates, pd.DataFrame) or edates.empty or "Reported EPS" not in edates:
        return pd.DataFrame()
    e = edates.copy()
    e["when"] = pd.to_datetime(e.index)
    e = e[e["Reported EPS"].notna()].sort_values("when")
    est = pd.to_numeric(e.get("EPS Estimate"), errors="coerce")
    rep = pd.to_numeric(e["Reported EPS"], errors="coerce")
    sp = (rep - est) / est.abs() * 100
    sp = sp.where(sp.abs() <= 300)  # 예상치 단위가 어긋난 비정상 자료는 뺌
    out = pd.DataFrame({"when": list(e["when"]), "est": est.values, "rep": rep.values, "surp": sp.values})  # 현지 시각(시간대) 유지
    return out.drop_duplicates("when").reset_index(drop=True)


def _timing(ts) -> str:
    """발표 시각 → 'bmo'(장 전·장중) / 'amc'(장 마감 후) / 'unk'(시각 정보 없음)."""
    ts = pd.Timestamp(ts)
    h = ts.hour + ts.minute / 60
    if h == 0:
        return "unk"
    return "amc" if h >= 15.5 else "bmo"


def reactions(S: pd.DataFrame, px: pd.Series, bench: pd.Series | None) -> pd.DataFrame:
    """실적 발표별 주가 반응. 기준가 = 발표 전 마지막 종가, 반응일 = 발표 내용이 처음 반영된 거래일."""
    if S.empty or px is None or len(px) < 30:
        return pd.DataFrame()
    p = px.copy()
    p.index = _naive(p.index).normalize()
    p = p[~p.index.duplicated(keep="last")].astype(float)
    lr = np.log(p).diff()
    days = p.index
    rows = []
    for _, r in S.iterrows():
        ts = pd.Timestamp(r["when"])
        tsn = ts.tz_localize(None) if ts.tzinfo is not None else ts
        d0 = tsn.normalize()
        kind = _timing(ts)
        i = days.searchsorted(d0)  # 발표일 또는 그 다음 거래일 위치
        if i >= len(days):
            continue
        on_day = days[i] == d0
        if kind == "amc":
            base_i = i if on_day else i - 1
            react_i = base_i + 1
        elif kind == "bmo":
            base_i = i - 1
            react_i = i
        else:  # 시각을 모르면 발표 전날 종가 → 발표 다음 거래일 종가(이틀)로 넓게 잡음
            base_i = i - 1
            react_i = i + 1 if on_day else i
        if base_i < 0 or react_i >= len(days):
            continue
        b = p.iloc[base_i]

        def ret(k):
            j = react_i + k - 1
            return float(p.iloc[j] / b - 1) if j < len(p) else np.nan

        row = {"발표일": d0, "시점": {"amc": "장 마감 후", "bmo": "장 전·장중", "unk": "시각 미상"}[kind],
               "예상 EPS": r["est"], "실제 EPS": r["rep"], "서프라이즈": r["surp"],
               "다음날 반응": ret(1), "5일 반응": ret(5), "20일 반응": ret(20),
               "실적 전 실현변동성": float(lr.iloc[max(base_i - 9, 1):base_i + 1].std() * math.sqrt(252)) if base_i > 10 else np.nan,
               "실적 후 실현변동성": float(lr.iloc[react_i + 1:react_i + 11].std() * math.sqrt(252)) if react_i + 11 <= len(lr) else np.nan}
        if bench is not None and len(bench):
            bb = bench.reindex(days).ffill()
            bbase = bb.iloc[base_i]
            j = react_i
            row["시장 대비"] = row["다음날 반응"] - (float(bb.iloc[j] / bbase - 1) if np.isfinite(bbase) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def implied_event(atm: list, t_event: float) -> dict:
    """atm: [(T년, ATM IV)] 만기순. 발표(t_event년 뒤)를 포함하는 첫 만기와 그 앞(또는 그다음) 만기로 발표 하루 변동을 분리.
    포함 만기: σ1²·T1 = σb²·T1 + m² (σb = 평소 변동성, m = 발표일 추가 변동)."""
    atm = [(T, v) for T, v in atm if np.isfinite(v) and T > 0]
    after = [(T, v) for T, v in atm if T >= t_event]
    before = [(T, v) for T, v in atm if T < t_event and T > 2 / 365]
    if not after:
        return {}
    T1, s1 = after[0]
    sb, how = None, ""
    if before:
        sb, how = before[-1][1], "발표 전 만기 IV"
    elif len(after) >= 2:
        T2, s2 = after[1]
        var_b = (s2 * s2 * T2 - s1 * s1 * T1) / (T2 - T1) if T2 > T1 else np.nan
        if np.isfinite(var_b) and var_b > 0:
            sb, how = math.sqrt(var_b), "발표 뒤 두 만기의 기간구조"
    if sb is None:
        return {"T1": T1, "s1": s1}
    m2 = (s1 * s1 - sb * sb) * T1
    m = math.sqrt(m2) if m2 > 0 else 0.0
    return {"T1": T1, "s1": s1, "sb": sb, "move": m, "abs_move": m * math.sqrt(2 / math.pi), "how": how,
            "crush": s1 - sb}


def _next_earnings(R: dict):
    cal = R.get("calendar") or {}
    ed = cal.get("Earnings Date")
    cands = []
    if ed:
        for d in (ed if isinstance(ed, (list, tuple)) else [ed]):
            try:
                cands.append(pd.Timestamp(d))
            except Exception:  # noqa: BLE001
                pass
    e = R.get("edates")
    when = None
    if isinstance(e, pd.DataFrame) and not e.empty:
        idx = pd.to_datetime(e.index)
        now = pd.Timestamp.now(tz=idx.tz) if idx.tz is not None else pd.Timestamp.now()
        fut = sorted(i for i in idx if i >= now - pd.Timedelta(hours=12))
        if fut:
            when = fut[0]
    if when is None and cands:
        c = [x for x in cands if x.tz_localize(None) >= pd.Timestamp.now().normalize()] if all(x.tzinfo is None for x in cands) else cands
        when = sorted(c)[0] if c else None
    return when


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
def _p(x, nd=1, sign=False):
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%"


def render_earnings(ticker: str, R: dict, domestic: bool = False) -> None:
    if yf is None:
        st.error("yfinance가 필요합니다.")
        return
    S = surprise_table(R.get("edates"))
    bench_sym = "^KS11" if domestic else "SPY"
    bench = _safe(lambda: load_bench(bench_sym))
    rx = reactions(S, R.get("hist"), bench)

    # ---- 요약
    st.markdown("##### 실적 서프라이즈와 주가 반응")
    if S.empty:
        st.info("Yahoo에 이 종목의 분기 실적(예상·실제 EPS) 기록이 없습니다.")
    else:
        sv = S["surp"].dropna()
        items = [("실적 상회 비율", f"{(sv > 0).mean() * 100:.0f}%" if len(sv) else "-", f"최근 {len(sv)}개 분기"),
                 ("서프라이즈 중앙값", f"{sv.median():+.1f}%" if len(sv) else "-", "예상치가 0에 가까운 분기에 휘둘리지 않게 중앙값")]
        if not rx.empty:
            r1 = rx["다음날 반응"].dropna()
            items += [("평균 주가 반응", _p(r1.mean(), 2, True)), ("평균 절대 반응", _p(r1.abs().mean(), 2)),
                      ("서프라이즈-반응 상관", f"{rx[['서프라이즈', '다음날 반응']].dropna().corr().iloc[0, 1]:.2f}"
                       if rx[["서프라이즈", "다음날 반응"]].dropna().shape[0] >= 4 else "-")]
        mgrid(items, cols=5)

        f = go.Figure()
        xs = [pd.Timestamp(w).strftime("%Y-%m-%d") for w in S["when"]]
        f.add_trace(go.Bar(x=xs, y=S["surp"].clip(-100, 100), name="서프라이즈(%)", marker_color=np.where(S["surp"] >= 0, "#5BB56E", "#E5534B"),
                           customdata=S["surp"], hovertemplate="%{x} 서프라이즈 %{customdata:+.1f}%<extra></extra>"))
        if not rx.empty:
            f.add_trace(go.Scatter(x=[d.strftime("%Y-%m-%d") for d in rx["발표일"]], y=rx["다음날 반응"] * 100, name="다음날 주가 반응(%)",
                                   yaxis="y2", mode="lines+markers", line=dict(color="#4C9BE8"),
                                   hovertemplate="%{x} 주가 반응 %{y:+.2f}%<extra></extra>"))
        f.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                        yaxis=dict(title="서프라이즈(%)", ticksuffix="%"),
                        yaxis2=dict(title="주가 반응(%)", overlaying="y", side="right", showgrid=False, ticksuffix="%"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(f, use_container_width=True)
        if not rx.empty:
            show = rx.sort_values("발표일", ascending=False).copy()
            show["발표일"] = show["발표일"].dt.strftime("%Y-%m-%d")
            for c in ("예상 EPS", "실제 EPS"):
                show[c] = show[c].map(lambda v: f"{v:,.2f}" if np.isfinite(v) else "-")
            show["서프라이즈"] = show["서프라이즈"].map(lambda v: f"{v:+.1f}%" if np.isfinite(v) else "-")
            for c in ("다음날 반응", "5일 반응", "20일 반응", "시장 대비"):
                if c in show:
                    show[c] = show[c].map(lambda v: _p(v, 2, True))
            for c in ("실적 전 실현변동성", "실적 후 실현변동성"):
                show[c] = show[c].map(lambda v: _p(v, 0))
            st.dataframe(show, use_container_width=True, hide_index=True, column_config=col_help(show.columns))
            if len(rx) >= 4:
                fs = go.Figure(go.Scatter(x=rx["서프라이즈"], y=rx["다음날 반응"] * 100, mode="markers",
                                          text=[d.strftime("%Y-%m") for d in rx["발표일"]],
                                          marker=dict(size=9, color=np.where(rx["다음날 반응"] >= 0, "#5BB56E", "#E5534B")),
                                          hovertemplate="%{text}<br>서프라이즈 %{x:+.1f}%<br>반응 %{y:+.2f}%<extra></extra>"))
                fs.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                                 xaxis=dict(title="서프라이즈(%)", ticksuffix="%", zeroline=True),
                                 yaxis=dict(title="다음날 주가 반응(%)", ticksuffix="%", zeroline=True))
                st.plotly_chart(fs, use_container_width=True)

    # ---- 다음 실적 발표와 옵션
    st.markdown("##### 다음 실적 발표 · 옵션 내재변동성")
    nxt = _next_earnings(R)
    if nxt is None:
        st.caption("다음 실적 발표일 정보가 없습니다.")
    else:
        kind = _timing(nxt)
        st.write(f"다음 실적 발표: **{pd.Timestamp(nxt):%Y-%m-%d}** "
                 + {"amc": "(장 마감 후)", "bmo": "(장 전)", "unk": ""}[kind])
        if domestic:
            st.caption("국내 개별 주식옵션 시세는 무료로 받을 수 없어, 실적 발표 예상 변동폭은 미국 종목만 계산합니다.")
        else:
            _render_event_iv(ticker, nxt, kind, rx)

    # ---- 추정치 변화
    st.markdown("##### 애널리스트 추정치 변화")
    E = load_estimates(ticker)
    tr = E.get("trend")
    if isinstance(tr, pd.DataFrame) and not tr.empty and "current" in tr:
        lab = {"0q": "이번 분기", "+1q": "다음 분기", "0y": "이번 회계연도", "+1y": "다음 회계연도"}
        def chg(per, col):
            try:
                a, b = float(tr.loc[per, "current"]), float(tr.loc[per, col])
                return (a / b - 1) if b else np.nan
            except Exception:  # noqa: BLE001
                return np.nan
        rv = E.get("rev")
        def rcount(per, cands):
            try:
                for c in cands:
                    if c in rv.columns:
                        return int(rv.loc[per, c])
            except Exception:  # noqa: BLE001
                pass
            return None
        per0 = "0y" if "0y" in tr.index else tr.index[0]
        ups = rcount(per0, ["upLast30days", "upLast30Days"])
        dns = rcount(per0, ["downLast30days", "downLast30Days"])
        mgrid([("EPS 추정치 변화 (30일)", _p(chg(per0, "30daysAgo"), 1, True), lab.get(per0, per0)),
               ("EPS 추정치 변화 (90일)", _p(chg(per0, "90daysAgo"), 1, True), lab.get(per0, per0)),
               ("상향 조정 (30일)", f"{ups}명" if ups is not None else "-"), ("하향 조정 (30일)", f"{dns}명" if dns is not None else "-")], cols=4)
        cols = [c for c in ("90daysAgo", "60daysAgo", "30daysAgo", "7daysAgo", "current") if c in tr.columns]
        xl = {"90daysAgo": "90일 전", "60daysAgo": "60일 전", "30daysAgo": "30일 전", "7daysAgo": "7일 전", "current": "현재"}
        f = go.Figure()
        for i, per in enumerate(tr.index):
            ys = pd.to_numeric(tr.loc[per, cols], errors="coerce").values
            f.add_trace(go.Scatter(x=[xl[c] for c in cols], y=ys, name=lab.get(per, per), mode="lines+markers",
                                   hovertemplate=lab.get(per, per) + " %{x} EPS %{y:,.2f}<extra></extra>"))
        f.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), yaxis_title="EPS 컨센서스",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(f, use_container_width=True)
        tb = pd.DataFrame({"기간": [lab.get(p, p) for p in tr.index]})
        for c in cols:
            tb[xl[c]] = [f"{float(v):,.2f}" if pd.notna(v) else "-" for v in tr[c]]
        tb["30일 변화"] = [_p(chg(p, "30daysAgo"), 1, True) for p in tr.index]
        tb["90일 변화"] = [_p(chg(p, "90daysAgo"), 1, True) for p in tr.index]
        if isinstance(rv, pd.DataFrame) and not rv.empty:
            for nm, cands in (("상향(7일)", ["upLast7days", "upLast7Days"]), ("상향(30일)", ["upLast30days", "upLast30Days"]),
                              ("하향(7일)", ["downLast7days", "downLast7Days"]), ("하향(30일)", ["downLast30days", "downLast30Days"])):
                tb[nm] = [rcount(p, cands) if rcount(p, cands) is not None else "-" for p in tr.index]
        st.dataframe(tb, use_container_width=True, hide_index=True)
    else:
        st.caption("EPS 추정치 추이 자료가 없습니다.")

    for key, title in (("eps", "EPS 컨센서스"), ("sales", "매출 컨센서스")):
        df = E.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty and "avg" in df:
            lab = {"0q": "이번 분기", "+1q": "다음 분기", "0y": "이번 회계연도", "+1y": "다음 회계연도"}
            show = pd.DataFrame({"기간": [lab.get(p, p) for p in df.index]})
            for c, nm in (("avg", "평균"), ("low", "최저"), ("high", "최고"), ("yearAgoEps", "1년 전 실적"), ("yearAgoRevenue", "1년 전 실적"),
                          ("numberOfAnalysts", "애널리스트 수"), ("growth", "성장률")):
                if c in df:
                    if c == "growth":
                        show[nm] = [_p(float(v), 1, True) if pd.notna(v) else "-" for v in df[c]]
                    elif c == "numberOfAnalysts":
                        show[nm] = [f"{int(v)}" if pd.notna(v) else "-" for v in df[c]]
                    elif key == "sales":
                        import money

                        show[nm] = [money.big(v) for v in df[c]]
                    else:
                        show[nm] = [f"{float(v):,.2f}" if pd.notna(v) else "-" for v in df[c]]
            with st.expander(title):
                st.dataframe(show, use_container_width=True, hide_index=True)

    ud = E.get("ud")
    if isinstance(ud, pd.DataFrame) and not ud.empty:
        u = ud.copy()
        u.index = _naive(u.index)
        u = u.sort_index(ascending=False)
        yr = u[u.index >= pd.Timestamp.now() - pd.DateOffset(years=1)]
        act = yr.get("Action", pd.Series(dtype=str)).astype(str).str.lower()
        mgrid([("투자의견 상향", f"{int((act == 'up').sum())}회", "최근 1년"), ("투자의견 하향", f"{int((act == 'down').sum())}회", "최근 1년"),
               ("신규 커버리지", f"{int((act == 'init').sum())}회", "최근 1년"), ("의견 유지", f"{int(act.isin(['main', 'reit']).sum())}회", "최근 1년")], cols=4)
        AK = {"up": "상향", "down": "하향", "main": "유지", "reit": "재확인", "init": "신규"}
        show = pd.DataFrame({"날짜": u.index.strftime("%Y-%m-%d"), "증권사": u.get("Firm"),
                             "변경": u.get("Action", pd.Series(dtype=str)).astype(str).str.lower().map(lambda a: AK.get(a, a)),
                             "이전 의견": u.get("FromGrade"), "새 의견": u.get("ToGrade")})
        if "currentPriceTarget" in u:
            show["목표주가"] = [f"{v:,.2f}" if pd.notna(v) and v else "-" for v in u["currentPriceTarget"]]
            show["이전 목표주가"] = [f"{v:,.2f}" if pd.notna(v) and v else "-" for v in u.get("priorPriceTarget", pd.Series(np.nan, index=u.index))]
            pt = u[pd.to_numeric(u["currentPriceTarget"], errors="coerce") > 0]
            pt = pt[pt.index >= pd.Timestamp.now() - pd.DateOffset(years=2)]
            hist = R.get("hist")
            if len(pt) >= 3 and isinstance(hist, pd.Series) and len(hist):
                h = hist.copy()
                h.index = _naive(h.index)
                h = h[h.index >= pt.index.min() - pd.Timedelta(days=30)]
                f = go.Figure()
                f.add_trace(go.Scatter(x=h.index, y=h.values, name="주가", line=dict(color="#888", width=1.3)))
                colors = pt.get("Action", pd.Series("", index=pt.index)).astype(str).str.lower().map(
                    {"up": "#5BB56E", "down": "#E5534B"}).fillna("#4C9BE8")
                f.add_trace(go.Scatter(x=pt.index, y=pt["currentPriceTarget"], mode="markers", name="목표주가",
                                       marker=dict(size=8, color=colors), text=pt.get("Firm"),
                                       hovertemplate="%{x|%Y-%m-%d} %{text}<br>목표주가 %{y:,.2f}<extra></extra>"))
                med = pt["currentPriceTarget"].sort_index().rolling("90D").median()
                f.add_trace(go.Scatter(x=med.index, y=med.values, name="목표주가 90일 중앙값", line=dict(color="#F2A93B", width=2)))
                f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
                st.plotly_chart(f, use_container_width=True)
        with st.expander("투자의견·목표주가 변경 이력"):
            st.dataframe(show.head(80), use_container_width=True, hide_index=True)


def _render_event_iv(ticker: str, nxt, kind: str, rx: pd.DataFrame) -> None:
    from optvol import _chain_cached, _params, expiry_stats
    from options import load_expiries, years_to_expiry

    try:
        exps = load_expiries(ticker)
        P = _params(ticker)
    except Exception as e:  # noqa: BLE001
        st.caption(f"옵션 데이터를 불러오지 못했습니다: {e}")
        return
    d0 = pd.Timestamp(nxt).tz_localize(None).normalize() if pd.Timestamp(nxt).tzinfo else pd.Timestamp(nxt).normalize()
    eff = d0 + pd.offsets.BDay(1) if kind in ("amc", "unk") else d0  # 발표가 처음 가격에 반영되는 날
    now = pd.Timestamp.now()
    t_event = max((eff - now).total_seconds() / (365 * 86400), 0.0)
    if (eff - now).days > 120:
        st.caption("다음 실적 발표가 120일 넘게 남아 옵션으로 실적 효과를 분리하기 어렵습니다.")
        return
    pick = []
    before = [e for e in exps if pd.Timestamp(e) < eff]
    after = [e for e in exps if pd.Timestamp(e) >= eff]
    if before and (pd.Timestamp(before[-1]) - now).days >= 2:
        pick.append(before[-1])
    pick += after[:2]
    atm = []
    stats = {}
    for e in pick:
        try:
            d = _chain_cached(ticker, e, P["S"], P["r"], P["q"], P["sig_theo"])
            s = expiry_stats(d, P["S"], years_to_expiry(e))
            stats[e] = s
            atm.append((s["T"], s["atm_iv"]))
        except Exception:  # noqa: BLE001
            pass
    res = implied_event(sorted(atm), t_event)
    if not res or "move" not in res:
        st.caption("실적 발표 전후 만기의 옵션 IV가 부족해 계산하지 못했습니다.")
        return
    hist_abs = rx["다음날 반응"].abs().mean() if isinstance(rx, pd.DataFrame) and not rx.empty else np.nan
    e1 = after[0] if after else None
    straddle = stats.get(e1, {}).get("move_pct", np.nan) if e1 else np.nan
    mgrid([("실적 발표 예상 변동폭", f"±{res['abs_move'] * 100:.1f}%", f"평균 기준 · 1σ ±{res['move'] * 100:.1f}% · {res['how']}"),
           ("과거 평균 실적 반응", _p(hist_abs, 1), "다음날 절대 수익률 평균"),
           ("실적 포함 IV", _p(res["s1"]), f"{e1} 만기"),
           ("실적 후 예상 IV", _p(res["sb"]), "발표가 지나간 뒤 같은 만기에 남을 변동성"),
           ("기대 변동폭", f"±{straddle * 100:.1f}%" if np.isfinite(straddle) else "-", f"{e1} 만기 ATM 스트래들")], cols=5)
    if np.isfinite(hist_abs) and res["abs_move"] > 0:
        ratio = res["abs_move"] / hist_abs
        tone = "더 크게(옵션이 비싼 편)" if ratio > 1.15 else ("더 작게(옵션이 싼 편)" if ratio < 0.85 else "과거와 비슷하게")
        st.write(f"옵션이 예상하는 발표일 평균 움직임(±{res['abs_move'] * 100:.1f}%)이 과거 평균 실제 반응(±{hist_abs * 100:.1f}%)의 "
                 f"**{ratio:.2f}배** — 실적 이벤트를 {tone} 가격에 반영하고 있습니다. "
                 f"발표 뒤 IV는 약 {res['crush'] * 100:.1f}%p 떨어질 것으로 계산됩니다.")
    rows = [{"만기": e, "남은 일수": int(round(s["T"] * 365)), "ATM IV": _p(s["atm_iv"]), "실적 포함": "예" if pd.Timestamp(e) >= eff else "아니오",
             "기대 변동폭": f"±{s['move_pct'] * 100:.1f}%" if np.isfinite(s["move_pct"]) else "-"} for e, s in stats.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
