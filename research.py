"""기업 리서치 탭 (해외 종목): Yahoo Finance(yfinance)에서 기업 개요·밸류에이션·수익성·재무제표·애널리스트 의견·일정·뉴스를 모아 보여준다."""
import html
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from options import OPT_PRESETS, is_us_ticker, mgrid

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None

REC_KO = {"strong_buy": "적극 매수", "buy": "매수", "hold": "보유", "underperform": "비중 축소", "sell": "매도",
          "strongBuy": "적극 매수", "strongSell": "적극 매도", "none": "-"}


# ----------------------------------------------------------------------
# 데이터
# ----------------------------------------------------------------------
def _try(fn, tries=4, base=1.5):
    err = None
    for i in range(tries):
        try:
            out = fn()
            if out is not None:
                return out
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(base * (i + 1))
    if err:
        raise err
    return None


def _safe(fn, default=None):
    try:
        v = fn()
        return default if v is None else v
    except Exception:  # noqa: BLE001
        return default


@st.cache_data(ttl=60 * 60, show_spinner="기업 정보를 불러오는 중... (Yahoo 응답이 느리면 다시 시도합니다)")
def load_research(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = _try(lambda: t.info) or {}
    if not info or (not info.get("longName") and not info.get("shortName") and not info.get("quoteType")):
        raise RuntimeError("기업 정보를 찾지 못했습니다 (티커를 확인해 주세요)")
    out = {"info": info}
    out["income"] = _safe(lambda: t.income_stmt, pd.DataFrame())
    out["balance"] = _safe(lambda: t.balance_sheet, pd.DataFrame())
    out["cashflow"] = _safe(lambda: t.cashflow, pd.DataFrame())
    out["recs"] = _safe(lambda: t.recommendations, pd.DataFrame())
    out["targets"] = _safe(lambda: dict(t.analyst_price_targets), {})
    out["calendar"] = _safe(lambda: dict(t.calendar), {})
    out["news"] = _safe(lambda: list(t.news), [])
    out["qincome"] = _safe(lambda: t.quarterly_income_stmt, pd.DataFrame())
    out["eps_est"] = _safe(lambda: t.earnings_estimate, pd.DataFrame())
    out["edates"] = _safe(lambda: t.get_earnings_dates(limit=60), pd.DataFrame())
    out["hist"] = _safe(lambda: t.history(period="10y", auto_adjust=False)["Close"].dropna(), pd.Series(dtype=float))
    out["splits"] = _safe(lambda: t.splits, pd.Series(dtype=float))
    return out


# ----------------------------------------------------------------------
# SEC EDGAR (미국 공시) 10개년 재무제표 — 키 없이 무료, Yahoo는 4~5년치만 주므로 이것을 우선 사용
# ----------------------------------------------------------------------
SEC_UA = "jatbit-stock-app/1.0 (educational Streamlit app; https://jatbit-jatbit-app-qni3e3.streamlit.app)"
SEC_ITEMS = [  # (표시 이름, us-gaap 태그 우선순위, 단위, 기간형 여부)
    ("매출", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
             "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"], "USD", True),
    ("매출총이익", ["GrossProfit"], "USD", True),
    ("영업이익", ["OperatingIncomeLoss"], "USD", True),
    ("순이익", ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"], "USD", True),
    ("희석 EPS", ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"], "USD/shares", True),
    ("영업현금흐름", ["NetCashProvidedByUsedInOperatingActivities",
                  "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"], "USD", True),
    ("설비투자", ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"], "USD", True),
    ("총자산", ["Assets"], "USD", False),
    ("총부채", ["Liabilities"], "USD", False),
    ("자기자본", ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "USD", False),
    # 아래는 표에는 안 보이고 재무비율 계산에만 쓰는 항목
    ("유동자산", ["AssetsCurrent"], "USD", False),
    ("유동부채", ["LiabilitiesCurrent"], "USD", False),
    ("이자비용", ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"], "USD", True),
    ("희석주식수", ["WeightedAverageNumberOfDilutedSharesOutstanding"], "shares", True),
    ("배당금", ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"], "USD", True),
    # 신용위험(credit.py) 계산용
    ("이익잉여금", ["RetainedEarningsAccumulatedDeficit"], "USD", False),
    ("단기차입금", ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"], "USD", False),
    ("장기차입금", ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebt"], "USD", False),
    ("현금성자산", ["CashAndCashEquivalentsAtCarryingValue",
               "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"], "USD", False),
    ("감가상각비", ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
               "DepreciationAmortizationAndAccretionNet", "Depreciation"], "USD", True),
]
SEC_SHOW = ["매출", "매출총이익", "영업이익", "순이익", "희석 EPS", "영업현금흐름", "잉여현금흐름", "총자산", "총부채", "자기자본"]


def _sec_json(url: str):
    import json
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": SEC_UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _sec_cik_map() -> dict:
    data = _try(lambda: _sec_json("https://www.sec.gov/files/company_tickers.json"), tries=3)
    return {str(v["ticker"]).upper(): int(v["cik_str"]) for v in data.values()}


def _sec_series(facts: dict, tags: list, unit: str, duration: bool, with_filed: bool = False):
    """10-K에 실린 연간 값. 회계연도 말 날짜별로, 앞 순위 태그를 우선·같은 태그는 가장 최근 제출값(정정 반영)."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    out = {}
    for tag in tags:
        arr = gaap.get(tag, {}).get("units", {}).get(unit, [])
        for e in arr:
            if e.get("form") not in ("10-K", "10-K/A", "10-KT"):
                continue
            end, start = e.get("end"), e.get("start")
            if duration:
                if not start:
                    continue
                days = (pd.Timestamp(end) - pd.Timestamp(start)).days
                if not 340 <= days <= 380:
                    continue
            elif start:
                continue
            prev = out.get(end)
            if prev is None or (prev[0] == tag and e.get("filed", "") > prev[2]):
                out[end] = (tag, e.get("val"), e.get("filed", ""))
    if not out:
        return (pd.Series(dtype=float), pd.Series(dtype=float)) if with_filed else pd.Series(dtype=float)
    vals = pd.Series({pd.Timestamp(k): float(v[1]) for k, v in out.items()}).sort_index()
    if not with_filed:
        return vals
    filed = pd.Series({pd.Timestamp(k): float(pd.Timestamp(v[2] or k).toordinal()) for k, v in out.items()}).sort_index()
    return vals, filed


@st.cache_data(ttl=24 * 60 * 60, show_spinner="SEC 공시에서 10개년 재무제표를 불러오는 중...")
def load_sec_financials(ticker: str, years: int = 10) -> pd.DataFrame:
    cik = _sec_cik_map().get(ticker.upper()) or _sec_cik_map().get(ticker.upper().replace(".", "-"))
    if not cik:
        raise RuntimeError("SEC에 등록된 미국 공시 기업이 아닙니다")
    facts = _try(lambda: _sec_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"), tries=3)
    rows = {name: _sec_series(facts, tags, unit, dur) for name, tags, unit, dur in SEC_ITEMS}
    for name, tags, unit, dur in SEC_ITEMS:  # 주당 값·주식 수는 제출일을 같이 저장(액면분할 보정용)
        if name in ("희석 EPS", "희석주식수"):
            rows[name], rows["_filed_" + name] = _sec_series(facts, tags, unit, dur, with_filed=True)
    base = rows["매출"] if len(rows["매출"]) else rows["순이익"]
    if len(base) == 0:
        raise RuntimeError("연간 재무 데이터(us-gaap)가 없습니다")
    # 회계연도 말 = 손익계산서 기간 끝 날짜. 며칠 차이 나는 재무상태표 날짜는 가까운 연도 말에 맞춘다.
    ends = sorted(base.index)[-years:]
    df = pd.DataFrame(index=list(rows), columns=ends, dtype=float)
    for name, ser in rows.items():
        for d in ends:
            if d in ser.index:
                df.loc[name, d] = ser[d]
            elif len(ser):
                near = ser.index[abs((ser.index - d).days) <= 10]
                if len(near):
                    df.loc[name, d] = ser[near[-1]]
    if df.loc["설비투자"].notna().any():
        df.loc["잉여현금흐름"] = df.loc["영업현금흐름"] - df.loc["설비투자"].fillna(0)
    df = df.drop(index="설비투자")
    return df.dropna(how="all")


# ----------------------------------------------------------------------
# 재무비율 (SEC 10개년 + 회계연도 말 주가)
# ----------------------------------------------------------------------
RATIOS = {  # 이름: (단위, 분류)
    "매출총이익률": ("%", "수익성"), "영업이익률": ("%", "수익성"), "순이익률": ("%", "수익성"), "FCF 마진": ("%", "수익성"),
    "ROE": ("%", "수익성"), "ROA": ("%", "수익성"),
    "매출 성장률": ("%", "성장성"), "영업이익 성장률": ("%", "성장성"), "순이익 성장률": ("%", "성장성"), "EPS 성장률": ("%", "성장성"),
    "부채비율": ("%", "안정성"), "자기자본비율": ("%", "안정성"), "유동비율": ("%", "안정성"), "이자보상배율": ("배", "안정성"),
    "총자산회전율": ("회", "효율성"), "영업현금흐름/순이익": ("배", "현금흐름"), "배당성향": ("%", "현금흐름"),
    "PER": ("배", "밸류에이션"), "PBR": ("배", "밸류에이션"), "PSR": ("배", "밸류에이션"),
}


def _split_factor(filed_ord: pd.Series, splits) -> pd.Series:
    """제출일 이후에 있었던 액면분할 비율의 곱 (그 뒤 분할만큼 주당 값을 나누고 주식 수를 곱해야 오늘 기준이 됨)."""
    if splits is None or len(splits) == 0 or filed_ord is None:
        return pd.Series(1.0, index=filed_ord.index if filed_ord is not None else [])
    sp = splits.copy()
    sp.index = pd.to_datetime(sp.index)
    if sp.index.tz is not None:
        sp.index = sp.index.tz_localize(None)
    out = {}
    for d, f in filed_ord.items():
        if not np.isfinite(f):
            out[d] = 1.0
            continue
        fdate = pd.Timestamp.fromordinal(int(f))
        out[d] = float(np.prod(sp[sp.index > fdate].values)) if (sp.index > fdate).any() else 1.0
    return pd.Series(out)


def compute_ratios(sec: pd.DataFrame, hist: pd.Series, splits) -> pd.DataFrame:
    g = lambda k: sec.loc[k] if k in sec.index else pd.Series(np.nan, index=sec.columns)  # noqa: E731
    rev, gp, oi, ni = g("매출"), g("매출총이익"), g("영업이익"), g("순이익")
    ocf, fcf, assets, eq = g("영업현금흐름"), g("잉여현금흐름"), g("총자산"), g("자기자본")
    liab = g("총부채").fillna(assets - eq)
    avg = lambda x: ((x + x.shift(1)) / 2).fillna(x)  # noqa: E731  평균잔액(전년 말과 올해 말의 평균)
    growth = lambda x: (x - x.shift(1)) / x.shift(1).abs() * 100  # noqa: E731
    fac_e = _split_factor(g("_filed_희석 EPS"), splits).reindex(sec.columns).fillna(1.0)
    fac_s = _split_factor(g("_filed_희석주식수"), splits).reindex(sec.columns).fillna(1.0)
    eps = g("희석 EPS") / fac_e
    shares = g("희석주식수") * fac_s
    price = pd.Series(np.nan, index=sec.columns)
    if isinstance(hist, pd.Series) and len(hist):
        h = hist.copy()
        h.index = pd.to_datetime(h.index)
        if h.index.tz is not None:
            h.index = h.index.tz_localize(None)
        for d in sec.columns:
            before = h[h.index <= d]
            if len(before) and (d - before.index[-1]).days <= 10:
                price[d] = before.iloc[-1]
    mcap = price * shares
    r = {
        "매출총이익률": gp / rev * 100, "영업이익률": oi / rev * 100, "순이익률": ni / rev * 100, "FCF 마진": fcf / rev * 100,
        "ROE": ni / avg(eq) * 100, "ROA": ni / avg(assets) * 100,
        "매출 성장률": growth(rev), "영업이익 성장률": growth(oi), "순이익 성장률": growth(ni), "EPS 성장률": growth(eps),
        "부채비율": liab / eq * 100, "자기자본비율": eq / assets * 100, "유동비율": g("유동자산") / g("유동부채") * 100,
        "이자보상배율": oi / g("이자비용"), "총자산회전율": rev / avg(assets), "영업현금흐름/순이익": ocf / ni,
        "배당성향": g("배당금") / ni * 100,
        "PER": (mcap / ni).where(ni > 0), "PBR": (mcap / eq).where(eq > 0), "PSR": mcap / rev,
    }
    out = pd.DataFrame(r).T
    return out.replace([np.inf, -np.inf], np.nan)


DAILY = {  # 이름: 단위
    "주가": "$", "12M 트레일링 PER": "배", "12M 선행 PER (사후 실현 EPS)": "배",
    "현재 12M 선행 PER (컨센서스)": "배", "트레일링 PER 평균 ±1σ 밴드": "배",
}


def _combo_chart(sec, R: dict, ticker: str, px, n_eps) -> None:
    """연간 재무비율(SEC)과 일별 주가·PER 지표를 골라서 한 그래프에 겹쳐 그린다."""
    rt = compute_ratios(sec, R.get("hist"), R.get("splits")) if sec is not None else None
    avail = [k for k in RATIOS if rt is not None and k in rt.index and rt.loc[k].notna().sum() >= 2]
    per = per_series(R.get("hist"), R.get("edates"))
    hist = R.get("hist")
    d_avail = ["주가"] if isinstance(hist, pd.Series) and len(hist) else []
    if per is not None:
        d_avail += ["12M 트레일링 PER", "12M 선행 PER (사후 실현 EPS)", "트레일링 PER 평균 ±1σ 밴드"]
    if px and n_eps and n_eps > 0:
        d_avail.append("현재 12M 선행 PER (컨센서스)")

    c1, c2 = st.columns(2)
    sel = c1.multiselect("① 재무비율 (연간 · SEC 10년)", avail,
                         default=[k for k in ("순이익률", "순이익 성장률") if k in avail], key=f"rs_ratios_{ticker}",
                         format_func=lambda k: f"{k} ({RATIOS[k][1]}, {RATIOS[k][0]})")
    dsel = c2.multiselect("② 주가 · PER 지표 (일별 · 선행)", d_avail,
                          default=[k for k in ("주가",) if k in d_avail], key=f"rs_daily_{ticker}")
    if avail:
        groups = {}
        for k in avail:
            groups.setdefault(RATIOS[k][1], []).append(k)
        st.caption("① " + " · ".join(f"{g}: {', '.join(v)}" for g, v in groups.items()))
    elif sec is None:
        st.caption("① 재무비율은 SEC 10개년 재무제표가 있는 미국 기업만 지원합니다.")
    if not sel and not dsel:
        st.info("①이나 ②에서 그릴 항목을 하나 이상 고르세요.")
        return

    # 단위별 축: % → 왼쪽(y), 배·회 → 오른쪽(y2), 주가 → 오른쪽 바깥(y3)
    units = []
    for k in sel:
        units.append("%" if RATIOS[k][0] == "%" else "x")
    for k in dsel:
        units.append("$" if DAILY[k] == "$" else "x")
    order = [u for u in ("%", "x", "$") if u in units]
    axis_of = {u: ("y" if i == 0 else f"y{i + 1}") for i, u in enumerate(order)}
    title_of = {"%": "%", "x": "배·회", "$": "주가($)"}

    # 일별 지표 시작일을 재무비율 기간에 맞춘다
    start = None
    if rt is not None and sel:
        start = pd.Timestamp(rt.columns[0]) - pd.DateOffset(months=6)
    palette = ["#4C9BE8", "#F2A93B", "#E5534B", "#5BB56E", "#9B6FD6", "#2BB3B1", "#D97AB0", "#8C8C8C", "#C9A227", "#3D7DCA"]
    fig = go.Figure()
    ci = 0
    for k in sel:
        unit = RATIOS[k][0]
        ax = axis_of["%" if unit == "%" else "x"]
        fig.add_trace(go.Scatter(x=list(rt.columns), y=rt.loc[k].values, name=f"{k} ({unit})", mode="lines+markers", yaxis=ax,
                                 line=dict(color=palette[ci % len(palette)], width=2), marker=dict(size=8),
                                 hovertemplate="%{x|%Y-%m} " + k + (" %{y:.1f}%" if unit == "%" else " %{y:.2f}" + unit) + "<extra></extra>"))
        ci += 1

    def _clip(sr):
        sr = sr.copy()
        sr.index = pd.to_datetime(sr.index)
        if sr.index.tz is not None:
            sr.index = sr.index.tz_localize(None)
        return sr[sr.index >= start] if start is not None else sr

    trail = fwd_r = None
    if per is not None:
        trail, fwd_r, _e = per
        trail, fwd_r = _clip(trail), _clip(fwd_r)
    for k in dsel:
        ax = axis_of["$" if DAILY[k] == "$" else "x"]
        color = palette[ci % len(palette)]
        if k == "주가":
            h = _clip(hist)
            fig.add_trace(go.Scatter(x=h.index, y=h.values, name="주가", mode="lines", yaxis=ax, line=dict(color=color, width=1.4),
                                     hovertemplate="%{x|%Y-%m-%d} 주가 %{y:,.2f}<extra></extra>"))
        elif k == "12M 트레일링 PER":
            fig.add_trace(go.Scatter(x=trail.index, y=trail.values, name=k, mode="lines", yaxis=ax, line=dict(color=color, width=1.4),
                                     hovertemplate="%{x|%Y-%m-%d} 트레일링 PER %{y:.1f}배<extra></extra>"))
        elif k == "12M 선행 PER (사후 실현 EPS)":
            fig.add_trace(go.Scatter(x=fwd_r.index, y=fwd_r.values, name=k, mode="lines", yaxis=ax,
                                     line=dict(color=color, width=1.4, dash="dash"),
                                     hovertemplate="%{x|%Y-%m-%d} 선행 PER(사후) %{y:.1f}배<extra></extra>"))
        elif k == "트레일링 PER 평균 ±1σ 밴드":
            tv = trail.dropna()
            if len(tv):
                m, sd = float(tv.mean()), float(tv.std())
                x0, x1 = tv.index[0], tv.index[-1]
                for yv, nm, dash in ((m + sd, f"평균+1σ {m + sd:.1f}배", "dot"), (m, f"PER 평균 {m:.1f}배", "dash"),
                                     (m - sd, f"평균−1σ {m - sd:.1f}배", "dot")):
                    fig.add_trace(go.Scatter(x=[x0, x1], y=[yv, yv], name=nm, mode="lines", yaxis=ax,
                                             line=dict(color="#888", width=1, dash=dash), hoverinfo="skip"))
        elif k == "현재 12M 선행 PER (컨센서스)":
            now = pd.Timestamp(hist.index[-1]).tz_localize(None) if getattr(hist.index, "tz", None) is not None else pd.Timestamp(hist.index[-1])
            fig.add_trace(go.Scatter(x=[now], y=[px / n_eps], name=f"현재 12M 선행 PER {px / n_eps:.1f}배", mode="markers", yaxis=ax,
                                     marker=dict(size=14, symbol="star", color="#E5534B"),
                                     hovertemplate=f"현재 12M 선행(NTM) PER {px / n_eps:.1f}배<extra></extra>"))
        ci += 1

    multi = len(order)
    lay = dict(height=420, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
               legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
               xaxis=dict(domain=[0, 0.86] if multi == 3 else [0, 1]))
    for i, u in enumerate(order):
        key = "yaxis" if i == 0 else f"yaxis{i + 1}"
        spec = dict(title=title_of[u], ticksuffix="%" if u == "%" else "", showgrid=(i == 0))
        if i == 1:
            spec.update(overlaying="y", side="right", anchor="x")
        elif i == 2:
            spec.update(overlaying="y", side="right", anchor="free", position=1.0)
        if u == "x" and dsel and any(k in dsel for k in ("12M 트레일링 PER", "12M 선행 PER (사후 실현 EPS)")) and trail is not None:
            vals = np.concatenate([trail.dropna().values, fwd_r.dropna().values] +
                                  [rt.loc[k].dropna().values for k in sel if RATIOS[k][0] != "%"])
            if len(vals):
                spec["range"] = [0, float(np.nanpercentile(vals, 98)) * 1.2]
        lay[key] = spec
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True)
    axes_txt = ", ".join(f"{title_of[u]} → {'왼쪽' if i == 0 else ('오른쪽' if i == 1 else '오른쪽 바깥')} 축" for i, u in enumerate(order))
    st.caption(f"축: {axes_txt}. 재무비율은 회계연도 말 시점의 점(SEC 10-K), 주가·PER은 일별 선입니다.")

    if sel:
        tbl = rt.loc[sel].copy()
        cols = [pd.Timestamp(c).strftime("%Y-%m") for c in rt.columns]
        tbl = tbl.apply(lambda row: [("-" if pd.isna(v) else (f"{v:.1f}%" if RATIOS[row.name][0] == "%" else f"{v:.2f}"))
                                     for v in row], axis=1, result_type="expand")
        tbl.columns = cols
        st.dataframe(tbl, use_container_width=True)
    notes = ["재무비율: ROE·ROA·총자산회전율은 전년 말과 올해 말의 평균 잔액 기준, 성장률은 전년 대비, "
             "PER·PBR·PSR은 회계연도 말 주가 × 희석 주식 수(액면분할 보정)로 계산. 회사가 공시하지 않은 항목은 빈칸입니다."]
    if any(k in dsel for k in ("12M 트레일링 PER", "12M 선행 PER (사후 실현 EPS)", "트레일링 PER 평균 ±1σ 밴드")):
        notes.append("12M 트레일링 PER = 주가 ÷ 그날까지 발표된 최근 4개 분기 EPS 합. 과거의 선행 PER 컨센서스는 무료로 받을 수 없어, "
                     "'선행 PER (사후 실현 EPS)'는 그 뒤 실제로 발표된 4개 분기 EPS로 계산한 값입니다(최근 1년은 계산 불가). "
                     "분기 EPS는 Yahoo의 발표 EPS(액면분할 반영)라 공시 순이익 기준 EPS와 다를 수 있습니다.")
    if "현재 12M 선행 PER (컨센서스)" in dsel:
        notes.append("별 표시는 오늘의 애널리스트 컨센서스로 계산한 12M 선행(NTM) PER입니다.")
    st.caption(" ".join(notes))



@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner="한국어로 번역하는 중...")
def translate_ko(text: str):
    """Google 번역(키 없는 공개 엔드포인트)으로 한국어 번역. 실패하면 None."""
    import json
    import urllib.parse
    import urllib.request

    if not text or not text.strip():
        return None
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=ko&dt=t"
    for i in range(3):
        try:
            data = urllib.parse.urlencode({"q": text}).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                j = json.loads(resp.read().decode("utf-8"))
            out = "".join(seg[0] for seg in j[0] if seg and seg[0])
            if out.strip():
                return out
        except Exception:  # noqa: BLE001
            time.sleep(1.0 + i)
    return None


# ----------------------------------------------------------------------
# 표시 도우미
# ----------------------------------------------------------------------
def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:  # noqa: BLE001
        return None


def big(x, cur="$"):
    v = _num(x)
    if v is None:
        return "-"
    a = abs(v)
    for unit, d in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if a >= d:
            return f"{'-' if v < 0 else ''}{cur}{a / d:,.2f}{unit}"
    return f"{'-' if v < 0 else ''}{cur}{a:,.2f}"


def pct(x, nd=1):
    v = _num(x)
    return "-" if v is None else f"{v * 100:.{nd}f}%"


def ratio(x, nd=2):
    v = _num(x)
    return "-" if v is None else f"{v:,.{nd}f}"


def _row(df: pd.DataFrame, *names):
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def ntm_eps(info: dict, est: pd.DataFrame):
    """12개월 선행(NTM) EPS = 이번 회계연도(0y)와 다음 회계연도(+1y) 컨센서스를 남은 기간 비율로 가중평균."""
    try:
        e0, e1 = float(est.loc["0y", "avg"]), float(est.loc["+1y", "avg"])
        fye = _num(info.get("nextFiscalYearEnd")) or _num(info.get("lastFiscalYearEnd"))
        if not fye:
            return None, None
        end = pd.Timestamp(int(fye), unit="s")
        while end < pd.Timestamp.now():
            end += pd.DateOffset(years=1)
        w = min(max((end - pd.Timestamp.now()).days / 365.0, 0.0), 1.0)
        return w * e0 + (1 - w) * e1, (e0, e1, w, end)
    except Exception:  # noqa: BLE001
        return None, None


def _div_yield(info: dict):
    px = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    rate = _num(info.get("dividendRate"))
    if px and rate:
        return rate / px
    y = _num(info.get("trailingAnnualDividendYield"))
    return y


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
def per_series(hist: pd.Series, edates: pd.DataFrame):
    """분기 실적(발표일·발표 EPS)으로 일별 12M PER 두 가지를 만든다.
    트레일링: 주가 ÷ 그날까지 발표된 최근 4개 분기 EPS 합.
    선행(사후): 주가 ÷ 그날 이후 발표된 4개 분기 EPS 합 (실제로 나온 이익 기준이라 최근 1년은 계산 불가)."""
    if not isinstance(edates, pd.DataFrame) or edates.empty or "Reported EPS" not in edates or not len(hist):
        return None
    e = edates[["Reported EPS"]].dropna().copy()
    idx = pd.to_datetime(e.index)
    e.index = idx.tz_localize(None) if idx.tz is not None else idx
    e = e[~e.index.duplicated()].sort_index()["Reported EPS"].astype(float)
    if len(e) < 4:
        return None
    h = hist.copy()
    h.index = pd.to_datetime(h.index)
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    h = h[h.index >= e.index[3]]
    ttm = e.rolling(4).sum()                       # 발표일 기준 최근 4분기 합
    fwd = e[::-1].rolling(4).sum()[::-1].shift(-1)  # 다음 발표부터 4분기 합
    pos = np.searchsorted(e.index.values, h.index.values, side="right") - 1
    t_eps = np.where(pos >= 0, ttm.values[np.clip(pos, 0, None)], np.nan)
    f_eps = np.where(pos >= 0, fwd.values[np.clip(pos, 0, None)], np.nan)
    trail = pd.Series(np.where(t_eps > 0, h.values / t_eps, np.nan), index=h.index)
    fwd_r = pd.Series(np.where(f_eps > 0, h.values / f_eps, np.nan), index=h.index)
    return trail, fwd_r, e


def _set_research(tk):
    st.session_state["rs_ticker"] = tk


def _send_to_options(tk):
    st.session_state["opt_ticker"] = tk


def render_research() -> None:
    if yf is None:
        st.error("yfinance 패키지가 설치되지 않았습니다.")
        return
    st.caption("미국 상장 종목·ETF의 기업 정보를 Yahoo Finance에서 모아 보여줍니다 (일부 항목은 종목에 따라 비어 있을 수 있음). 국내 종목은 다음 단계에서 지원 예정입니다.")

    res = st.session_state.get("pf_opt_result") or {}
    pf = [it["code"] for it in res.get("items", []) if not it.get("domestic") and is_us_ticker(it["code"])]
    presets = list(dict.fromkeys(pf + OPT_PRESETS))[:10]
    if hasattr(st, "pills"):
        st.pills("빠른 선택", presets, key="rs_pill",
                 on_change=lambda: _set_research(st.session_state.get("rs_pill")) if st.session_state.get("rs_pill") else None)
    raw = st.text_input("미국 티커 (예: AAPL, MSFT, NVDA)", key="rs_ticker", placeholder="AAPL")
    ticker = (raw or "").strip().upper()
    if not ticker:
        st.info("티커를 입력하거나 위에서 고르세요.")
        return
    if not is_us_ticker(ticker):
        st.warning("지금은 미국 종목만 지원합니다. 미국 티커를 입력해 주세요.")
        return
    try:
        R = load_research(ticker)
    except Exception as e:  # noqa: BLE001
        st.error(f"{ticker} 정보를 불러오지 못했습니다: {e}. 잠시 후 다시 시도해 주세요.")
        return
    info = R["info"]
    cur = "$" if (info.get("currency") or "USD") == "USD" else ""

    # ---- 1. 개요
    name = info.get("longName") or info.get("shortName") or ticker
    st.subheader(f"{name} ({ticker})")
    meta = [x for x in [info.get("sector"), info.get("industry"), info.get("country"),
                        f"직원 {int(info['fullTimeEmployees']):,}명" if _num(info.get("fullTimeEmployees")) else None,
                        info.get("exchange")] if x]
    st.caption(" · ".join(map(str, meta)) + (f" · [홈페이지]({info['website']})" if info.get("website") else ""))
    ko = st.toggle("한국어 번역 (Google 자동 번역)", value=True, key="rs_ko")
    summary = info.get("longBusinessSummary")
    if summary:
        tr = translate_ko(summary) if ko else None
        with st.expander("사업 설명", expanded=True):
            if tr:
                st.write(tr.replace("$", "\\$"))
                st.caption("자동 번역이라 고유명사·전문용어가 어색할 수 있습니다.")
            elif ko:
                st.caption("번역에 실패해 영문 원문을 보여줍니다.")
            if not tr:
                st.write(summary.replace("$", "\\$"))
        if tr:
            with st.expander("사업 설명 (영문 원문)"):
                st.write(summary.replace("$", "\\$"))
    st.button("이 종목을 '옵션 조회' 탭에 설정", key="rs_to_opt", on_click=_send_to_options, args=(ticker,))
    from news import render_latest_news
    render_latest_news(ticker, info.get("shortName") or name, R.get("news") or [], ko, translate_ko)

    tab_main, tab_dcf, tab_cr, tab_ins = st.tabs(["개요·재무", "DCF 가치평가", "신용위험·회사채", "내부자 거래·지분 구조"])
    _sec = None
    try:
        _sec = load_sec_financials(ticker)
    except Exception:  # noqa: BLE001
        _sec = None
    with tab_dcf:
        from dcf import render_dcf
        render_dcf(ticker, R, _sec)
    with tab_cr:
        from credit import render_credit
        render_credit(ticker, R, _sec)
    with tab_ins:
        from insider import render_insider
        render_insider(ticker, R)
    with tab_main:
        _render_main(ticker, R, info, cur, ko)


def _render_main(ticker: str, R: dict, info: dict, cur: str, ko: bool) -> None:
    # ---- 2. 주가·밸류에이션
    px = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    lo52, hi52 = _num(info.get("fiftyTwoWeekLow")), _num(info.get("fiftyTwoWeekHigh"))
    pos52 = (px - lo52) / (hi52 - lo52) if px and lo52 and hi52 and hi52 > lo52 else None
    st.markdown("##### 주가 · 밸류에이션")
    n_eps, n_info = ntm_eps(info, R.get("eps_est"))
    fwd_eps = _num(info.get("forwardEps"))
    fwd_note = "Yahoo 'Forward P/E' = 현재가 ÷ forwardEps"
    if n_info:
        e0, e1, w, fy_end = n_info
        which = min(((abs((fwd_eps or 0) - e0), "이번 회계연도"), (abs((fwd_eps or 0) - e1), "다음 회계연도")))[1] if fwd_eps else "?"
        fwd_note += f" ({fwd_eps:.2f}, {which} 추정치 기준)" if fwd_eps else ""
    mgrid([("현재가", f"{cur}{px:,.2f}" if px else "-"),
           ("시가총액", big(info.get("marketCap"), cur)),
           ("기업가치(EV)", big(info.get("enterpriseValue"), cur)),
           ("52주 범위", f"{lo52:,.2f} ~ {hi52:,.2f}" if lo52 and hi52 else "-",
            f"현재가는 52주 범위의 {pos52 * 100:.0f}% 위치" if pos52 is not None else ""),
           ("PER (최근 12개월)", ratio(info.get("trailingPE"), 1)),
           ("선행 PER (Yahoo)", ratio(info.get("forwardPE"), 1), fwd_note),
           ("12M 선행 PER (NTM)", ratio(px / n_eps, 1) if px and n_eps and n_eps > 0 else "-",
            f"현재가 ÷ 향후 12개월 EPS {n_eps:.2f} (이번 회계연도 {n_info[0]:.2f} × {n_info[2]:.2f} + 다음 회계연도 {n_info[1]:.2f} × {1 - n_info[2]:.2f})" if n_eps else "추정치 없음"),
           ("PBR", ratio(info.get("priceToBook"))),
           ("EV/EBITDA", ratio(info.get("enterpriseToEbitda"), 1)),
           ("PEG", ratio(info.get("trailingPegRatio") or info.get("pegRatio"))),
           ("PSR", ratio(info.get("priceToSalesTrailing12Months"))),
           ("배당수익률", pct(_div_yield(info), 2)),
           ("베타", ratio(info.get("beta")))])
    if n_info:
        st.caption(f"선행 PER(Yahoo)은 현재가 ÷ Yahoo forwardEps({fwd_eps:.2f})이며, 이 값은 {which} EPS 컨센서스"
                   f"({n_info[0] if which == '이번 회계연도' else n_info[1]:.2f})에 가장 가깝습니다. 즉 '향후 12개월'이 아니라 회계연도 기준입니다. 12M 선행 PER(NTM)은 이번 회계연도({n_info[3]:%Y-%m} 종료) 남은 비중 {n_info[2]:.0%}만큼 "
                   f"이번 회계연도 EPS {n_info[0]:.2f}, 나머지는 다음 회계연도 EPS {n_info[1]:.2f}를 섞어 향후 12개월 EPS {n_eps:.2f}로 계산했습니다."
                   if fwd_eps else "12M 선행 PER(NTM)은 이번·다음 회계연도 EPS 컨센서스를 남은 기간 비율로 섞은 향후 12개월 EPS 기준입니다.")

    # ---- 3. 수익성·성장·재무건전성
    st.markdown("##### 수익성 · 성장 · 재무건전성")
    de = _num(info.get("debtToEquity"))
    mgrid([("매출 성장률(YoY)", pct(info.get("revenueGrowth"))),
           ("이익 성장률(YoY)", pct(info.get("earningsGrowth"))),
           ("매출총이익률", pct(info.get("grossMargins"))),
           ("영업이익률", pct(info.get("operatingMargins"))),
           ("순이익률", pct(info.get("profitMargins"))),
           ("ROE", pct(info.get("returnOnEquity"))),
           ("ROA", pct(info.get("returnOnAssets"))),
           ("부채비율(D/E)", f"{de:.0f}%" if de is not None else "-", "총부채 ÷ 자기자본 (Yahoo는 % 단위로 제공)"),
           ("현금성 자산", big(info.get("totalCash"), cur)),
           ("총차입금", big(info.get("totalDebt"), cur)),
           ("잉여현금흐름(FCF)", big(info.get("freeCashflow"), cur)),
           ("유동비율", ratio(info.get("currentRatio")))])

    # ---- 4. 애널리스트
    st.markdown("##### 애널리스트 의견")
    tg = R["targets"] or {}
    mean_t = _num(tg.get("mean")) or _num(info.get("targetMeanPrice"))
    up = (mean_t / px - 1) if mean_t and px else None
    rec = info.get("recommendationKey") or "none"
    mgrid([("목표주가 평균", f"{cur}{mean_t:,.2f}" if mean_t else "-"),
           ("상승 여력", pct(up), "목표주가 평균 ÷ 현재가 − 1"),
           ("목표주가 범위", f"{_num(tg.get('low')) or _num(info.get('targetLowPrice')) or 0:,.0f} ~ "
                         f"{_num(tg.get('high')) or _num(info.get('targetHighPrice')) or 0:,.0f}" if mean_t else "-"),
           ("투자의견", f"{REC_KO.get(rec, rec)} ({ratio(info.get('recommendationMean'), 2)})",
            "1=적극 매수 … 5=매도"),
           ("의견 수", f"{int(info['numberOfAnalystOpinions'])}명" if _num(info.get("numberOfAnalystOpinions")) else "-")])
    recs = R["recs"]
    if isinstance(recs, pd.DataFrame) and not recs.empty and {"strongBuy", "buy", "hold", "sell", "strongSell"} <= set(recs.columns):
        r = recs.copy()
        labels = {"0m": "이번 달", "-1m": "1개월 전", "-2m": "2개월 전", "-3m": "3개월 전"}
        r["기간"] = r["period"].map(lambda p: labels.get(p, p)) if "period" in r else [str(i) for i in range(len(r))]
        r = r.iloc[::-1]
        fig = go.Figure()
        for col, nm, color in [("strongBuy", "적극 매수", "#1B7F3B"), ("buy", "매수", "#5BB56E"), ("hold", "보유", "#B0B4BA"),
                               ("sell", "매도", "#E8875B"), ("strongSell", "적극 매도", "#C93C3C")]:
            fig.add_trace(go.Bar(y=r["기간"], x=r[col], name=nm, orientation="h", marker_color=color,
                                 hovertemplate=nm + " %{x}명<extra></extra>"))
        fig.update_layout(barmode="stack", height=200, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), xaxis_title="애널리스트 수")
        st.plotly_chart(fig, use_container_width=True)

    # ---- 5. 연간 재무제표
    inc, bal, cf = R["income"], R["balance"], R["cashflow"]
    sec, sec_err = None, "데이터 부족"
    try:
        sec = load_sec_financials(ticker)
        if sec.shape[1] < 2:
            sec = None
    except Exception as e:  # noqa: BLE001
        sec_err = str(e)
    if sec is not None:
        st.markdown(f"##### 연간 재무제표 (최근 {sec.shape[1]}개 회계연도 · SEC 공시)")
    else:
        st.markdown("##### 연간 재무제표 (최근 4개 회계연도 · Yahoo)")
        st.caption(f"SEC 10개년 데이터를 쓰지 못해 Yahoo 데이터로 대신합니다 ({sec_err}).")
    items = [
        ("매출", _row(inc, "Total Revenue", "Operating Revenue")),
        ("매출총이익", _row(inc, "Gross Profit")),
        ("영업이익", _row(inc, "Operating Income")),
        ("EBITDA", _row(inc, "EBITDA", "Normalized EBITDA")),
        ("순이익", _row(inc, "Net Income", "Net Income Common Stockholders")),
        ("희석 EPS", _row(inc, "Diluted EPS", "Basic EPS")),
        ("영업현금흐름", _row(cf, "Operating Cash Flow")),
        ("잉여현금흐름", _row(cf, "Free Cash Flow")),
        ("총자산", _row(bal, "Total Assets")),
        ("총부채", _row(bal, "Total Liabilities Net Minority Interest")),
        ("자기자본", _row(bal, "Stockholders Equity", "Common Stock Equity")),
    ]
    items = [(k, v) for k, v in items if v is not None]
    if sec is not None or items:
        if sec is not None:
            fs = sec.loc[[r for r in SEC_SHOW if r in sec.index]].copy()
        else:
            fs = pd.DataFrame({k: pd.to_numeric(v, errors="coerce") for k, v in items}).T
            fs = fs.dropna(axis=1, how="all")
            fs = fs[sorted(fs.columns)[-4:]]
        fs.columns = [pd.Timestamp(c).strftime("%Y-%m") for c in fs.columns]
        show = fs.copy().astype(object)
        for k in show.index:
            show.loc[k] = [ratio(v) if k == "희석 EPS" else big(v, cur) for v in fs.loc[k]]
        if "매출" in fs.index:
            rev = fs.loc["매출"]
            show.loc["매출 성장률"] = [pct(v) for v in rev.pct_change()]
            for k, lab in (("영업이익", "영업이익률"), ("순이익", "순이익률")):
                if k in fs.index:
                    show.loc[lab] = [pct(a / b) if b else "-" for a, b in zip(fs.loc[k], rev)]
        st.dataframe(show, use_container_width=True)
        if sec is not None:
            st.caption("출처: SEC EDGAR XBRL(10-K). 회사가 공시한 원 수치이며, 회계기준 변경·정정 공시가 있으면 가장 최근 제출값을 씁니다. "
                       "잉여현금흐름 = 영업현금흐름 − 유형자산 취득(설비투자). 표가 넓으면 옆으로 밀어서 보세요.")
        if "매출" in fs.index:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=fs.columns, y=fs.loc["매출"], name="매출", marker_color="#4C9BE8",
                                 hovertemplate="%{x} 매출 %{y:$,.0f}<extra></extra>"))
            if "순이익" in fs.index:
                fig.add_trace(go.Bar(x=fs.columns, y=fs.loc["순이익"], name="순이익", marker_color="#F2A93B",
                                     hovertemplate="%{x} 순이익 %{y:$,.0f}<extra></extra>"))
            if "영업이익" in fs.index:
                fig.add_trace(go.Scatter(x=fs.columns, y=(fs.loc["영업이익"] / fs.loc["매출"] * 100), name="영업이익률(%)",
                                         yaxis="y2", mode="lines+markers", line=dict(color="#E5534B"),
                                         hovertemplate="%{x} 영업이익률 %{y:.1f}%<extra></extra>"))
            fig.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                              yaxis=dict(title="금액"), yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("재무제표를 불러오지 못했습니다 (ETF이거나 Yahoo 응답 지연).")

    # ---- 5-1. 재무비율·주가·PER 그래프 (직접 선택)
    st.markdown("##### 재무비율 · 주가 · PER 그래프 (직접 선택해서 겹쳐 보기)")
    _combo_chart(sec, R, ticker, px, n_eps)


    # ---- 6. 일정
    cal = R["calendar"] or {}
    ev = []
    ed = cal.get("Earnings Date")
    if ed:
        ed = ed if isinstance(ed, (list, tuple)) else [ed]
        ev.append(("다음 실적 발표", " ~ ".join(str(pd.Timestamp(d).date()) for d in ed[:2])))
    if cal.get("Ex-Dividend Date"):
        ev.append(("배당락일", str(pd.Timestamp(cal["Ex-Dividend Date"]).date())))
    if cal.get("Dividend Date"):
        ev.append(("배당 지급일", str(pd.Timestamp(cal["Dividend Date"]).date())))
    if _num(cal.get("Earnings Average")):
        ev.append(("다음 분기 EPS 예상", ratio(cal.get("Earnings Average"))))
    if ev:
        st.markdown("##### 주요 일정")
        mgrid(ev)

    # ---- 7. 뉴스
    news = R["news"] or []
    rows = []
    for n in news[:12]:
        c = n.get("content") if isinstance(n, dict) and isinstance(n.get("content"), dict) else n
        if not isinstance(c, dict):
            continue
        title = c.get("title")
        url = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else None
        url = url or (c.get("clickThroughUrl") or {}).get("url") if isinstance(c.get("clickThroughUrl"), dict) else url
        url = url or c.get("link")
        prov = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else c.get("publisher")
        when = c.get("pubDate") or c.get("displayTime")
        if not when and c.get("providerPublishTime"):
            when = datetime.utcfromtimestamp(int(c["providerPublishTime"])).isoformat()
        if title:
            rows.append((title, url, prov, str(when)[:10] if when else ""))
    if rows:
        st.markdown("##### 최근 뉴스")
        kos = None
        if ko:
            joined = translate_ko("\n".join(r[0].replace("\n", " ") for r in rows))
            kos = joined.split("\n") if joined else None
            if kos is not None and len(kos) != len(rows):
                kos = None
        lines = []
        for i, (title, url, prov, when) in enumerate(rows):
            main = kos[i].strip() if kos and kos[i].strip() else title
            t = html.escape(main)
            link = f'<a href="{html.escape(url, quote=True)}" target="_blank">{t}</a>' if url else t
            orig = f'<br><span style="font-size:0.8rem;opacity:.55">{html.escape(title)}</span>' if main != title else ""
            lines.append(f'<div style="margin:0 0 10px">{link}{orig}<br><span style="font-size:0.8rem;opacity:.65">'
                         f'{html.escape(prov or "")} · {when}</span></div>')
        st.markdown("".join(lines).replace("$", "&#36;"), unsafe_allow_html=True)
    st.caption("출처: Yahoo Finance. 지연·누락이 있을 수 있으며 투자 판단의 참고 자료로만 사용하세요.")
