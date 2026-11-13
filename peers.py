"""동종업계 비교 (기업 리서치 탭)
- 1순위: 증권사 리포트·업종 분석에서 실제로 같이 묶어 보는 경쟁사 그룹 (아래 PEER_GROUPS_KR / PEER_GROUPS_US)
- 2순위: 국내는 WICS 같은 업종, 미국은 S&P 500 GICS 같은 세부 업종에서 시가총액이 비슷한 기업
- 3순위: Yahoo 산업 분류 대표 기업
- 밸류에이션·수익성·성장·재무 지표를 한 표로 비교하고, 업계 안에서 이 회사가 몇 등(백분위)인지 보여준다.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import money
from glossary import CSS, icon, tip

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None

# 열 이름: (Yahoo info 키, 형식, 높을수록 좋은지 — True 좋음 / False 나쁨 / None 중립)
METRICS = {
    "PER": ("trailingPE", "x", False), "선행 PER": ("forwardPE", "x", False), "PBR": ("priceToBook", "x", False),
    "PSR": ("priceToSalesTrailing12Months", "x", False), "EV/EBITDA": ("enterpriseToEbitda", "x", False),
    "매출 성장률": ("revenueGrowth", "%", True), "이익 성장률": ("earningsGrowth", "%", True),
    "매출총이익률": ("grossMargins", "%", True), "영업이익률": ("operatingMargins", "%", True), "순이익률": ("profitMargins", "%", True),
    "ROE": ("returnOnEquity", "%", True), "ROA": ("returnOnAssets", "%", True),
    "부채비율": ("debtToEquity", "d", False), "배당수익률": ("dividendYield", "y", None), "베타": ("beta", "r", None),
    "1년 수익률": ("52WeekChange", "%", True),
}
GOOD_LOW_SKIP_NEG = {"PER", "선행 PER", "PBR", "PSR", "EV/EBITDA"}  # 음수(적자)는 순위에서 뺌


# 시장에서 함께 비교하는 경쟁사 그룹 (증권사 산업 리포트·업종 분석 기준, 2026년). 종목코드 기준.
PEER_GROUPS_KR = {
    "방산": ["012450", "064350", "079550", "047810", "272210", "103140"],
    "조선": ["009540", "329180", "042660", "010140", "443060"],
    "조선 엔진·기자재": ["082740", "071970", "017960", "033500", "077970"],
    "전력기기·전선": ["267260", "298040", "010120", "062040", "103590", "033100", "006260", "001440"],
    "원전": ["034020", "052690", "051600", "083650", "032820", "015760"],
    "2차전지 셀": ["373220", "006400", "096770"],
    "2차전지 소재": ["247540", "086520", "003670", "066970", "348370", "005070", "450080"],
    "메모리 반도체": ["005930", "000660"],
    "반도체 장비·부품": ["042700", "036930", "240810", "039030", "403870", "058470", "095340", "089030", "319660", "357780"],
    "인터넷 플랫폼": ["035420", "035720", "377300", "323410"],
    "게임": ["259960", "251270", "036570", "263750", "462870", "293490", "225570"],
    "엔터테인먼트": ["352820", "035900", "041510", "122870"],
    "화장품": ["278470", "090430", "051900", "192820", "161890", "257720", "483650", "241710"],
    "바이오·제약": ["207940", "068270", "196170", "000100", "128940", "141080", "028300", "326030"],
    "자동차": ["005380", "000270", "012330", "204320", "011210", "018880"],
    "은행": ["105560", "055550", "086790", "316140", "024110", "138930", "175330"],
    "증권": ["006800", "071050", "005940", "016360", "039490", "138040"],
    "보험": ["032830", "000810", "005830", "088350", "001450"],
    "통신": ["017670", "030200", "032640"],
    "정유": ["096770", "010950", "078930", "267250"],
    "철강·비철": ["005490", "004020", "010130", "001230", "460860"],
    "화학": ["051910", "011170", "009830", "011780", "011790"],
    "건설": ["000720", "028050", "006360", "047040", "375500", "294870"],
    "항공": ["003490", "089590", "272450", "091810"],
    "해운": ["011200", "028670", "005880"],
    "유통": ["139480", "023530", "004170", "069960", "282330", "007070"],
    "음식료": ["003230", "271560", "004370", "097950", "007310", "280360"],
    "지주회사": ["034730", "003550", "028260", "000880", "001040", "000150"],
}
PEER_GROUPS_US = {
    "AI·GPU 반도체": ["NVDA", "AMD", "AVGO", "INTC", "QCOM", "MRVL", "ARM", "TSM"],
    "메모리·스토리지": ["MU", "WDC", "STX", "SNDK"],
    "반도체 장비": ["AMAT", "LRCX", "KLAC", "ASML", "TER"],
    "클라우드·기업용 소프트웨어": ["ORCL", "CRM", "NOW", "ADBE", "SAP", "INTU", "WDAY", "SNOW"],
    "데이터·AI 소프트웨어": ["PLTR", "SNOW", "DDOG", "MDB", "AI"],
    "사이버보안": ["CRWD", "PANW", "FTNT", "ZS", "NET", "OKTA"],
    "인터넷 광고·SNS": ["GOOGL", "META", "PINS", "SNAP", "RDDT"],
    "이커머스": ["AMZN", "SHOP", "MELI", "PDD", "BABA", "EBAY"],
    "전기차·자동차": ["TSLA", "GM", "F", "RIVN", "TM", "STLA"],
    "스트리밍·미디어": ["NFLX", "DIS", "WBD", "SPOT", "ROKU"],
    "결제·카드": ["V", "MA", "PYPL", "AXP", "FI", "XYZ"],
    "대형 은행·IB": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "가상자산·증권 플랫폼": ["COIN", "HOOD", "MSTR", "SCHW", "IBKR"],
    "제약": ["LLY", "NVO", "JNJ", "MRK", "PFE", "ABBV", "BMY", "AZN"],
    "의료기기": ["ABT", "MDT", "SYK", "BSX", "ISRG", "EW"],
    "건강보험": ["UNH", "ELV", "CI", "CVS", "HUM"],
    "대형 유통": ["WMT", "COST", "TGT", "BJ", "DG"],
    "주택용품 유통": ["HD", "LOW"],
    "외식": ["MCD", "SBUX", "CMG", "YUM", "QSR", "DRI"],
    "음료": ["KO", "PEP", "KDP", "MNST"],
    "석유 메이저": ["XOM", "CVX", "COP", "SHEL", "BP", "OXY"],
    "방산·항공우주": ["LMT", "RTX", "NOC", "GD", "LHX", "BA"],
    "전력·AI 인프라": ["GEV", "VRT", "ETN", "PWR", "CEG", "VST"],
    "항공사": ["DAL", "UAL", "AAL", "LUV"],
    "통신": ["T", "VZ", "TMUS"],
    "빅테크": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
}


def _curated(key: str, groups: dict):
    """그 종목이 들어 있는 첫 그룹 (그룹 이름, 종목 목록)."""
    for name, members in groups.items():
        if key in members:
            return name, members
    return None, []


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _info(sym: str) -> dict:
    for _ in range(3):
        try:
            d = yf.Ticker(sym).info or {}
            if d:
                return {k: d.get(k) for k in [v[0] for v in METRICS.values()] +
                        ["shortName", "longName", "marketCap", "currency", "industry", "sector", "industryKey", "currentPrice", "regularMarketPrice"]}
        except Exception:  # noqa: BLE001
            pass
    return {}


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def us_peers(industry_key: str, n: int = 10) -> list:
    try:
        ind = yf.Industry(industry_key)
        top = ind.top_companies
        if isinstance(top, pd.DataFrame) and not top.empty:
            return [str(s) for s in top.index[:n]]
    except Exception:  # noqa: BLE001
        pass
    return []


def _near_size(df: pd.DataFrame, key_col: str, me: str, cap_col: str, n: int) -> list:
    """같은 업종 안에서 업계 대표(시가총액 상위 2개) + 시가총액이 비슷한 기업."""
    d = df.dropna(subset=[cap_col])
    d = d[d[cap_col] > 0]
    if d.empty:
        return []
    top = list(d.nlargest(2, cap_col)[key_col])
    mine = d.loc[d[key_col] == me, cap_col]
    if len(mine):
        lc = np.log(float(mine.iloc[0]))
        near = d.assign(_dist=(np.log(d[cap_col]) - lc).abs()).nsmallest(n + 1, "_dist")[key_col].tolist()
    else:
        near = list(d.nlargest(n, cap_col)[key_col])
    return list(dict.fromkeys(top + near))[:n + 1]


def kr_peers(code: str, listing: pd.DataFrame, n: int = 9):
    """(야후 심볼 목록, 설명)."""
    from kr import yahoo_symbol

    mk = dict(zip(listing["Code"], listing["Market"]))
    sym = lambda c: yahoo_symbol(c, mk.get(c, "KOSPI"))  # noqa: E731
    name, members = _curated(code, PEER_GROUPS_KR)
    if members:
        ok = [c for c in members if c in mk]
        return [sym(c) for c in ok], f"시장에서 함께 비교하는 '{name}' 대표 기업 (증권사 업종 리포트 기준)"
    try:
        import sectors

        m, _day = sectors.wics_map()
        row = m[m["Code"] == code]
        if len(row):
            ind = row["Industry"].iloc[0]
            same = m[m["Industry"] == ind]
            codes = _near_size(same, "Code", code, "Mcap", n)
            return [sym(c) for c in codes if c in mk], f"WICS 업종 '{ind}'에서 업계 대표와 시가총액이 비슷한 기업"
    except Exception:  # noqa: BLE001
        pass
    me = listing[listing["Code"] == code]
    if me.empty or "Industry" not in listing:
        return [], ""
    ind = me["Industry"].iloc[0]
    if not isinstance(ind, str) or not ind or "기업부" in ind:
        return [], ""
    same = listing[listing["Industry"] == ind]
    codes = _near_size(same, "Code", code, "Marcap", n)
    return [sym(c) for c in codes], f"KRX 업종 '{ind}'에서 시가총액이 비슷한 기업"


def us_peer_list(ticker: str, info: dict, n: int = 9):
    """(심볼 목록, 설명)."""
    t = ticker.upper().replace("-", ".")
    name, members = _curated(t, {k: [x.replace("-", ".") for x in v] for k, v in PEER_GROUPS_US.items()})
    if members:
        return [x.replace(".", "-") for x in members], f"시장에서 함께 비교하는 '{name}' 대표 기업 (증권사 업종 리포트 기준)"
    try:
        import sectors

        sp = sectors.sp500_list()
        row = sp[sp["key"] == t]
        if len(row):
            ind = row["Industry"].iloc[0]
            same = sp[sp["Industry"] == ind].copy()
            if len(same) >= 3:
                try:
                    q = sectors.nasdaq_snapshot()
                    same = same.merge(q[["key", "Marcap"]], on="key", how="left")
                except Exception:  # noqa: BLE001
                    same["Marcap"] = 1.0
                keys = _near_size(same.fillna({"Marcap": 1.0}), "key", t, "Marcap", n)
                return [k.replace(".", "-") for k in keys], f"S&P 500 GICS 세부 업종 '{ind}'에서 시가총액이 비슷한 기업"
    except Exception:  # noqa: BLE001
        pass
    ik = info.get("industryKey")
    peers = us_peers(ik) if ik else []
    return peers, (f"Yahoo 산업 분류 '{info.get('industry') or ik}' 대표 기업" if peers else "")


def _val(info: dict, key: str, kind: str):
    v = info.get(key)
    try:
        v = float(v)
    except Exception:  # noqa: BLE001
        return np.nan
    if not np.isfinite(v):
        return np.nan
    if kind == "%":
        return v * 100
    if kind == "y":  # Yahoo 배당수익률은 버전에 따라 0.0123 또는 1.23
        return v if v > 0.5 else v * 100
    return v


def _fmt(v, kind):
    if not np.isfinite(v):
        return "-"
    if kind in ("%", "y"):
        return f"{v:.1f}%"
    if kind == "d":
        return f"{v:.0f}%"
    return f"{v:.2f}" if kind == "r" else f"{v:.1f}"


def percentile_rank(values: pd.Series, target: str, higher_good):
    """업계 안에서 '좋은 쪽'으로 몇 % 위치인지 (100 = 가장 좋음)."""
    s = values.dropna()
    if target not in s.index or len(s) < 3 or higher_good is None:
        return np.nan
    x = s[target]
    others = s
    if higher_good:
        return float((others < x).sum() + 0.5 * ((others == x).sum() - 1)) / (len(others) - 1) * 100
    return float((others > x).sum() + 0.5 * ((others == x).sum() - 1)) / (len(others) - 1) * 100


def _place(values: pd.Series, target: str, higher_good) -> str:
    """'N개 중 k위' (좋은 쪽이 1위)."""
    s = values.dropna()
    if target not in s.index or len(s) < 3:
        return "순위 없음"
    x = s[target]
    better = (s > x).sum() if higher_good else (s < x).sum()
    return f"{len(s)}개 중 {int(better) + 1}위"


def render_peers(ticker: str, R: dict, domestic: bool = False, code: str | None = None) -> None:
    if yf is None:
        st.error("yfinance가 필요합니다.")
        return
    info = R.get("info") or {}
    peers = []
    if domestic:
        from kr import load_krx

        listing = load_krx()
        peers, why = kr_peers(code, listing)
    else:
        peers, why = us_peer_list(ticker, info)
    if peers and why:
        st.caption(why + " · 아래 칸에 직접 입력해 바꿀 수 있습니다")
    k = f"pe_{ticker}"
    extra = st.text_input("비교 종목 추가·변경 (쉼표로 구분, 비우면 자동)", key=k + "_x",
                          placeholder="예: MSFT, GOOGL" if not domestic else "예: 000660, 005380")
    if extra.strip():
        toks = [t.strip().upper() for t in extra.split(",") if t.strip()]
        if domestic:
            from kr import resolve

            conv = []
            for t in toks:
                c = resolve(t)
                conv.append(c[0]["yahoo"] if c else t)
            toks = conv
        peers = toks
    peers = [p for p in dict.fromkeys([ticker] + peers)][:12]
    if len(peers) < 2:
        st.info("비교할 동종업계 기업을 찾지 못했습니다. 위에 비교 종목을 직접 입력하세요.")
        return
    with st.spinner(f"{len(peers)}개 기업 지표를 불러오는 중..."):
        with ThreadPoolExecutor(max_workers=6) as ex:
            infos = dict(zip(peers, ex.map(_info, peers)))
    infos = {p: d for p, d in infos.items() if d}
    if ticker not in infos:
        infos[ticker] = info
    krx_names = {}
    if domestic:
        from kr import load_krx

        L = load_krx()
        krx_names = {f"{c}.KS": n for c, n in zip(L["Code"], L["Name"])}
        krx_names.update({f"{c}.KQ": n for c, n in zip(L["Code"], L["Name"])})
    label = {p: krx_names.get(p) or d.get("shortName") or p for p, d in infos.items()}
    T = pd.DataFrame({m: {p: _val(d, key, kind) for p, d in infos.items()} for m, (key, kind, _g) in METRICS.items()})
    T["시가총액"] = pd.Series({p: float(d.get("marketCap") or np.nan) for p, d in infos.items()})
    T = T.sort_values("시가총액", ascending=False)
    for m in GOOD_LOW_SKIP_NEG:
        T.loc[T[m] <= 0, m] = np.nan

    # ---- 이 회사의 업계 내 위치
    st.markdown(f"##### {label.get(ticker, ticker)}의 업계 내 위치 ({len(T)}개 기업 중)")
    ranks = {m: percentile_rank(T[m], ticker, g) for m, (_k, _kind, g) in METRICS.items()}
    med = T.median(numeric_only=True)
    cells = ""
    for m, (_k, kind, g) in METRICS.items():
        if g is None:
            continue
        v = T.at[ticker, m] if ticker in T.index else np.nan
        rk = ranks[m]
        col = "#5BB56E" if np.isfinite(rk) and rk >= 66 else ("#E5534B" if np.isfinite(rk) and rk <= 33 else "#B0B4BA")
        bar = (f'<div style="height:5px;background:rgba(128,128,128,.2);border-radius:3px;margin-top:3px">'
               f'<div style="width:{rk if np.isfinite(rk) else 0:.0f}%;height:5px;background:{col};border-radius:3px"></div></div>')
        cells += (f'<div style="min-width:0"><div style="font-size:0.78rem"><span style="opacity:.7">{m}</span>{icon(tip(m))}</div>'
                  f'<div style="font-size:1.0rem;font-weight:600">{_fmt(v, kind)} <span style="font-size:0.75rem;opacity:.6">중앙값 {_fmt(med.get(m, np.nan), kind)}</span></div>'
                  f'{bar}<div style="font-size:0.7rem;opacity:.6">{_place(T[m], ticker, g)}</div></div>')
    st.markdown(CSS + f'<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px 14px;margin:2px 0 6px">{cells}</div>'
                f'<div style="font-size:0.75rem;opacity:.6;margin-bottom:10px">막대가 길수록(초록) 업계에서 좋은 쪽입니다 — 배수는 낮을수록, 마진·성장·ROE는 높을수록 좋게 봅니다.</div>',
                unsafe_allow_html=True)

    # ---- 비교표
    st.markdown("##### 지표 비교")
    show = pd.DataFrame(index=T.index)
    show["기업"] = [("▶ " if p == ticker else "") + label.get(p, p) for p in T.index]
    show["시가총액"] = [money.big(v, (infos.get(p) or {}).get("currency") or "USD") for p, v in T["시가총액"].items()]
    for m, (_k, kind, _g) in METRICS.items():
        show[m] = [_fmt(v, kind) for v in T[m]]
    show.loc["업계 중앙값"] = ["업계 중앙값", "-"] + [_fmt(med.get(m, np.nan), METRICS[m][1]) for m in METRICS]
    from glossary import col_help

    st.dataframe(show.reset_index(drop=True), use_container_width=True, hide_index=True, column_config=col_help(show.columns))

    # ---- 산점도
    st.markdown("##### 두 지표로 비교")
    c1, c2 = st.columns(2)
    num = [m for m in METRICS if T[m].notna().sum() >= 3]
    if len(num) < 2:
        return
    xm = c1.selectbox("가로축", num, index=num.index("이익 성장률") if "이익 성장률" in num else 0, key=k + "_sx")
    ym = c2.selectbox("세로축", num, index=num.index("PER") if "PER" in num else 1, key=k + "_sy")
    d = T[[xm, ym, "시가총액"]].dropna(subset=[xm, ym])
    if len(d) < 2:
        st.caption("두 지표가 모두 있는 기업이 부족합니다.")
        return
    sz = np.sqrt(d["시가총액"].fillna(d["시가총액"].median()).clip(lower=1))
    sz = 12 + 30 * (sz - sz.min()) / (sz.max() - sz.min() + 1e-9)
    f = go.Figure(go.Scatter(x=d[xm], y=d[ym], mode="markers+text", text=[label.get(p, p) for p in d.index], textposition="top center",
                             marker=dict(size=sz, color=["#E5534B" if p == ticker else "#4C9BE8" for p in d.index], opacity=0.8),
                             hovertemplate="%{text}<br>" + xm + " %{x:.2f}<br>" + ym + " %{y:.2f}<extra></extra>"))
    if len(d) >= 3:
        try:
            b = np.polyfit(d[xm].values, d[ym].values, 1)
            xs = np.linspace(d[xm].min(), d[xm].max(), 20)
            f.add_trace(go.Scatter(x=xs, y=b[0] * xs + b[1], mode="lines", line=dict(color="#888", dash="dash", width=1),
                                   name="추세선", hoverinfo="skip"))
        except Exception:  # noqa: BLE001
            pass
    f.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), showlegend=False,
                    xaxis_title=xm, yaxis_title=ym)
    st.plotly_chart(f, use_container_width=True)
