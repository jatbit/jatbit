"""내부자 거래·지분 구조 — 기업 리서치 탭의 하위 탭
- 내부자 거래: SEC EDGAR Form 4 원문(XML)을 직접 읽어 거래 내역을 만든다 (실패하면 Yahoo 요약으로 대신)
- 지분 구조: 내부자·기관 보유 비율, 상위 기관·펀드 보유자, 공매도 잔고
"""
import concurrent.futures as cf
import time
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from options import mgrid

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None

CODE_KO = {"P": "장내 매수", "S": "장내 매도", "A": "주식 보상(부여)", "M": "옵션·RSU 행사", "F": "세금 납부용 처분",
           "G": "증여", "C": "전환", "X": "옵션 행사", "D": "회사에 처분", "J": "기타", "W": "상속", "I": "재량 거래",
           "V": "자발적 보고", "E": "만기 소멸", "H": "만기 소멸", "O": "옵션 행사(외가)", "K": "스왑", "L": "소액 취득", "Z": "신탁"}


def _sec_get(url: str, ua: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _txt(el, path):
    x = el.find(path)
    if x is None:
        return None
    v = x.find("value")
    s = (v.text if v is not None else x.text) or ""
    s = s.strip()
    return s or None


def _f(s):
    try:
        return float(str(s).replace(",", ""))
    except Exception:  # noqa: BLE001
        return np.nan


def parse_form4(xml: bytes, filed: str = "", url: str = "") -> list:
    """Form 4 XML → 거래 행 목록 (비파생 거래 + 파생 거래)."""
    root = ET.fromstring(xml)
    owners, roles = [], []
    for ro in root.findall("reportingOwner"):
        nm = (ro.findtext("reportingOwnerId/rptOwnerName") or "").strip()
        rel = ro.find("reportingOwnerRelationship")
        r = []
        if rel is not None:
            if (rel.findtext("isDirector") or "").strip() in ("1", "true"):
                r.append("이사")
            if (rel.findtext("isOfficer") or "").strip() in ("1", "true"):
                r.append((rel.findtext("officerTitle") or "임원").strip())
            if (rel.findtext("isTenPercentOwner") or "").strip() in ("1", "true"):
                r.append("10% 이상 주주")
            if (rel.findtext("isOther") or "").strip() in ("1", "true"):
                r.append((rel.findtext("otherText") or "기타").strip())
        owners.append(nm)
        roles.append(", ".join(r))
    plan_ids = {f.get("id") for f in root.findall("footnotes/footnote") if "10b5-1" in "".join(f.itertext())}
    plan_all = (root.findtext("aff10b5One") or "").strip() in ("1", "true")
    rows = []
    for tbl, deriv in (("nonDerivativeTable/nonDerivativeTransaction", False), ("derivativeTable/derivativeTransaction", True)):
        for t in root.findall(tbl):
            code = (t.findtext("transactionCoding/transactionCode") or "").strip()
            sh = _f(_txt(t, "transactionAmounts/transactionShares"))
            pxs = _f(_txt(t, "transactionAmounts/transactionPricePerShare"))
            ad = _txt(t, "transactionAmounts/transactionAcquiredDisposedCode")
            plan = plan_all or any(x.get("id") in plan_ids for x in t.iter("footnoteId"))
            rows.append({
                "date": _txt(t, "transactionDate"), "filed": filed, "owner": " / ".join(owners), "role": " / ".join(x for x in roles if x),
                "security": _txt(t, "securityTitle") or "", "derivative": deriv, "code": code, "code_ko": CODE_KO.get(code, code),
                "shares": sh, "price": pxs, "ad": ad, "signed": sh if ad == "A" else (-sh if ad == "D" else np.nan),
                "value": sh * pxs if np.isfinite(sh) and np.isfinite(pxs) else np.nan,
                "after": _f(_txt(t, "postTransactionAmounts/sharesOwnedFollowingTransaction")),
                "direct": _txt(t, "ownershipNature/directOrIndirectOwnership"), "plan": plan, "url": url})
    return rows


@st.cache_data(ttl=6 * 60 * 60, show_spinner="SEC에서 내부자 거래 공시(Form 4)를 읽는 중... (건수가 많으면 30초 이상 걸립니다)")
def load_form4(ticker: str, months: int = 12, max_n: int = 80) -> pd.DataFrame:
    from research import SEC_UA, _sec_cik_map, _sec_json, _try

    cik = _sec_cik_map().get(ticker.upper()) or _sec_cik_map().get(ticker.upper().replace(".", "-"))
    if not cik:
        raise RuntimeError("SEC 등록 기업이 아닙니다")
    sub = _try(lambda: _sec_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"), tries=3)
    rec = sub.get("filings", {}).get("recent", {})
    df = pd.DataFrame({k: rec.get(k, []) for k in ("form", "accessionNumber", "filingDate", "primaryDocument")})
    if df.empty:
        return pd.DataFrame()
    cut = (pd.Timestamp.now() - pd.DateOffset(months=months)).strftime("%Y-%m-%d")
    df = df[(df["form"] == "4") & (df["filingDate"] >= cut)].head(max_n)
    jobs = []
    for _, r in df.iterrows():
        doc = r["primaryDocument"].split("/")[-1]
        acc = r["accessionNumber"].replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
        jobs.append((url, r["filingDate"]))

    def one(job):
        url, filed = job
        for i in range(3):
            try:
                time.sleep(0.15)  # SEC 공정 사용 정책(초당 10회 이하)
                return parse_form4(_sec_get(url, SEC_UA), filed, url)
            except Exception:  # noqa: BLE001
                time.sleep(1.0 * (i + 1))
        return []

    rows = []
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for out in ex.map(one, jobs):
            rows.extend(out)
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["date"].str[:10], errors="coerce")
    return d.sort_values(["date", "filed"], ascending=False).reset_index(drop=True)


@st.cache_data(ttl=6 * 60 * 60, show_spinner="지분 구조를 불러오는 중...")
def load_holders(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    out = {}
    for k, fn in (("major", lambda: t.major_holders), ("inst", lambda: t.institutional_holders),
                  ("fund", lambda: t.mutualfund_holders), ("yins", lambda: t.insider_transactions),
                  ("roster", lambda: t.insider_roster_holders)):
        for i in range(3):
            try:
                v = fn()
                out[k] = v if isinstance(v, pd.DataFrame) else pd.DataFrame()
                break
            except Exception:  # noqa: BLE001
                out[k] = pd.DataFrame()
                time.sleep(1.0 * (i + 1))
    return out


def _big(x):
    if x is None or not np.isfinite(x):
        return "-"
    a = abs(x)
    for d, u in ((1e12, "조"), (1e9, "십억"), (1e6, "백만")):
        if a >= d:
            return f"{'-' if x < 0 else ''}${a / d:,.2f}{u}"
    return f"{'-' if x < 0 else ''}${a:,.0f}"


def _pct(x):
    try:
        v = float(x)
        return f"{v * 100:.2f}%" if np.isfinite(v) else "-"
    except Exception:  # noqa: BLE001
        return "-"


def _hold_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    ren = {"Holder": "보유 기관", "Shares": "보유 주식 수", "Date Reported": "보고일", "pctHeld": "지분율", "Value": "평가액",
           "pctChange": "직전 대비 증감", "% Out": "지분율"}
    d = d.rename(columns=ren)
    if "보고일" in d:
        d["보고일"] = pd.to_datetime(d["보고일"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "지분율" in d:
        d["지분율"] = d["지분율"].map(_pct)
    if "직전 대비 증감" in d:
        d["직전 대비 증감"] = d["직전 대비 증감"].map(lambda v: f"{float(v) * 100:+.1f}%" if pd.notna(v) else "-")
    if "보유 주식 수" in d:
        d["보유 주식 수"] = d["보유 주식 수"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "-")
    if "평가액" in d:
        d["평가액"] = d["평가액"].map(lambda v: _big(float(v)) if pd.notna(v) else "-")
    return d[[c for c in ["보유 기관", "지분율", "보유 주식 수", "평가액", "직전 대비 증감", "보고일"] if c in d]]


def render_insider(ticker: str, R: dict) -> None:
    info = R.get("info") or {}
    st.markdown("##### 지분 구조")
    try:
        H = load_holders(ticker)
    except Exception as e:  # noqa: BLE001
        H = {}
        st.caption(f"Yahoo 보유자 정보를 불러오지 못했습니다: {e}")
    mj = H.get("major", pd.DataFrame())
    mv = {}
    if isinstance(mj, pd.DataFrame) and not mj.empty:
        col = "Value" if "Value" in mj.columns else mj.columns[-1]
        mv = {str(i): mj.loc[i, col] for i in mj.index}
    ins = mv.get("insidersPercentHeld", info.get("heldPercentInsiders"))
    inst = mv.get("institutionsPercentHeld", info.get("heldPercentInstitutions"))
    try:
        ins_f, inst_f = float(ins or 0), float(inst or 0)
    except Exception:  # noqa: BLE001
        ins_f, inst_f = 0.0, 0.0
    so, fl = info.get("sharesOutstanding"), info.get("floatShares")
    mgrid([("내부자 보유", _pct(ins), "임원·이사·10% 이상 주주"),
           ("기관 보유", _pct(inst), "13F 보고 기관 합계"),
           ("기관 보유 (유통주식 대비)", _pct(mv.get("institutionsFloatPercentHeld"))),
           ("보고 기관 수", f"{float(mv['institutionsCount']):,.0f}곳" if mv.get("institutionsCount") is not None else "-"),
           ("발행주식수", f"{so / 1e6:,.1f}백만주" if so else "-"),
           ("유통주식수", f"{fl / 1e6:,.1f}백만주" if fl else "-", "내부자·전략적 보유분을 뺀 시장 유통 물량"),
           ("공매도 잔고 (유통 대비)", _pct(info.get("shortPercentOfFloat")),
            f"{info['sharesShort'] / 1e6:,.1f}백만주" if info.get("sharesShort") else ""),
           ("공매도 상환일수", f"{info['shortRatio']:.1f}일" if info.get("shortRatio") else "-", "공매도 잔고 ÷ 일평균 거래량")], cols=4)
    if ins_f or inst_f:
        other = max(0.0, 1 - ins_f - min(inst_f, 1 - ins_f))
        pie = go.Figure(go.Pie(labels=["내부자", "기관", "개인·기타"], values=[ins_f, min(inst_f, 1 - ins_f), other], hole=0.5,
                               marker=dict(colors=["#F2A93B", "#4C9BE8", "#B0B4BA"]), sort=False,
                               hovertemplate="%{label} %{percent}<extra></extra>"))
        pie.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                          legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0))
        st.plotly_chart(pie, use_container_width=True)
        st.caption("기관 보유 비율은 13F 보고 시점(분기 말, 최대 45일 지연) 기준이며 일부 중복 집계로 100%를 넘을 수 있어 그림에서는 상한을 둡니다.")
    t1, t2 = st.tabs(["상위 기관 보유자", "상위 펀드 보유자"])
    with t1:
        ti = _hold_table(H.get("inst"))
        st.dataframe(ti, use_container_width=True, hide_index=True) if not ti.empty else st.caption("자료 없음")
    with t2:
        tf = _hold_table(H.get("fund"))
        st.dataframe(tf, use_container_width=True, hide_index=True) if not tf.empty else st.caption("자료 없음")

    # ---- 내부자 거래
    st.markdown("##### 내부자 거래 (SEC Form 4)")
    months = st.select_slider("조회 기간", options=[3, 6, 12, 24], value=12, key=f"ins_m_{ticker}", format_func=lambda m: f"최근 {m}개월")
    d, err = pd.DataFrame(), None
    try:
        d = load_form4(ticker, int(months), 80 if months <= 12 else 150)
    except Exception as e:  # noqa: BLE001
        err = str(e)
    if d.empty:
        y = H.get("yins", pd.DataFrame())
        if err:
            st.caption(f"SEC Form 4를 읽지 못해 Yahoo 요약으로 대신합니다 ({err}).")
        if isinstance(y, pd.DataFrame) and not y.empty:
            y = y.rename(columns={"Insider": "내부자", "Position": "직위", "Text": "내용", "Shares": "주식 수", "Value": "금액",
                                  "Start Date": "거래일", "Ownership": "보유 형태", "Transaction": "구분"})
            st.dataframe(y.drop(columns=[c for c in ("URL",) if c in y]), use_container_width=True, hide_index=True)
        else:
            st.caption("최근 내부자 거래가 없거나 자료를 불러오지 못했습니다.")
        return

    nd = d[~d["derivative"]]
    ps = nd[nd["code"].isin(["P", "S"])]
    now = pd.Timestamp.now()
    summ = []
    for m in (3, 6, 12):
        if m > months:
            continue
        x = ps[ps["date"] >= now - pd.DateOffset(months=m)]
        buy, sell = x[x["code"] == "P"], x[x["code"] == "S"]
        summ.append((f"최근 {m}개월 순매수", _big(buy["value"].sum() - sell["value"].sum()),
                     f"매수 {len(buy)}건 {_big(buy['value'].sum())} · 매도 {len(sell)}건 {_big(sell['value'].sum())}"))
    plan_share = ps[ps["code"] == "S"]["plan"].mean() if (ps["code"] == "S").any() else np.nan
    buyers = ps[ps["code"] == "P"]["owner"].nunique()
    summ += [("매수한 내부자 수", f"{buyers}명", "여러 명이 동시에 장내 매수하면(클러스터 매수) 의미 있는 신호로 자주 해석됨"),
             ("10b5-1 계획 매도 비중", f"{plan_share * 100:.0f}%" if np.isfinite(plan_share) else "-",
              "미리 정해 둔 자동 매도 계획에 따른 매도. 비중이 높을수록 매도의 신호 의미가 약함"),
             ("공시 건수", f"{d['filed'].nunique()}건", f"거래 {len(d)}줄 (파생 포함)")]
    mgrid(summ, cols=len(summ) if len(summ) <= 5 else 3)

    hist = R.get("hist")
    if isinstance(hist, pd.Series) and len(hist):
        h = hist.copy()
        h.index = pd.to_datetime(h.index).tz_localize(None) if getattr(h.index, "tz", None) is not None else pd.to_datetime(h.index)
        h = h[h.index >= now - pd.DateOffset(months=int(months) + 1)]
        f = go.Figure(go.Scatter(x=h.index, y=h.values, name="주가", line=dict(color="#888", width=1.5),
                                 hovertemplate="%{x|%Y-%m-%d} %{y:,.2f}<extra></extra>"))
        for code, nm, color, sym in (("P", "장내 매수", "#1B9E4B", "triangle-up"), ("S", "장내 매도", "#E5534B", "triangle-down")):
            x = ps[ps["code"] == code].groupby(["date", "owner"]).agg(value=("value", "sum"), shares=("shares", "sum"),
                                                                    role=("role", "first"), plan=("plan", "max")).reset_index()
            if x.empty:
                continue
            y = h.reindex(x["date"], method="nearest").values if len(h) else x["value"] * 0
            size = np.clip(np.sqrt(x["value"].fillna(0) / 1e5) * 3, 7, 30)
            f.add_trace(go.Scatter(x=x["date"], y=y, mode="markers", name=nm, marker=dict(color=color, symbol=sym, size=size, line=dict(width=0.5, color="white")),
                                   customdata=np.stack([x["owner"], x["role"], x["shares"], x["value"] / 1e6, np.where(x["plan"], "10b5-1 계획", "")], axis=1),
                                   hovertemplate="%{x|%Y-%m-%d} " + nm + "<br>%{customdata[0]} (%{customdata[1]})<br>%{customdata[2]:,.0f}주 · %{customdata[3]:,.2f}백만$ %{customdata[4]}<extra></extra>"))
        f.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="closest",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(f, use_container_width=True)
        st.caption("삼각형 크기는 거래 금액에 비례합니다. 장내 매수(P)·매도(S)만 표시하고 주식 보상·세금 처분·옵션 행사는 제외했습니다.")

    if not ps.empty:
        st.markdown("###### 내부자별 장내 매매 합계")
        by = ps.assign(buy=np.where(ps["code"] == "P", ps["value"], 0.0), sell=np.where(ps["code"] == "S", ps["value"], 0.0),
                       bsh=np.where(ps["code"] == "P", ps["shares"], 0.0), ssh=np.where(ps["code"] == "S", ps["shares"], 0.0))
        g = by.groupby("owner").agg(직위=("role", "first"), 매수액=("buy", "sum"), 매도액=("sell", "sum"), 매수주식=("bsh", "sum"),
                                    매도주식=("ssh", "sum"), 최근거래=("date", "max"), 현재보유=("after", "first"))
        g["순매수액"] = g["매수액"] - g["매도액"]
        g = g.sort_values("순매수액")
        show = g.copy().astype(object)
        for c in ("매수액", "매도액", "순매수액"):
            show[c] = g[c].map(_big)
        for c in ("매수주식", "매도주식", "현재보유"):
            show[c] = g[c].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "-")
        show["최근거래"] = g["최근거래"].dt.strftime("%Y-%m-%d")
        st.dataframe(show[["직위", "순매수액", "매수액", "매도액", "매수주식", "매도주식", "현재보유", "최근거래"]], use_container_width=True)

    st.markdown("###### 거래 내역")
    only_ps = st.toggle("장내 매수·매도만 보기", value=False, key=f"ins_ps_{ticker}")
    v = ps if only_ps else d
    t = pd.DataFrame({"거래일": v["date"].dt.strftime("%Y-%m-%d"), "공시일": v["filed"], "내부자": v["owner"], "직위": v["role"],
                      "구분": v["code"] + " " + v["code_ko"], "증권": v["security"],
                      "주식 수": v["signed"].map(lambda x: f"{x:+,.0f}" if pd.notna(x) else "-"),
                      "단가": v["price"].map(lambda x: f"${x:,.2f}" if pd.notna(x) and x > 0 else "-"),
                      "금액": v["value"].map(lambda x: _big(x) if pd.notna(x) and x > 0 else "-"),
                      "거래 후 보유": v["after"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "-"),
                      "직접/간접": v["direct"].map({"D": "직접", "I": "간접"}).fillna("-"),
                      "10b5-1": np.where(v["plan"], "예", ""), "원문": v["url"]})
    try:
        st.dataframe(t, use_container_width=True, hide_index=True,
                     column_config={"원문": st.column_config.LinkColumn("원문", display_text="보기")})
    except Exception:  # noqa: BLE001
        st.dataframe(t, use_container_width=True, hide_index=True)
    st.caption("출처: SEC Form 4. 구분 코드: P 장내 매수, S 장내 매도, A 보상 부여, M·X 옵션/RSU 행사, F 세금 납부용 처분, G 증여")
