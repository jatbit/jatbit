"""나라 지도 화면: 종목 본사 위치를 점으로 찍고, 구역(미국 주 · 한국 시도)을 고른 기준으로 색칠한다.

데이터(본사 위치·경계·값 계산)는 geodata.py. 시장 페이지(market.py)가 세계 지도 아래에서 render()를 부른다.

화면 구성
- 색 기준: 수익률 · 지수 대비(초과수익) · 거래대금 증가율 · PER · 종합 점수 (PER·점수는 버튼을 눌러 따로 불러옴)
- 섹터 필터: 고른 섹터 종목만 지도·구역 값에 넣는다
- 기간 재생: 1일 → 1주 → … → 1년을 지도 아래 재생 버튼·슬라이더로 넘겨 본다 (수익률·지수 대비만)
- 구역을 누르면 아래에 그 구역 카드·시장 지도·종목 목록, 점을 누르면 그 종목 카드
수익률 기간은 세계 지도에서 고른 기간(mk_world_per)을 그대로 쓴다.
세션 키 접두어: gm_usa_ · gm_kor_
"""
from __future__ import annotations

import html as _html
import math

import numpy as np
import pandas as pd
import streamlit as st

import geodata as gd
import ui

UPDOWN = [[0, ui.DOWN], [0.5, "#D9D6CF"], [1, ui.UP]]
SCALES = {
    "수익률": UPDOWN, "지수 대비": UPDOWN, "종합 점수": UPDOWN,
    "거래대금 증가율": [[0, "#7D8696"], [0.5, "#DEDAD2"], [1, "#E8912D"]],
    "PER": [[0, "#2F8F83"], [0.5, "#DEDAD2"], [1, "#B5532E"]],
}
INDEX = {"USA": ("^GSPC", "S&P 500"), "KOR": ("^KS11", "코스피")}


def fmt(basis: str, v, short: bool = False) -> str:
    if v is None or v != v:
        return "-"
    if basis == "수익률":
        return f"{v:+.1f}%" if short else f"{v:+.2f}%"
    if basis == "지수 대비":
        return f"{v:+.1f}%p" if short else f"{v:+.2f}%p"
    if basis == "거래대금 증가율":
        return f"{v:+.0f}%"
    if basis == "PER":
        return f"{v:.0f}배" if short else f"{v:.1f}배"
    return f"{v:.0f}점"


def to_z(basis: str, vals, ref: float = np.nan) -> np.ndarray:
    """색에 쓸 숫자 (가운데 0). PER은 나라 중앙값 대비 배수의 log2, 거래대금은 평소 대비 배수의 log2, 점수는 50 기준."""
    v = np.asarray([x if x is not None else np.nan for x in vals], dtype=float)
    if basis == "거래대금 증가율":
        z = np.log2(np.clip(1 + v / 100, 0.05, None))
    elif basis == "PER":
        z = np.where(v > 0, np.log2(np.clip(v, 1e-6, None) / (ref if ref and ref == ref and ref > 0 else 15.0)), np.nan)
    elif basis == "종합 점수":
        z = v - 50
    else:
        z = v
    return z


def z_range(basis: str, zs) -> float:
    a = np.abs(np.asarray([x for x in zs if x == x], dtype=float))
    lo = {"수익률": 0.5, "지수 대비": 0.5, "거래대금 증가율": 0.4, "PER": 0.5, "종합 점수": 10}[basis]
    hi = {"PER": 2.0, "거래대금 증가율": 2.0, "종합 점수": 40}.get(basis, np.inf)
    return float(min(max(float(np.percentile(a, 90)) if len(a) else lo, lo), hi))


def _nz(z) -> list:
    return [float(x) if x == x else 0.0 for x in z]


# ----------------------------------------------------------------------
# 그림
# ----------------------------------------------------------------------
def _traces(iso, reg, pts, gj, basis, label, ref, zm, sel, lab_index, static_pts: bool = False):
    """지도 그림의 trace 3개: [구역 색칠, 종목 점, 구역 글자]. 날짜 장면(frame)도 같은 함수로 만든다.
    static_pts: 점에 마우스를 올렸을 때 값은 빼고 이름·본사·시가총액만 (장면마다 글자를 다시 보내지 않으려고)."""
    import plotly.graph_objects as go

    scale = SCALES[basis]
    rz = _nz(to_z(basis, reg["V"], ref))
    rtxt = [f"<b>{_html.escape(str(r.Name))}</b><br>{label} {fmt(basis, r.V)}<br>{int(r.N)}종목 · 시가총액 {r.CapTxt}<br>{_html.escape(str(r.Top))}"
            for r in reg.itertuples()]
    lw = [2.6 if k == sel else 0.6 for k in reg.index]
    lc = ["#111" if k == sel else "rgba(255,255,255,.75)" for k in reg.index]
    common = dict(colorscale=scale, zmin=-zm, zmax=zm, showscale=False, hovertemplate="%{text}<extra></extra>")
    if iso == "USA":
        t0 = go.Choropleth(locations=list(reg["Abbr"]), locationmode="USA-states", z=rz, text=rtxt,
                           marker=dict(line=dict(width=lw, color=lc)), **common)
    elif gj is not None:  # 종목이 없는 시도도 빈칸으로 그려 지도 모양을 유지한다
        locs = list(dict.fromkeys(list(reg.index) + [f["properties"]["_k"] for f in gj["features"]]))
        extra = [k for k in locs if k not in reg.index]
        t0 = go.Choropleth(geojson=gj, featureidkey="properties._k", locations=locs, z=rz + [0.0] * len(extra),
                           text=rtxt + [f"<b>{gd.SIDO[k][0] if k in gd.SIDO else k}</b><br>목록 안에 본사를 둔 종목 없음" for k in extra],
                           marker=dict(line=dict(width=lw + [0.6] * len(extra), color=lc + ["rgba(255,255,255,.75)"] * len(extra))),
                           **common)
    else:  # 경계를 못 받으면 시도 중심에 큰 원으로
        cmax = float(reg["Cap"].max()) if len(reg) else 1.0
        t0 = go.Scattergeo(lat=list(reg["lat"]), lon=list(reg["lon"]), mode="markers", text=rtxt, hoverinfo="text",
                           marker=dict(size=[12 + 30 * math.sqrt(c / cmax) for c in reg["Cap"]], color=rz, colorscale=scale,
                                       cmin=-zm, cmax=zm, opacity=.45, line=dict(width=lw, color=lc)))
    cap_max = float(pts["Marcap"].max()) if len(pts) else 1.0
    pz = _nz(to_z(basis, pts["V"], ref))
    extra_ret = basis != "수익률"
    if static_pts:
        ptxt = [f"<b>{_html.escape(str(r.Label))}</b><br>본사 {_html.escape(str(r.Place))}<br>시가총액 {r.CapTxt}" for r in pts.itertuples()]
    else:
        ptxt = [f"<b>{_html.escape(str(r.Label))}</b><br>본사 {_html.escape(str(r.Place))}<br>{label} {fmt(basis, r.V)}"
                + (f"<br>수익률 {fmt('수익률', r.Ret)}" if extra_ret else "") + f"<br>시가총액 {r.CapTxt}" for r in pts.itertuples()]
    t1 = go.Scattergeo(lat=list(pts["plat"]), lon=list(pts["plon"]), mode="markers", text=ptxt, hoverinfo="text",
                       marker=_pt_marker(pts, pz, scale, zm, cap_max))
    lab = reg.loc[lab_index]
    t2 = go.Scattergeo(lat=list(lab["lat"]), lon=list(lab["lon"]), mode="text", hoverinfo="skip",
                       text=[f"<b>{r.Short}</b><br>{fmt(basis, r.V, short=True)}" for r in lab.itertuples()],
                       textfont=dict(size=11 if iso == "USA" else 12, color="#111"))
    return [t0, t1, t2]


def _pt_marker(pts, pz, scale, zm, cap_max) -> dict:
    return dict(size=[4 + 22 * math.sqrt(max(c, 0) / cap_max) for c in pts["Marcap"]], color=pz, colorscale=scale,
                cmin=-zm, cmax=zm, opacity=.9, line=dict(width=.8, color="rgba(20,20,20,.55)"))


def _layout(fig, iso: str, gj) -> None:
    if iso == "USA":
        fig.update_geos(scope="usa", projection_type="albers usa", showland=True, landcolor="rgba(128,128,128,.10)",
                        showlakes=False, bgcolor="rgba(0,0,0,0)", subunitcolor="rgba(255,255,255,.6)")
        h = 470
    else:
        fig.update_geos(projection_type="mercator", lataxis_range=[33.0, 38.75], lonaxis_range=[125.0, 130.0],
                        showland=True, landcolor="rgba(128,128,128,.12)", showcountries=True, countrycolor="rgba(128,128,128,.4)",
                        showocean=False, showcoastlines=True, coastlinecolor="rgba(128,128,128,.4)", resolution=50,
                        showframe=False, bgcolor="rgba(0,0,0,0)")
        if gj is not None:
            fig.update_geos(fitbounds="locations", visible=False)
        h = 600
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      dragmode="pan", clickmode="event+select")


def _add_player(fig, frames: list, start: str) -> None:
    """날짜 슬라이더(끌어서 넘기기)와 재생 버튼. 브라우저 안에서 넘어가므로 다시 계산하지 않는다.
    frames: [(이름 'YYYY-MM-DD', [trace, ...]), ...]"""
    import plotly.graph_objects as go

    fig.frames = [go.Frame(name=nm, data=tr, traces=[0, 1, 2]) for nm, tr in frames]
    names = [nm for nm, _ in frames]
    anim = dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
    dur = 1300 if len(names) <= 8 else (400 if len(names) <= 30 else 220)
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=0.0, xanchor="left", yanchor="top", pad=dict(t=30, r=6),
                          showactive=False, buttons=[
                              dict(label="▶", method="animate",
                                   args=[None, dict(mode="immediate", fromcurrent=True, frame=dict(duration=dur, redraw=True),
                                                    transition=dict(duration=0))]),
                              dict(label="❚❚", method="animate", args=[[None], anim])])],
        sliders=[dict(active=names.index(start) if start in names else len(names) - 1, x=0.1, y=0.0, len=0.9, yanchor="top",
                      pad=dict(t=8), currentvalue=dict(prefix="", font=dict(size=13)), ticklen=3,
                      steps=[dict(label=nm[5:] if len(nm) == 10 else nm, name=nm, method="animate", args=[[nm], anim]) for nm in names])],
        margin=dict(l=0, r=0, t=0, b=80))
    fig.update_layout(height=(fig.layout.height or 500) + 80)


# ----------------------------------------------------------------------
# 값 준비
# ----------------------------------------------------------------------
def _cap_txt(d: pd.DataFrame, cap_fmt) -> pd.DataFrame:
    return d.assign(CapTxt=[cap_fmt(v, "cap") for v in d["Marcap"]])


def _scores_for(d: pd.DataFrame, domestic: bool, p: str) -> list:
    """종합 점수 (스크리너와 같은 저장소를 쓴다: 스크리너에서 계산한 종목은 바로 보임)."""
    import screener

    store = st.session_state.get("sc_scores", {})
    out = []
    for s in d["ysym"].astype(str):
        r = store.get((domestic, s))
        out.append(screener._total(r)[0] if r and not r.get("err") else np.nan)
    return out


def _score_button(d: pd.DataFrame, domestic: bool, p: str) -> None:
    import screener

    store = st.session_state.get("sc_scores", {})
    todo = d[[(domestic, s) not in store for s in d["ysym"].astype(str)]]
    if todo.empty:
        return
    est = max(1, round(len(todo) * (6 if domestic else 4) / 3 / 60))
    c1, c2 = st.columns([3, 1])
    c1.caption(f"종합 점수가 아직 없는 종목 {len(todo):,}개 · 계산에 약 {est}분 (섹터를 고르면 줄어듭니다). 스크리너에서 계산한 종목은 함께 쓰입니다.")
    if c2.button(f"점수 계산 ({len(todo):,}종목)", key=f"{p}_score_btn", type="primary", use_container_width=True):
        jobs = screener._score_jobs(todo.assign(Symbol=todo["ysym"], Code=todo["Key"] if domestic else todo["ysym"]), domestic)
        screener._run_scores(jobs, domestic)
        st.rerun()


def _pe_button(d: pd.DataFrame, domestic: bool, p: str) -> None:
    need = gd.pe_missing(list(d["ysym"].astype(str)))
    if not need:
        return
    c1, c2 = st.columns([3, 1])
    c1.caption(f"PER이 아직 없는 종목 {len(need):,}개 · Yahoo에서 받는 데 수십 초 걸립니다 (하루 동안 다시 씀).")
    if c2.button(f"PER 불러오기 ({len(need):,}종목)", key=f"{p}_pe_btn", type="primary", use_container_width=True):
        bar = st.progress(0.0, text="준비 중...")
        gd.fetch_pe(need, domestic, progress=lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t))
        bar.empty()
        st.rerun()


def _values(d: pd.DataFrame, basis: str, per: str, ret: pd.DataFrame, idx: dict, iso: str, p: str) -> pd.DataFrame:
    """d에 V(색칠 값)와 Ret(고른 기간 수익률)을 붙인다. 거래대금은 A1·A0, PER은 PE 열도."""
    d = d.copy()
    d["Ret"] = ret[per]
    domestic = iso == "KOR"
    if basis == "수익률":
        d["V"] = d["Ret"]
    elif basis == "지수 대비":
        d["V"] = d["Ret"] - idx.get(per, np.nan)
    elif basis == "거래대금 증가율":
        a = gd.amount_ratio(d)
        d["A1"], d["A0"] = a["A1"], a["A0"]
        d["V"] = (d["A1"] / d["A0"] - 1) * 100
    elif basis == "PER":
        d["PE"] = gd.pe_values(d["ysym"].astype(str))
        d["V"] = d["PE"].where(d["PE"] > 0)
    else:
        d["V"] = _scores_for(d, domestic, p)
    return d


def _label(basis: str, per: str, iso: str, idx: dict) -> tuple:
    """(값 이름, 색 설명)."""
    inm = INDEX[iso][1]
    if basis == "수익률":
        return f"{per} 수익률", f"색 = {per} 수익률 (빨강 상승 · 파랑 하락, 구역은 시가총액 가중)"
    if basis == "지수 대비":
        return (f"{inm} 대비", f"색 = {per} 수익률 − {inm} {per} 수익률 ({fmt('수익률', idx.get(per))}) · 빨강 = 지수보다 잘함, 파랑 = 못함")
    if basis == "거래대금 증가율":
        return ("거래대금 증가율", "색 = 최근 거래일 거래대금 ÷ 직전 20거래일 평균 − 1 · 주황 = 평소보다 많이 거래, 회색 = 적게 "
                                 "(장중엔 오늘 거래가 덜 쌓여 낮게 나올 수 있음)")
    if basis == "PER":
        return ("PER", "색 = 후행 PER, 나라 안 중앙값 대비 · 청록 = 싸다, 갈색 = 비싸다 · 구역 PER = Σ시가총액 ÷ Σ순이익 (적자 제외)")
    return ("종합 점수", "색 = 기업 리서치 종합 점수 (6개월 상승 우세 정도, 50 = 중립) · 빨강 = 높음, 파랑 = 낮음")


# ----------------------------------------------------------------------
# 한 나라 화면
# ----------------------------------------------------------------------
def _controls(iso: str, p: str, sectors_list: list) -> tuple:
    """(색 기준, 섹터, 날짜 슬라이더 기간 또는 None)."""
    basis = ui.seg("색 기준", gd.BASES, default="수익률", key=f"{p}_basis") or "수익률"
    try:
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
    except TypeError:
        c1, c2 = st.columns([3, 2])
    sec = c1.selectbox("섹터", ["전체 섹터"] + sectors_list, key=f"{p}_sector", label_visibility="collapsed")
    rng = None
    if basis in gd.PERIOD_BASES:
        if c2.toggle("날짜별로 보기", key=f"{p}_play", help="고른 기간의 시작일부터 날짜마다 누적 수익률이 어떻게 변했는지, "
                                                         "지도 아래 슬라이더를 끌어서 봅니다 (▶ 누르면 자동으로 넘어감)"):
            per = st.session_state.get("mk_world_per") or "1일"
            dflt = {"3개월": "3개월", "연초 이후": "연초 이후", "1년": "1년"}.get(per, "1개월")
            rng = ui.seg("슬라이더 기간", list(gd.PATH_RANGES), default=dflt, key=f"{p}_rng") or dflt
    return basis, (None if sec == "전체 섹터" else sec), rng


def _regions(d: pd.DataFrame, basis: str, iso: str, cap_fmt) -> pd.DataFrame:
    reg = gd.region_values(d, basis)
    names = ({k: (f"{v[1]} ({v[0]})", v[0], v[2], v[3]) for k, v in gd.US_STATES.items()} if iso == "USA"
             else {k: (v[0], k, v[1], v[2]) for k, v in gd.SIDO.items()})
    blank = ("", "", np.nan, np.nan)
    reg["Name"] = [names.get(k, (k,) + blank[1:])[0] for k in reg.index]
    reg["Short"] = [names.get(k, (k, k) + blank[2:])[1] for k in reg.index]
    reg["lat"] = [names.get(k, blank)[2] for k in reg.index]
    reg["lon"] = [names.get(k, blank)[3] for k in reg.index]
    reg["CapTxt"] = [cap_fmt(v, "cap") for v in reg["Cap"]]
    if iso == "USA":
        reg["Abbr"] = [gd.US_STATES[k][0] if k in gd.US_STATES else None for k in reg.index]
        reg = reg[reg["Abbr"].notna()]
    return reg


MAP_CONFIG = {"displayModeBar": False, "scrollZoom": True, "doubleClick": "reset"}


def _date_frames(iso, d, reg, gj, basis, rng, close, idx_sym, ref, sel, lab_index, cap_fmt):
    """날짜마다 '범위 시작일부터 그날까지 누적 수익률'로 장면을 만든다. 반환 (색 범위, [(날짜, traces), ...]) 또는 None."""
    import plotly.graph_objects as go

    with st.spinner(f"{rng} 날짜별 수익률을 계산하는 중 (전 종목 일봉을 처음 받을 때는 1~2분 걸릴 수 있습니다)..."):
        ds, cum = gd.stock_path(d, rng)
        ipath = gd.index_path(close, idx_sym, rng, ds) if basis == "지수 대비" else None
    if len(ds) == 0 or cum.empty:
        st.caption("일봉을 받지 못해 날짜별로 보기를 할 수 없습니다.")
        return None
    scenes = []
    for k, dt in enumerate(ds):
        v = cum.iloc[:, k] - (float(ipath.iloc[k]) if ipath is not None else 0.0)
        dq = d.assign(V=v.values, Ret=cum.iloc[:, k].values)
        rq = _regions(dq, basis, iso, cap_fmt).reindex(reg.index)  # 구역 순서를 고정 (클릭 번호가 어긋나지 않게)
        rq[["Name", "Short", "lat", "lon", "CapTxt", "N", "Top"]] = reg[["Name", "Short", "lat", "lon", "CapTxt", "N", "Top"]]
        if iso == "USA":
            rq["Abbr"] = reg["Abbr"]
        scenes.append((f"{dt:%Y-%m-%d}", rq, dq.dropna(subset=["plat", "plon"]).reset_index(drop=True)))
    zs = [z for _n, rq, pq in scenes for z in list(to_z(basis, rq["V"], ref)) + list(to_z(basis, pq["V"], ref))]
    zm = z_range(basis, zs)
    start = scenes[0][0]
    word = "수익률" if basis == "수익률" else f"{INDEX[iso][1]} 대비"
    frames = []
    for k, (nm, rq, pq) in enumerate(scenes):
        tr = _traces(iso, rq, pq, gj, basis, f"{start}~{nm[5:]} {word}", ref, zm, sel, lab_index, static_pts=True)
        if k < len(scenes) - 1:  # 점은 색만 바꾼다 (글자까지 다시 보내면 너무 무거움)
            cap_max = float(pq["Marcap"].max()) if len(pq) else 1.0
            tr[1] = go.Scattergeo(marker=_pt_marker(pq, _nz(to_z(basis, pq["V"], ref)), SCALES[basis], zm, cap_max))
        frames.append((nm, tr))
    return zm, frames


def _render(iso: str, inside: pd.DataFrame, outside: pd.DataFrame, gj, per: str, cap_fmt, note: str, close) -> None:
    import market

    ss = st.session_state
    p = f"gm_{iso.lower()}"
    basis, sec, rng = _controls(iso, p, gd.sectors_by_cap(inside, outside))
    if sec:
        inside, outside = inside[inside["Sector"] == sec], outside[outside["Sector"] == sec]
    if inside.empty:
        st.info("고른 섹터에 지도 안에 본사를 둔 종목이 없습니다.")
        return
    idx_sym = INDEX[iso][0]
    pers = [per]
    if basis == "PER":
        _pe_button(pd.concat([inside, outside]), iso == "KOR", p)
    if basis == "종합 점수":
        _score_button(pd.concat([inside, outside]), iso == "KOR", p)
    quick = per == "1일" and not rng and basis in ("수익률", "지수 대비", "PER", "종합 점수")
    with st.spinner("값 계산 중..." if quick else "일봉을 받아 계산하는 중..."):
        ret_in, ret_out = gd.returns_for(inside, pers), gd.returns_for(outside, [per]) if len(outside) else None
        idx = gd.index_returns(close, idx_sym, pers)
        d = _cap_txt(_values(inside, basis, per, ret_in, idx, iso, p), cap_fmt)
        out = _cap_txt(_values(outside, basis, per, ret_out, idx, iso, p), cap_fmt) if len(outside) else outside
    label, desc = _label(basis, per, iso, idx)
    ref = float(np.nanmedian(d["V"])) if basis == "PER" and d["V"].notna().any() else np.nan
    reg = _regions(d, basis, iso, cap_fmt)
    pts = d.dropna(subset=["plat", "plon"]).reset_index(drop=True)
    sel = ss.get(f"{p}_region")
    sel = sel if sel in reg.index else None
    lab_index = [k for k in reg.head(12 if iso == "USA" else 17).index if reg.loc[k, "V"] == reg.loc[k, "V"]]

    import plotly.graph_objects as go

    frames = _date_frames(iso, d, reg, gj, basis, rng, close, idx_sym, ref, sel, lab_index, cap_fmt) if rng else None
    if frames:
        zm, frames = frames
        fig = go.Figure(data=frames[-1][1])
        _layout(fig, iso, gj)
        _add_player(fig, frames, frames[-1][0])
    else:
        zm = z_range(basis, list(to_z(basis, reg["V"], ref)) + list(to_z(basis, pts["V"], ref)))
        fig = go.Figure(data=_traces(iso, reg, pts, gj, basis, label, ref, zm, sel, lab_index))
        _layout(fig, iso, gj)

    hit = ui.new_click(ui.chart_points(fig, f"{p}_map", config=MAP_CONFIG), f"{p}_map")
    if hit:
        cn, i = hit
        if cn == 0 and i < len(reg):
            k = reg.index[i]
            ss[f"{p}_region"] = None if k == sel else k
            ss.pop(f"{p}_point", None)
            st.rerun()
        if cn == 1 and i < len(pts):
            ss[f"{p}_point"] = str(pts.loc[i, "Key"])
            ss[f"{p}_region"] = pts.loc[i, "Region"]
            st.rerun()
        if cn == 2 and i < len(lab_index):
            k = lab_index[i]
            ss[f"{p}_region"] = None if k == sel else k
            ss.pop(f"{p}_point", None)
            st.rerun()
    if frames:
        st.caption(f"색 = {rng} 시작일부터 슬라이더 날짜까지의 누적 {'수익률' if basis == '수익률' else INDEX[iso][1] + ' 대비 초과수익'} "
                   "(빨강 상승 · 파랑 하락) · 슬라이더를 끌어서 날짜를 넘기고 ▶로 자동 재생 · 아래 카드·목록은 세계 지도에서 고른 기간 기준")
    else:
        st.caption(desc)
    st.caption("지도: 스크롤하면 그 위치를 중심으로 확대·축소, 끌어서 이동, 두 번 누르면 처음 크기")
    st.caption(note + (f" · 섹터: {sec}" if sec else ""))
    _detail(iso, p, d, out, reg, basis, per, label, cap_fmt)


def _region_name(reg, k, basis: str) -> str:
    v = reg.loc[k, "V"]
    return f"{reg.loc[k, 'Name']} · {int(reg.loc[k, 'N'])}종목" + (f" · {fmt(basis, v)}" if v == v else "")


def _detail(iso, p, d, out, reg, basis, per, label, cap_fmt) -> None:
    """지도 아래: 구역 고르기 · 고른 종목 · 고른 구역 (카드·시장 지도·목록) · 구역별 표."""
    import market

    ss = st.session_state
    opts = ["전체"] + list(reg.index) + (["해외 본사"] if len(out) else [])
    region = ss.get(f"{p}_region")
    cur = region if region in opts else "전체"
    ss[f"{p}_sel"] = cur

    def _on_pick():
        v = ss.get(f"{p}_sel")
        ss[f"{p}_region"] = None if v in (None, "전체") else v
        ss.pop(f"{p}_point", None)

    st.selectbox("구역", opts, key=f"{p}_sel", label_visibility="collapsed", on_change=_on_pick,
                 format_func=lambda k: "전체 구역 보기" if k == "전체" else (f"해외 본사 ({len(out)}종목)" if k == "해외 본사" else _region_name(reg, k, basis)))

    pk = ss.get(f"{p}_point")
    if pk is not None and (d["Key"].astype(str) == pk).any():
        r = d[d["Key"].astype(str) == pk].iloc[0]
        items = [(str(r["Label"]), fmt(basis, r["V"]), f"{label} · 본사 {r['Place']}", r["V"] - (50 if basis == "종합 점수" else 0)
                  if basis not in ("PER", "거래대금 증가율") else None)]
        if basis != "수익률":
            items.append((f"{per} 수익률", fmt("수익률", r["Ret"]), "", r["Ret"]))
        items.append(("시가총액", r["CapTxt"], str(r.get("Sector") or "")))
        ui.cards(items, cols=len(items))
        _chart_button(str(r["Label"]), r["Query"], key=f"{p}_pt")
        if st.button("선택 해제", key=f"{p}_clr"):
            ss.pop(f"{p}_point", None)
            st.rerun()

    if cur == "해외 본사":
        sub, rname = out, "해외 본사"
    elif cur in reg.index:
        sub, rname = d[d["Region"] == cur], reg.loc[cur, "Name"]
    else:
        sub = None
    if sub is not None and len(sub):
        rv = gd.agg_value(sub, basis)
        rr = sub.dropna(subset=["Ret"])
        wret = float(np.average(rr["Ret"], weights=rr["Marcap"])) if len(rr) and rr["Marcap"].sum() > 0 else np.nan
        up, dn = int((sub["Ret"] > 0).sum()), int((sub["Ret"] < 0).sum())
        items = [(f"{rname} {label}", fmt(basis, rv), "시가총액 가중" if basis in ("수익률", "지수 대비", "종합 점수") else "구역 합계 기준",
                  (rv - 50 if basis == "종합 점수" else rv) if basis not in ("PER", "거래대금 증가율") else None)]
        if basis != "수익률":
            items.append((f"{per} 수익률", fmt("수익률", wret), "시가총액 가중", wret))
        items += [("종목 수", f"{len(sub):,}", f"상승 {up} · 하락 {dn}", up - dn), ("시가총액 합계", cap_fmt(float(sub["Marcap"].sum()), "cap"), "")]
        ui.cards(items, cols=len(items))
        m = sub.copy()
        m["ChangesRatio"] = m["Ret"]
        if iso == "USA":
            m["FullName"], m["Name"] = m["Name"], m["Label"]
        if len(m) >= 2:
            market._treemap(m, f"{rname} 시장 지도", f"{rname} 본사 종목 · 칸 크기 = 시가총액 · 색 = {per} 수익률", cap_fmt,
                            height=420 if len(m) < 40 else 520)
        _list(sub, p, basis, per, label)
    else:
        exp, op = ui.more("구역별 표", key=f"{p}_more_tbl")
        if op:
            with exp:
                t = pd.DataFrame({"구역": reg["Name"], label: [fmt(basis, v) for v in reg["V"]], "종목 수": reg["N"].astype(int),
                                  "시가총액": reg["CapTxt"], "큰 종목": reg["Top"]})
                st.dataframe(t, use_container_width=True, hide_index=True)


def _chart_button(name: str, query: str, key: str) -> None:
    c1, _ = st.columns([1, 3])
    if c1.button(f"{name} 차트 보기", key=f"{key}_ch", use_container_width=True):
        ui.go("chart", "single_query", str(query))


def _list(sub: pd.DataFrame, p: str, basis: str, per: str, label: str) -> None:
    s = sub.sort_values("Marcap", ascending=False).reset_index(drop=True)
    out = pd.DataFrame({"종목": s["Label"], "본사": s["Place"]})
    if basis != "수익률":
        out[label] = [fmt(basis, v) for v in s["V"]]
    out[f"{per} 수익률(%)"] = s["Ret"].round(2)
    out["시가총액"], out["섹터"] = s["CapTxt"], s.get("Sector")
    i = ui.table_pick(out, f"{p}_tbl", use_container_width=True, hide_index=True, height=min(36 * (len(out) + 1) + 4, 420),
                      column_config={f"{per} 수익률(%)": st.column_config.NumberColumn(format="%+.2f")})
    if i is not None:
        r = s.iloc[i]
        _chart_button(str(r["Label"]), r["Query"], key=f"{p}_tbl")
    else:
        st.caption("행을 누르면 그 종목 차트로 바로 갈 수 있습니다.")


# ----------------------------------------------------------------------
# 나라별 진입점
# ----------------------------------------------------------------------
def render_us(per: str, close=None) -> None:
    import market

    ui.section("미국 지도", "점 = 본사 위치 (크기 = 시가총액) · 주 색 = 고른 기준 · 주나 점을 누르면 아래에 자세히")
    try:
        with st.spinner("본사 위치를 불러오는 중..."):
            inside, outside, n_exact = gd.us_frame()
    except Exception as e:  # noqa: BLE001
        st.caption(f"본사 위치를 불러오지 못했습니다: {e}")
        return
    note = (f"S&P 500 · 본사: 위키백과 S&P 500 목록, 좌표: 도시 문서 ({n_exact}/{len(inside)}곳 도시 좌표, 나머지는 주 중심)"
            + (f" · 해외 본사 {len(outside)}종목은 지도 밖 (아래 구역 목록의 '해외 본사')" if len(outside) else ""))
    _render("USA", inside, outside, None, per, market._usd, note, close)


def render_kr(snap: pd.DataFrame, per: str, close=None) -> None:
    import kr
    import market

    ui.section("한국 지도", "점 = 본사 위치 (크기 = 시가총액) · 시도 색 = 고른 기준 · 시도나 점을 누르면 아래에 자세히")
    if snap is None or snap.empty or not {"Code", "Name", "Marcap", "ChangesRatio"}.issubset(snap.columns):
        st.caption("국내 종목 시세를 불러오지 못했습니다.")
        return
    key = kr.dart_key()
    try:
        with st.spinner("본사 위치를 불러오는 중" + (" (DART 주소는 처음 한 번만 조금 걸립니다)..." if key else "...")):
            inside, outside, n_exact = gd.kr_frame(snap, None, key)
    except Exception as e:  # noqa: BLE001
        st.caption(f"본사 지역 정보를 불러오지 못했습니다: {e}")
        return
    gj = gd.kr_geojson()
    how = (f"시·군·구 위치 {n_exact}/{len(inside)}곳 (DART 기업개황 주소)" if key else
           "시도 안에 흩어 찍음 · OpenDART 키를 넣으면 시·군·구까지 정확히 찍습니다")
    note = (f"코스피·코스닥 전체 {len(inside) + len(outside):,}종목 (우선주 제외) · 본사 시도: KRX 상장회사 정보 · {how}"
            + ("" if gj is not None else " · 시도 경계를 받지 못해 원으로 표시")
            + (f" · 해외 본사·지역 미상 {len(outside)}종목" if len(outside) else ""))
    _render("KOR", inside, outside, gj, per, market._krw, note, close)


def render(country: str, snap=None, close=None) -> None:
    per = st.session_state.get("mk_world_per") or "1일"
    if country == "미국":
        render_us(per, close)
    elif country == "한국":
        render_kr(snap, per, close)
