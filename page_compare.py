"""수익률 비교 페이지."""
from common import *  # noqa: F401,F403


def render_compare(listing: pd.DataFrame) -> None:
    ordered = listing.assign(_pf=~listing["Code"].astype(str).str.endswith("0")).sort_values(["_pf", "Marcap"], ascending=[True, False], na_position="last")  # 보통주·시가총액 큰 순 (검색해도 삼성전자가 먼저)
    options = {f"{r.Name} ({r.Code})": r.Code for r in ordered.itertuples()}
    names = {r.Code: r.Name for r in ordered.itertuples()}

    _ok = set(options)
    if isinstance(st.session_state.get("cmp_selected"), list):  # 주소로 들어온 값 중 목록에 없는 종목은 뺀다
        st.session_state["cmp_selected"] = [x for x in st.session_state["cmp_selected"] if x in _ok]
    c1, c2, c3 = _row([3, 2, 1])
    selected = c1.multiselect(
        "국내 종목 (이름을 입력하면 검색됩니다)",
        list(options.keys()), key="cmp_selected", placeholder="종목 이름이나 코드를 입력하세요",
        on_change=_mirror, args=("cmp_selected", "pf_selected"),
    )
    extra = c2.text_input(
        "해외 티커 · 선물 · 종목 추가 (쉼표로 구분)", key="cmp_extra",
        placeholder="예: AAPL, NVDA, 원유 선물, 금 선물",
        on_change=_mirror, args=("cmp_extra", "pf_extra"),
    )
    with c3:
        with ui.popover("설정", icon=":material/tune:"):
            fut_sel = st.multiselect(
                "해외 선물 (목록에서 선택)", list(FUT_OPTIONS), key="cmp_fut",
                on_change=_mirror, args=("cmp_fut", "pf_fut"),
            )
            rate_sel = rate_controls("cmp")
    period = period_input("cmp_period", mirror="pf_period", presets=True)

    # 선택 항목 정리 (중복 제거)
    items: list[dict] = []
    seen: set[str] = set()
    for label in selected:
        code = options[label]
        if code not in seen:
            seen.add(code)
            items.append({"code": code, "name": names[code], "domestic": True})
    for it in _futures_items(fut_sel):
        if it["code"] not in seen:
            seen.add(it["code"])
            items.append(it)
    for token in [t.strip() for t in extra.split(",") if t.strip()]:
        cands = find_candidates(token, listing)
        if not cands:
            st.warning(f"'{token}' 종목을 찾지 못했습니다.")
            continue
        c = cands[0]
        if c["code"] not in seen:
            seen.add(c["code"])
            items.append({"code": c["code"], "name": c["name"], "domestic": c["domestic"]})

    if not items:
        st.info("비교할 종목을 선택하세요. 종목 2개 이상부터 비교 효과가 있습니다.")
        return
    if period is None:
        st.info("기간의 시작일과 종료일을 모두 선택하세요.")
        return
    start_d, end_d = period

    prefetch_prices([it["code"] for it in items], start_d.isoformat())  # 여러 종목을 동시에 받아 둠
    fig = go.Figure()
    rows = []
    for i, it in enumerate(items):
        try:
            df = load_prices(it["code"], start_d.isoformat(), end_d.isoformat())
        except Exception as e:  # noqa: BLE001
            st.warning(friendly_error(e, it["name"]) + " 이 종목은 제외하고 비교합니다.")
            continue
        s = df["Close"].dropna() if df is not None and not df.empty else pd.Series(dtype=float)
        if s.empty or float(s.iloc[0]) == 0:
            st.warning(f"{it['name']}: 선택한 기간에 데이터가 없습니다.")
            continue

        base = float(s.iloc[0])
        ret = (s / base - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=ret.index, y=ret, mode="lines", name=it["name"],
                line=dict(color=series_color(i), width=2),
                hovertemplate="%{y:+.2f}%<extra>" + it["name"] + "</extra>",
            )
        )
        rd = s.pct_change().dropna()
        rows.append({
            "종목": it["name"],
            "마지막 종가": fmt_price(float(s.iloc[-1]), it["domestic"]),
            "기간 수익률(%)": round(float(ret.iloc[-1]), 2),
            "연 변동성(%)": round(float(rd.std(ddof=1)) * TRADING_DAYS ** 0.5 * 100, 1) if len(rd) > 2 else None,
            "최대 낙폭(%)": round(float((s / s.cummax() - 1).min()) * 100, 1),
            "기간 내 최고(%)": round(float(ret.max()), 2),
            "기간 내 최저(%)": round(float(ret.min()), 2),
            "시작일": s.index[0].strftime("%Y-%m-%d"),
        })

    if not rows:
        return

    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#718096")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="수익률 (기간 시작일 = 0%)", ticksuffix="%")
    fig.update_xaxes(hoverformat="%Y-%m-%d")
    if rate_sel:
        add_rates(fig, rate_sel, start_d, end_d, 2)
    fig.update_layout(
        height=520, hovermode="x unified", font=dict(size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    summary = pd.DataFrame(rows).sort_values("기간 수익률(%)", ascending=False)
    ui.section("기간 수익률 순위")
    from glossary import col_help

    cfg = col_help(summary.columns)
    for c in ("기간 수익률(%)", "기간 내 최고(%)", "기간 내 최저(%)"):
        cfg[c] = st.column_config.NumberColumn(c, format="%+.2f%%")
    cfg["연 변동성(%)"] = st.column_config.NumberColumn("연 변동성(%)", format="%.1f%%")
    cfg["최대 낙폭(%)"] = st.column_config.NumberColumn("최대 낙폭(%)", format="%.1f%%")
    st.dataframe(ui.signed_style(summary, ["기간 수익률(%)", "기간 내 최고(%)", "기간 내 최저(%)"]), use_container_width=True, hide_index=True, column_config=cfg)
    st.caption("수익률은 각 종목의 기간 첫 거래일 종가를 0%로 놓고 계산한 값입니다. 배당은 반영하지 않은 가격 기준이며, 해외 종목은 현지 통화(달러 등) 기준이라 환율 변동은 들어 있지 않습니다.")
    if len(items) > 8:
        st.caption("종목이 많아 선이 겹쳐 보일 수 있습니다. 범례의 종목 이름을 누르면 그 선을 숨기거나 다시 볼 수 있어요.")


# ======================================================================
# 탭 3: 포트폴리오 최적화
# ======================================================================
