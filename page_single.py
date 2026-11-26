"""주가 조회 페이지: 차트·기술적 지표·수익률/위험 지표."""
from common import *  # noqa: F401,F403


# ======================================================================
# 탭 1: 주가 조회
# ======================================================================
IND_OPTS = ["이동평균선", "볼린저밴드", "RSI", "MACD", "일목균형표", "피보나치", "이동평균 이격(σ)"]
IND_DEFAULTS = {"single_ma_n": "5, 20, 60, 120", "single_bb_n": 20, "single_bb_k": 2.0, "single_rsi_n": 14, "single_rsi_hi": 70,
                "single_rsi_lo": 30, "single_macd_f": 12, "single_macd_s": 26, "single_macd_g": 9,
                "single_ichi_t": 9, "single_ichi_k": 26, "single_ichi_s": 52, "single_ichi_d": 26,
                "single_fib_lv": "0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.618, 2.618", "single_fib_sens": 10.0, "single_fib_nth": 1, "single_sg_n": 20}


def _clean_ind_state() -> None:
    """예전 버전의 지표 이름(괄호 안 숫자 포함)이 남아 있으면 새 이름으로 바꾼다."""
    v = st.session_state.get("single_ind")
    if isinstance(v, (list, tuple)):
        st.session_state["single_ind"] = [o for o in IND_OPTS if any(str(x).startswith(o) for x in v)]


def _reset_ind() -> None:
    for k, v in IND_DEFAULTS.items():
        st.session_state[k] = v


def _ind_settings(inds: list) -> dict:
    """선택한 보조지표의 기간·기준값 입력칸 (기본값이 채워져 있고, 지우고 원하는 값을 넣으면 됨)."""
    import ta

    P = {}
    if not inds:
        return P
    st.markdown("**보조지표 설정**")
    if IND_OPTS[0] in inds:
        txt = st.text_input("이동평균 기간 (일, 쉼표로 구분)", value=IND_DEFAULTS["single_ma_n"], key="single_ma_n",
                            help="예: 5, 20, 60, 120. 2~400일, 최대 8개")
        P["ma"] = ta.parse_periods(txt)
    if IND_OPTS[1] in inds:
        a, b = st.columns(2)
        P["bb_n"] = int(a.number_input("볼린저 기간", 2, 400, IND_DEFAULTS["single_bb_n"], step=1, key="single_bb_n"))
        P["bb_k"] = float(b.number_input("표준편차 배수", 0.5, 5.0, IND_DEFAULTS["single_bb_k"], step=0.5, key="single_bb_k"))
    if IND_OPTS[2] in inds:
        a, b, c = st.columns(3)
        P["rsi_n"] = int(a.number_input("RSI 기간", 2, 200, IND_DEFAULTS["single_rsi_n"], step=1, key="single_rsi_n"))
        P["rsi_hi"] = int(b.number_input("과매수", 50, 100, IND_DEFAULTS["single_rsi_hi"], step=5, key="single_rsi_hi"))
        P["rsi_lo"] = int(c.number_input("과매도", 0, 50, IND_DEFAULTS["single_rsi_lo"], step=5, key="single_rsi_lo"))
    if IND_OPTS[3] in inds:
        a, b, c = st.columns(3)
        P["macd_f"] = int(a.number_input("MACD 단기", 2, 200, IND_DEFAULTS["single_macd_f"], step=1, key="single_macd_f"))
        P["macd_s"] = int(b.number_input("MACD 장기", 3, 400, IND_DEFAULTS["single_macd_s"], step=1, key="single_macd_s"))
        P["macd_g"] = int(c.number_input("시그널", 2, 100, IND_DEFAULTS["single_macd_g"], step=1, key="single_macd_g"))
        if P["macd_f"] >= P["macd_s"]:
            st.caption("단기 기간이 장기 기간보다 짧아야 합니다. 두 값을 바꿔서 계산합니다.")
            P["macd_f"], P["macd_s"] = min(P["macd_f"], P["macd_s"]), max(P["macd_f"], P["macd_s"]) + (P["macd_f"] == P["macd_s"])
    if IND_OPTS[4] in inds:
        a, b, c, d = st.columns(4)
        P["ichi"] = (int(a.number_input("전환선", 1, 200, IND_DEFAULTS["single_ichi_t"], step=1, key="single_ichi_t")),
                     int(b.number_input("기준선", 1, 400, IND_DEFAULTS["single_ichi_k"], step=1, key="single_ichi_k")),
                     int(c.number_input("선행스팬2", 1, 400, IND_DEFAULTS["single_ichi_s"], step=1, key="single_ichi_s")),
                     int(d.number_input("이동", 1, 200, IND_DEFAULTS["single_ichi_d"], step=1, key="single_ichi_d")))
    if IND_OPTS[5] in inds:
        import drawings as dw

        kind = st.radio("피보나치 종류", dw.FIB_KINDS, horizontal=True, key="single_fib_kind")
        lv = st.text_input("레벨 (쉼표로 구분, 소수·음수 가능)", value=IND_DEFAULTS["single_fib_lv"], key="single_fib_lv",
                           help="예: -0.618, 0, 0.382, 0.5, 0.618, 1, 1.618, 2.618, 4.236")
        mode = st.radio("기준점", ["자동", "직접 지정"], horizontal=True, key="single_fib_mode")
        F = {"kind": kind, "levels": dw.parse_levels(lv), "mode": mode, "dates": []}
        if mode == "자동":
            a, b = st.columns(2)
            F["sens"] = float(a.number_input("민감도 (%)", 0.5, 50.0, IND_DEFAULTS["single_fib_sens"], step=0.5, format="%.1f",
                                             key="single_fib_sens", help="이만큼 반대로 움직이면 고점·저점으로 봅니다. 작을수록 짧은 파동"))
            F["nth"] = int(b.number_input("몇 번째 최근 파동", 1, 20, IND_DEFAULTS["single_fib_nth"], step=1, key="single_fib_nth"))
        else:
            names = (["1 (추세선 시작)", "2 (추세선 끝)", "3 (채널 폭)"] if kind == dw.FIB_KINDS[0]
                     else ["1 (파동 시작)", "2 (파동 끝)", "3 (되돌림)"] if kind == dw.FIB_KINDS[1] else ["1 (시작)", "2 (끝)"])
            cols = st.columns(len(names))
            F["dates"] = [col.date_input(nm, value=None, key=f"single_fib_d{i}") for i, (col, nm) in enumerate(zip(cols, names))]
            F["sens"], F["nth"] = IND_DEFAULTS["single_fib_sens"], 1
        P["fib"] = F
    if IND_OPTS[6] in inds:
        P["sg_n"] = int(st.number_input("이격 기준 이동평균 기간 (일)", 5, 400, IND_DEFAULTS["single_sg_n"], step=1, key="single_sg_n",
                                        help="종가가 이 기간 이동평균보다 이 기간 표준편차의 몇 배만큼 위·아래인지 계산합니다"))
    st.button("기본값으로 되돌리기", key="single_ind_reset", on_click=_reset_ind)
    return P


def _ohlc_base(code: str, start_d, end_d, weekly: bool, longest: int = 120):
    """일목균형표 계산용 고가·저가·종가 (앞쪽 계산 구간 여유 포함)."""
    back = int(longest * (7 if weekly else 1.5)) + 60
    try:
        ext = load_prices(code, (pd.Timestamp(start_d) - pd.Timedelta(days=back)).date().isoformat(), end_d.isoformat())
    except Exception:  # noqa: BLE001
        return None
    if ext is None or ext.empty or not {"High", "Low", "Close"} <= set(ext.columns):
        return None
    ext = ext[["High", "Low", "Close"]].dropna()
    if weekly:
        ext = ext.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    return ext


def _indicator_base(code: str, start_d, end_d, weekly: bool, longest: int = 120):
    """보조지표 계산용 종가: 앞쪽 계산 구간(가장 긴 기간)이 비지 않게 조회 시작일보다 넉넉히 앞에서부터."""
    back = int(longest * (7 if weekly else 1.5)) + 60  # 거래일 → 달력일 여유
    try:
        ext = load_prices(code, (pd.Timestamp(start_d) - pd.Timedelta(days=back)).date().isoformat(),
                          end_d.isoformat())
    except Exception:  # noqa: BLE001
        return None
    if ext is None or ext.empty:
        return None
    c = ext["Close"].dropna()
    return c.resample("W-FRI").last().dropna() if weekly else c



def _sigma_chart(s: pd.Series, n: int, x0, title: str):
    import ta

    g = ta.sigma_gap(s, n)
    g = g[g.index >= x0].dropna()
    if g.empty:
        return None, float("nan")
    cur = float(g.iloc[-1])
    f = go.Figure(go.Scatter(x=g.index, y=g, mode="lines", line=dict(color="#4A5568", width=1.4),
                             hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.2f}σ<extra></extra>", name=title))
    for lv, col, dash in ((2, UP_COLOR, "dot"), (1, "#A0AEC0", "dot"), (0, "#718096", "solid"), (-1, "#A0AEC0", "dot"), (-2, DOWN_COLOR, "dot")):
        f.add_hline(y=lv, line_width=1, line_dash=dash, line_color=col)
    f.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], hoverformat="%Y-%m-%d")
    f.update_yaxes(title_text="σ", tickvals=[-3, -2, -1, 0, 1, 2, 3], ticksuffix="σ", zeroline=False)
    f.update_layout(height=230, margin=dict(l=10, r=10, t=28, b=10), font=dict(size=11), showlegend=False,
                    title=dict(text=title, font=dict(size=13), x=0.0))
    return f, cur


def _sigma_state(v: float) -> str:
    if v != v:
        return "-"
    a = abs(v)
    return "평균 부근" if a < 1 else ("평균보다 높음" if v > 0 else "평균보다 낮음") if a < 2 else ("평균보다 크게 높음" if v > 0 else "평균보다 크게 낮음")


def _sigma_section(chosen: dict, start_d, end_d, n: int) -> None:
    """이동평균 이격(σ): 종목은 종목대로, 소속 시장 지수는 지수대로 따로 보여준다."""
    st.markdown(f"**이동평균 이격 ({n}일 기준, σ)**")
    bcode, bname = benchmark_for(chosen.get("market", ""), chosen["domestic"])
    items = [(chosen["label"].split(" (")[0], chosen["code"], "종목")]
    if bcode != chosen["code"]:
        items.append((bname, bcode, "지수"))
    back = (pd.Timestamp(start_d) - pd.Timedelta(days=int(n * 1.5) + 60)).date().isoformat()
    cols = st.columns(len(items))
    for col, (nm, code, kind) in zip(cols, items):
        with col:
            try:
                d = load_prices(code, back, end_d.isoformat())
                s = d["Close"].dropna() if d is not None and not d.empty else None
            except Exception:  # noqa: BLE001
                s = None
            if s is None or len(s) <= n:
                st.caption(f"{nm}: 이동평균을 계산할 시세가 부족합니다.")
                continue
            f, cur = _sigma_chart(s, n, pd.Timestamp(start_d), f"{kind} · {nm}")
            if f is None:
                st.caption(f"{nm}: 조회 기간 안에 계산된 값이 없습니다.")
                continue
            st.metric(f"{nm} 현재", f"{cur:+.2f}σ", _sigma_state(cur), delta_color="off")
            st.plotly_chart(f, use_container_width=True)
    st.caption(f"(종가 − {n}일 이동평균) ÷ {n}일 가격 표준편차. 0 = 이동평균 위치, ±2σ 밖은 최근 {n}일 흐름에서 많이 벗어난 상태입니다. "
               "과거 데이터로 계산한 참고 값이며 매매 판단이 아닙니다.")


def render_single(listing: pd.DataFrame) -> None:
    c1, c2 = _row([5, 1])
    query = c1.text_input(
        "종목 검색", key="single_query", label_visibility="collapsed",
        placeholder="종목명 (예: 삼성전자) 또는 종목코드 / 해외 티커 (예: AAPL, 원유 선물)",
    )
    with c2:
        with ui.popover("차트 설정", icon=":material/tune:"):
            chart_type = st.radio("차트 종류", ["선", "캔들"], horizontal=True, key="single_type")
            basis = st.radio("표시 기준", ["주가", "수익률"], horizontal=True, key="single_basis")
            _clean_ind_state()
            inds = st.multiselect("보조지표", IND_OPTS, key="single_ind", placeholder="이동평균선, 볼린저밴드, RSI, MACD, 일목균형표, 피보나치, 이동평균 이격(σ)")
            IP = _ind_settings(inds)
            rate_sel = rate_controls("single")
    period = period_input("single_period")

    if not query.strip():
        st.info("종목명이나 코드를 입력하세요. 예: 삼성전자, 005930, AAPL, 원유 선물")
        return
    if period is None:
        st.info("기간의 시작일과 종료일을 모두 선택하세요.")
        return
    start_d, end_d = period

    candidates = find_candidates(query, listing)
    if not candidates:
        _sim = similar_names(query, listing)
        st.warning("일치하는 종목이 없습니다. 종목명이나 코드를 다시 확인해 주세요. 예: 삼성전자, 005930, AAPL"
                   + (f" · 혹시 이 종목인가요? {', '.join(_sim)}" if _sim else ""))
        return

    if len(candidates) > 1:
        labels = [c["label"] for c in candidates]
        chosen = candidates[labels.index(st.selectbox("종목 선택", labels, key="single_pick"))]
    else:
        chosen = candidates[0]

    # 종목과 CAPM 벤치마크 지수를 동시에 받아 둔다 (종목 하나는 한 번만 받고 기간·지표·52주는 잘라 씀)
    prefetch_prices([chosen["code"], benchmark_for(chosen.get("market", ""), chosen["domestic"])[0]],
                    min(start_d, kst_today() - timedelta(days=372)).isoformat())
    try:
        df = load_prices(chosen["code"], start_d.isoformat(), end_d.isoformat())
    except Exception as e:  # noqa: BLE001
        st.error(friendly_error(e, chosen.get("name", "")))
        return
    if df is None or df.empty:
        st.warning("선택한 기간에 데이터가 없습니다. 기간을 바꿔 보세요.")
        return

    domestic = chosen["domestic"]
    # 최근 1년(52주) 시세: 현재가·전일 대비·52주 범위용
    today = kst_today()
    y1 = df
    if (pd.Timestamp(df.index[0]) > pd.Timestamp(today - timedelta(days=360))) or end_d < today:
        try:
            y1 = load_prices(chosen["code"], (today - timedelta(days=372)).isoformat(), today.isoformat())
        except Exception:  # noqa: BLE001
            y1 = df
        if y1 is None or y1.empty:
            y1 = df
    y1 = y1[y1.index >= y1.index[-1] - pd.Timedelta(days=365)]
    last = y1.iloc[-1]
    px = float(last["Close"])
    prev = float(y1["Close"].iloc[-2]) if len(y1) >= 2 else float("nan")
    chg = px - prev
    pct_d = chg / prev * 100 if prev == prev and prev else float("nan")
    name_part = chosen["label"].split(" (")[0]
    code_part = chosen["label"][len(name_part):].strip(" ()")
    ui.quote_header(name_part, code_part, "", fmt_price(px, domestic), fmt_price(abs(chg), domestic) if chg == chg else None,
                    pct_d, f"{y1.index[-1]:%Y-%m-%d} 종가")

    first_close = float(df["Close"].iloc[0])
    period_ret = float(df["Close"].iloc[-1]) / first_close - 1 if first_close else float("nan")
    mdd = float((df["Close"] / df["Close"].cummax() - 1).min())
    hi52, lo52 = float(y1["High"].max()), float(y1["Low"].min())
    vol_txt, vol_sub = "-", ""
    if "Volume" in y1.columns and float(last.get("Volume", 0) or 0) > 0:
        v = float(last["Volume"])
        avg20 = float(y1["Volume"].iloc[-21:-1].mean()) if len(y1) > 21 else float("nan")
        vol_txt = f"{v:,.0f}"
        vol_sub = f"20일 평균의 {v / avg20 * 100:.0f}%" if avg20 == avg20 and avg20 > 0 else ""
    ui.cards([
        ("시가", fmt_price(float(last["Open"]), domestic)),
        ("고가", fmt_price(float(last["High"]), domestic)),
        ("저가", fmt_price(float(last["Low"]), domestic)),
        ("거래량", vol_txt, vol_sub),
        ("52주 최고", fmt_price(hi52, domestic), f"현재가 대비 {(px / hi52 - 1) * 100:+.1f}%" if hi52 else ""),
        ("52주 최저", fmt_price(lo52, domestic), f"현재가 대비 {(px / lo52 - 1) * 100:+.1f}%" if lo52 else ""),
        ("기간 수익률", _pct(period_ret, True), f"{df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}", period_ret),
        ("최대 낙폭(MDD)", _pct(mdd), "조회 기간 고점 대비 최대 하락"),
    ], cols=4)

    # 캔들이 너무 많으면 주봉으로 묶어서 깨져 보이지 않게 한다
    plot_df = df
    weekly = False
    if chart_type == "캔들" and len(df) > 300:
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in df.columns:
            agg["Volume"] = "sum"
        plot_df = df.resample("W-FRI").agg(agg).dropna(subset=["Open", "Close"])
        weekly = True
    n_bars = len(plot_df)
    raw_plot = plot_df  # 피보나치 기준점은 가격 그대로에서 찾는다
    pct_mode = basis == "수익률"
    if pct_mode:
        base0 = float(df["Close"].dropna().iloc[0])
        plot_df = plot_df.copy()
        for _c in ("Open", "High", "Low", "Close"):
            if _c in plot_df.columns:
                plot_df[_c] = (plot_df[_c] / base0 - 1) * 100

    import ta

    want_ma = IND_OPTS[0] in inds
    want_bb = IND_OPTS[1] in inds
    want_rsi = IND_OPTS[2] in inds
    want_macd = IND_OPTS[3] in inds
    want_ichi = IND_OPTS[4] in inds
    want_fib = IND_OPTS[5] in inds and "fib" in IP
    longest = max([*IP.get("ma", [0]), IP.get("bb_n", 0), IP.get("rsi_n", 0) * 3, IP.get("macd_s", 0) * 3 + IP.get("macd_g", 0), 20])
    base_c = _indicator_base(chosen["code"], start_d, end_d, weekly, longest) if (want_ma or want_bb or want_rsi or want_macd) else None
    x0 = plot_df.index[0]

    def _disp(sr):  # 가격 단위 → 화면 단위(수익률 모드면 %)
        sr = sr[sr.index >= x0].reindex(plot_df.index)
        return (sr / base0 - 1) * 100 if pct_mode else sr

    n_rows = 2 + int(want_rsi and base_c is not None) + int(want_macd and base_c is not None)
    heights = {2: [0.75, 0.25], 3: [0.62, 0.17, 0.21], 4: [0.54, 0.14, 0.16, 0.16]}[n_rows]
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.03)
    if chart_type == "캔들":
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"],
                increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
                increasing_line_width=1, decreasing_line_width=1,
                whiskerwidth=0.4,
                name="주가",
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index, y=plot_df["Close"], mode="lines",
                line=dict(color=LINE_COLOR, width=2), name="종가",
                hovertemplate="%{x|%Y-%m-%d}<br>수익률 %{y:+.2f}%<extra></extra>" if pct_mode else "%{x|%Y-%m-%d}<br>종가 %{y:,.0f}<extra></extra>" if domestic
                else "%{x|%Y-%m-%d}<br>종가 %{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
    if base_c is not None:
        if want_ma:
            unit = "주" if weekly else "일"
            for i_ma, n in enumerate(IP["ma"]):
                fig.add_trace(go.Scatter(x=plot_df.index, y=_disp(ta.sma(base_c, n)), mode="lines", name=f"{n}{unit}선",
                                         line=dict(color=ta.ma_color(n, i_ma), width=1.1),
                                         hovertemplate=f"{n}{unit}선 %{{y:,.2f}}<extra></extra>"), row=1, col=1)
        if want_bb:
            mid, upb, lob = ta.bollinger(base_c, IP["bb_n"], IP["bb_k"])
            fig.add_trace(go.Scatter(x=plot_df.index, y=_disp(upb), mode="lines", name="볼린저 상단",
                                     line=dict(color="#A0AEC0", width=1, dash="dot"), hovertemplate="상단 %{y:,.2f}<extra></extra>"), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=_disp(lob), mode="lines", name="볼린저 하단", fill="tonexty",
                                     fillcolor="rgba(160,174,192,0.10)", line=dict(color="#A0AEC0", width=1, dash="dot"),
                                     hovertemplate="하단 %{y:,.2f}<extra></extra>"), row=1, col=1)
            if not want_ma:
                fig.add_trace(go.Scatter(x=plot_df.index, y=_disp(mid), mode="lines", name=f"볼린저 중심({IP['bb_n']})",
                                         line=dict(color=ta.MA_COLORS[20], width=1), hovertemplate="중심 %{y:,.2f}<extra></extra>"), row=1, col=1)
    import drawings as dw

    def _conv(v):  # 가격 → 화면 단위
        return (v / base0 - 1) * 100 if pct_mode else v

    fit_lo, fit_hi = [], []  # 피보나치를 그릴 때 처음 화면의 세로 범위 (먼 레벨선 때문에 차트가 납작해지지 않게)
    for _c in ("Low", "Close") if "Low" in plot_df.columns else ("Close",):
        fit_lo.append(float(plot_df[_c].min()))
    for _c in ("High", "Close") if "High" in plot_df.columns else ("Close",):
        fit_hi.append(float(plot_df[_c].max()))
    if want_ichi and "ichi" in IP:
        it, ik, is_, idd = IP["ichi"]
        ohlc = _ohlc_base(chosen["code"], start_d, end_d, weekly, max(it, ik, is_) + idd)
        if ohlc is not None and len(ohlc) > max(it, ik, is_):
            I = dw.ichimoku(ohlc, it, ik, is_, idd, weekly)

            def _cut(sr, keep_future=False):
                sr = sr[sr.index >= x0]
                if not keep_future:
                    sr = sr.reindex(plot_df.index)
                return _conv(sr)

            sa, sb = _cut(I["span_a"], True), _cut(I["span_b"], True)
            for seg_x, seg_a, seg_b, up in dw.cloud_segments(sa, sb):
                fig.add_trace(go.Scatter(x=seg_x, y=seg_b, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=seg_x, y=seg_a, mode="lines", line=dict(width=0), fill="tonexty", hoverinfo="skip",
                                         fillcolor="rgba(229,72,77,0.16)" if up else "rgba(59,130,246,0.16)", showlegend=False),
                              row=1, col=1)
            for nm, sr, col, w in (("선행스팬1", sa, "#E5484D", 0.8), ("선행스팬2", sb, "#3B82F6", 0.8),
                                   ("전환선", _cut(I["tenkan"]), "#D9B85E", 1.2), ("기준선", _cut(I["kijun"]), "#B794F4", 1.2),
                                   ("후행스팬", _cut(I["chikou"]), "#48BB78", 1.0)):
                fig.add_trace(go.Scatter(x=sr.index, y=sr.values, mode="lines", name=nm, line=dict(color=col, width=w),
                                         hovertemplate=f"{nm} %{{y:,.2f}}<extra></extra>"), row=1, col=1)
            for sr in (sa, sb):
                if sr.notna().any():
                    fit_lo.append(float(sr.min()))
                    fit_hi.append(float(sr.max()))
        else:
            st.caption("일목균형표 계산에 필요한 과거 시세를 불러오지 못했습니다.")
    if want_fib:
        F = IP["fib"]
        anc = None
        if F["mode"] == "직접 지정":
            anc = dw.manual_anchors(F["kind"], raw_plot, F["dates"])
            if anc is None:
                st.caption("피보나치 기준점 날짜를 모두 고르면 그 점으로 그립니다. 지금은 자동으로 잡은 기준점입니다.")
        if anc is None:
            hi_s = raw_plot["High"] if "High" in raw_plot.columns else raw_plot["Close"]
            lo_s = raw_plot["Low"] if "Low" in raw_plot.columns else raw_plot["Close"]
            anc = dw.auto_anchors(F["kind"], dw.zigzag(hi_s, lo_s, F["sens"]), F["nth"])
        if anc is None:
            st.caption("피보나치를 그릴 고점·저점이 부족합니다. 민감도(%)를 낮추거나 기간을 늘려 보세요.")
        else:
            ext_n = max(5, n_bars // 15)
            xfull = plot_df.index.append(dw.future_index(plot_df.index, ext_n, weekly))
            for j, (lv, s0, ys) in enumerate(dw.fib_lines(F["kind"], anc, F["levels"], n_bars, ext_n)):
                col = dw.FIB_COLORS[j % len(dw.FIB_COLORS)]
                xs = xfull[s0:s0 + len(ys)]
                yv = _conv(np.asarray(ys, dtype=float))
                lab = f"{lv:g}  ({yv[-1]:+.1f}%)" if pct_mode else f"{lv:g}  ({yv[-1]:,.0f})" if domestic else f"{lv:g}  ({yv[-1]:,.2f})"
                fig.add_trace(go.Scatter(x=xs, y=yv, mode="lines", line=dict(color=col, width=1), hoverinfo="skip",
                                         showlegend=False, meta="nofit"), row=1, col=1)
                fig.add_trace(go.Scatter(x=[xs[-1]], y=[yv[-1]], mode="text", text=[lab], textposition="top left",
                                         textfont=dict(color=col, size=10), hoverinfo="skip", showlegend=False, meta="nofit"),
                              row=1, col=1)
            nums = ["1", "3", "2"] if F["kind"] == dw.FIB_KINDS[0] else ["1", "2", "3"]
            fig.add_trace(go.Scatter(x=[plot_df.index[a["i"]] for a in anc], y=[_conv(a["p"]) for a in anc], mode="markers+text",
                                     text=nums[:len(anc)], textposition=["bottom center" if a["k"] == "L" else "top center" for a in anc],
                                     marker=dict(size=7, color="#ECE6D8", line=dict(color="#C9A44C", width=1.5)),
                                     textfont=dict(color="#ECE6D8", size=11), hoverinfo="skip", showlegend=False, meta="nofit"),
                          row=1, col=1)
            lo_v, hi_v = min(fit_lo), max(fit_hi)  # 이미 화면 단위
            pad = (hi_v - lo_v) * 0.06 or abs(hi_v) * 0.05 or 1
            fig.update_yaxes(range=[lo_v - pad, hi_v + pad], row=1, col=1)
            fig.update_layout(meta={"tv_minmax": ["yaxis"]})
    if base_c is not None:
        r_i = 3
        if want_rsi:
            rv = ta.rsi(base_c, IP["rsi_n"])
            rv = rv[rv.index >= x0].reindex(plot_df.index)
            fig.add_trace(go.Scatter(x=plot_df.index, y=rv, mode="lines", name="RSI", line=dict(color="#6B46C1", width=1.3),
                                     hovertemplate="RSI %{y:.1f}<extra></extra>"), row=r_i, col=1)
            for lv, col in ((IP["rsi_hi"], UP_COLOR), (IP["rsi_lo"], DOWN_COLOR)):
                fig.add_hline(y=lv, row=r_i, col=1, line_width=1, line_dash="dot", line_color=col)
            fig.update_yaxes(title_text=f"RSI({IP['rsi_n']})", range=[0, 100], tickvals=sorted({IP["rsi_lo"], 50, IP["rsi_hi"]}), row=r_i, col=1)
            r_i += 1
        if want_macd:
            ml, sg, hist = (x[x.index >= x0].reindex(plot_df.index) for x in ta.macd(base_c, IP["macd_f"], IP["macd_s"], IP["macd_g"]))
            fig.add_trace(go.Bar(x=plot_df.index, y=hist, name="MACD 히스토그램",
                                 marker_color=np.where(hist.fillna(0) >= 0, UP_COLOR, DOWN_COLOR), opacity=0.5,
                                 hovertemplate="히스토그램 %{y:,.2f}<extra></extra>"), row=r_i, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=ml, mode="lines", name="MACD", line=dict(color="#2B6CB0", width=1.3),
                                     hovertemplate="MACD %{y:,.2f}<extra></extra>"), row=r_i, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=sg, mode="lines", name="시그널", line=dict(color="#DD6B20", width=1.1),
                                     hovertemplate="시그널 %{y:,.2f}<extra></extra>"), row=r_i, col=1)
            fig.update_yaxes(title_text=f"MACD({IP['macd_f']},{IP['macd_s']},{IP['macd_g']})", row=r_i, col=1)
    elif want_ma or want_bb or want_rsi or want_macd:
        st.caption("보조지표 계산에 필요한 과거 시세를 불러오지 못했습니다.")
    if "Volume" in plot_df.columns:
        fig.add_trace(
            go.Bar(x=plot_df.index, y=plot_df["Volume"], marker_color=VOLUME_COLOR, name="거래량",
                   hovertemplate="%{x|%Y-%m-%d}<br>거래량 %{y:,.0f}<extra></extra>"),
            row=2, col=1,
        )
    # 봉이 많을수록 간격을 넓혀 봉을 가늘게 (굵고 뭉개져 보이는 현상 방지)
    box_gap = 0.3 if n_bars <= 60 else 0.45 if n_bars <= 150 else 0.6
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider_visible=False, hoverformat="%Y-%m-%d")
    if rate_sel:
        add_rates(fig, rate_sel, start_d, end_d, n_rows + 1, domain=list(fig.layout.yaxis.domain))
    fig.update_yaxes(title_text="수익률(%)" if pct_mode else "가격", row=1, col=1, tickformat=",.1f" if pct_mode else ",", ticksuffix="%" if pct_mode else "")
    if pct_mode:
        fig.add_hline(y=0, row=1, col=1, line_width=1, line_dash="dot", line_color="#718096")
    fig.update_yaxes(title_text="거래량", row=2, col=1, tickformat=",")
    fig.update_layout(
        height=560 + 130 * (n_rows - 2), showlegend=bool((base_c is not None and (want_ma or want_bb)) or want_ichi), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=10)),
        margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
        boxgap=box_gap, bargap=0.3 if n_bars <= 150 else 0.4,
    )
    st.plotly_chart(fig, use_container_width=True)
    if weekly:
        st.caption("기간이 길어 캔들을 주봉(1주 = 1개)으로 묶어서 표시합니다. 위 지표는 일별 데이터 기준입니다.")
    if IND_OPTS[6] in inds and "sg_n" in IP:
        _sigma_section(chosen, start_d, end_d, IP["sg_n"])

    exp, op = ui.more("상세 통계 더보기 · 변동성 · 샤프비율 · CAPM · VaR", key="single_more_stats")
    if op:
        with exp:
            rf, rf_label = risk_free_for(domestic, df.index[-1].date())
            show_stats(df, rf, rf_label, benchmark_for(chosen.get("market", ""), domestic))
            r_d = df["Close"].dropna().pct_change().dropna()
            if len(r_d) >= 20:
                st.markdown("**VaR · ES (1일 손실률)**")
                risk_grid(risk_metrics(r_d))
                st.caption(RISK_NOTE)
    exp2, op2 = ui.more("일별 시세 표 · CSV 내려받기", key="single_more_data")
    if op2:
        with exp2:
            st.dataframe(df.sort_index(ascending=False), use_container_width=True)
            st.download_button(
                "CSV 내려받기",
                df.to_csv().encode("utf-8-sig"),
                file_name=f"{chosen['code']}_{start_d}_{end_d}.csv",
                mime="text/csv",
            )


# ======================================================================
# 탭 2: 수익률 비교
# ======================================================================
