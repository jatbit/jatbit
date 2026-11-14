"""포트폴리오 페이지: 최적 비중(최대 샤프·최소 분산 등)과 위험 지표."""
from common import *  # noqa: F401,F403


def _port_stats(W: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float):
    ret = W @ mu
    vol = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", W, cov, W), 1e-18))
    return ret, vol, (ret - rf) / vol


def _refine(w: np.ndarray, score_batch, cap: float) -> np.ndarray:
    """한 종목에서 다른 종목으로 비중을 옮기는 모든 경우를 한꺼번에 평가해, 점수가 가장 많이 오르는 이동을 반복한다.
    (공매도 불가, 종목당 상한 cap) 종목 수가 많아도 빠르게 돌도록 벡터화했다."""
    w = w.copy()
    n = len(w)
    best = float(score_batch(w[None, :])[0])
    ii, jj = np.where(~np.eye(n, dtype=bool))
    for step in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        for _ in range(5000):
            amt = np.minimum(step, np.minimum(w[jj], cap - w[ii]))
            ok = amt > 1e-12
            if not ok.any():
                break
            I, J, A = ii[ok], jj[ok], amt[ok]
            C = np.tile(w, (len(I), 1))
            idx = np.arange(len(I))
            C[idx, I] += A
            C[idx, J] -= A
            sc = score_batch(C)
            k = int(np.argmax(sc))
            if sc[k] > best + 1e-12:
                w, best = C[k], float(sc[k])
            else:
                break
    return w


@st.cache_data(show_spinner="최적 비중을 계산하는 중...", max_entries=30, ttl=60 * 60)
def optimize_portfolios(mu: np.ndarray, cov: np.ndarray, rf: float, cap: float, seed: int = 42):
    """무작위 후보 + 비중 이동 탐색으로 최대 샤프 / 최소 분산 포트폴리오를 찾는다. (SciPy 없이 numpy만 사용)"""
    n = len(mu)
    rng = np.random.default_rng(seed)
    per = 15000  # 정확도를 위해 후보를 넉넉히 만든다 (시간이 조금 더 걸림)
    W = np.vstack([rng.dirichlet(np.ones(n) * a, size=per) for a in (0.3, 1.0, 3.0)] + [np.full((1, n), 1.0 / n)])
    W = W[W.max(axis=1) <= cap + 1e-9]
    ret, vol, sh = _port_stats(W, mu, cov, rf)

    def sharpe_f(Wb):
        return _port_stats(Wb, mu, cov, rf)[2]

    def negvar(Wb):
        return -np.einsum("ij,jk,ik->i", Wb, cov, Wb)

    # 한 군데에서만 출발하면 국소 최적에 머물 수 있어, 상위 후보 여러 개에서 각각 다듬은 뒤 가장 좋은 것을 고른다
    def best_of(starts, score):
        cands = [_refine(W[i], score, cap) for i in starts]
        return max(cands, key=lambda x: float(score(x[None, :])[0]))

    k = min(8, len(W))
    w_sh = best_of(np.argsort(-sh)[:k], sharpe_f)
    w_mv = best_of(np.argsort(vol)[:k], negvar)
    return {
        "max_sharpe": w_sh, "min_var": w_mv, "equal": np.full(n, 1.0 / n),
        "cloud": (vol[::6], ret[::6]),
    }


def render_portfolio(listing: pd.DataFrame) -> None:
    ordered = listing.sort_values("Name")
    options = {f"{r.Name} ({r.Code})": (r.Code, r.Name, str(r.Market)) for r in ordered.itertuples()}

    c1, c2, c3 = _row([3, 2, 1])
    selected = c1.multiselect(
        "국내 종목 (2개 이상, 이름을 입력하면 검색됩니다)",
        list(options.keys()), key="pf_selected",
        on_change=_mirror, args=("pf_selected", "cmp_selected"),
        help="수익률 비교 페이지와 종목·기간 입력이 함께 바뀝니다",
    )
    extra = c2.text_input(
        "해외 티커 · 선물 · 종목 추가 (쉼표로 구분)", key="pf_extra",
        placeholder="예: AAPL, NVDA, 원유 선물, 금 선물",
        on_change=_mirror, args=("pf_extra", "cmp_extra"),
    )
    with c3:
        with ui.popover("설정", icon=":material/tune:"):
            fut_sel = st.multiselect(
                "해외 선물 (목록에서 선택)", list(FUT_OPTIONS), key="pf_fut",
                on_change=_mirror, args=("pf_fut", "cmp_fut"),
            )
            pf_set = st.container()
    period = period_input("pf_period", mirror="cmp_period", presets=True)

    items: list[dict] = []
    seen: set[str] = set()
    for label in selected:
        code, name, market = options[label]
        if code not in seen:
            seen.add(code)
            items.append({"code": code, "name": name, "domestic": True, "market": market})
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
            items.append({"code": c["code"], "name": c["name"], "domestic": c["domestic"], "market": c.get("market", "")})

    if len(items) < 2:
        st.info("종목을 2개 이상 선택하세요.")
        return
    if period is None:
        st.info("기간의 시작일과 종료일을 모두 선택하세요.")
        return
    start_d, end_d = period
    n = len(items)

    # 벤치마크: 전부 국내 종목이면 국내 지수, 해외가 섞이면 S&P 500 (직접 바꿀 수 있음)
    bench_opts = {"코스피 지수": "KS11", "코스닥 지수": "KQ11", "S&P 500": "US500"}
    if all(it["domestic"] for it in items):
        default_bench = 1 if all(it["market"].upper().startswith("KOSDAQ") for it in items) else 0
    else:
        default_bench = 2
    with pf_set:
        bench_name = st.selectbox("CAPM 벤치마크(시장 지수)", list(bench_opts), index=default_bench, key="pf_bench")
        cap_pct = st.slider("종목당 최대 비중 (%)", min_value=int(math.ceil(100 / n)), max_value=100, value=100, step=1)
    cap = cap_pct / 100
    bench_code = bench_opts[bench_name]

    names = [it["name"] for it in items]
    if len(names) != len(set(names)):
        st.error("이름이 같은 종목이 있습니다. 종목을 다시 선택하세요.")
        return

    # 가격 불러오기 → 공통 거래일로 정렬
    series = {}
    for it in items + [{"code": bench_code, "name": bench_name}]:
        try:
            d = load_prices(it["code"], start_d.isoformat(), end_d.isoformat())
        except Exception as e:  # noqa: BLE001
            st.error(f"{it['name']}: 데이터를 불러오지 못했습니다 ({e})")
            return
        if d is None or d.empty or "Close" not in d:
            st.error(f"{it['name']}: 선택한 기간에 데이터가 없습니다.")
            return
        series[it["name"]] = d["Close"].dropna()
    P = pd.concat(series, axis=1, join="inner").dropna()
    R = P.pct_change().dropna()
    if len(R) < 30:
        st.warning(f"모든 종목이 겹치는 거래일이 {len(R)}일뿐입니다. 기간을 늘리거나 종목을 바꿔 보세요.")
        return
    Rs, m = R[names], R[bench_name]

    rf, rf_label = risk_free_for(bench_code != "US500", R.index[-1].date())
    mu = Rs.mean().values * TRADING_DAYS
    cov = Rs.cov().values * TRADING_DAYS

    # ---- 최적화 (같은 입력이면 다시 계산하지 않음)
    res = optimize_portfolios(mu, cov, rf, cap)
    ports = {
        "최대 샤프 (최적)": res["max_sharpe"],
        "최소 분산": res["min_var"],
        "동일 비중": res["equal"],
    }

    st.session_state["pf_opt_result"] = {
        "items": [{"code": it["code"], "name": it["name"], "domestic": bool(it["domestic"])} for it in items],
        "weights": [float(x) for x in res["max_sharpe"]],
        "weight_sets": {k: [float(x) for x in w] for k, w in ports.items()},
        "start": f"{P.index[0]:%Y-%m-%d}",
        "prices": {nm: float(P[nm].iloc[-1]) for nm in names},
        "asof": f"{P.index[-1]:%Y-%m-%d}",
    }

    pseries, pstat = {}, {}
    for k, w in ports.items():
        rp = pd.Series(Rs.values @ w, index=Rs.index)
        pseries[k] = rp
        c = capm_stats(rp, m, rf)
        cum = (1 + rp).cumprod()
        pstat[k] = {"c": c, "rk": risk_metrics(rp), "total": float(cum.iloc[-1] - 1),
                    "mdd": float((cum / cum.cummax() - 1).min()),
                    "sharpe": (c["mean_a"] - rf) / c["std_a"] if c["std_a"] > 0 else float("nan")}

    # ---- 핵심 결과: 최대 샤프 포트폴리오
    o, e = pstat["최대 샤프 (최적)"], pstat["동일 비중"]
    ui.section("최적 포트폴리오 (최대 샤프)", f"{P.index[0]:%Y-%m-%d} ~ {P.index[-1]:%Y-%m-%d} · 공통 거래일 {len(R):,}일")
    ui.cards([
        ("연환산 기대수익률", _pct(o["c"]["mean_a"], True), f"동일 비중 {_pct(e['c']['mean_a'], True)}", o["c"]["mean_a"]),
        ("연환산 표준편차", _pct(o["c"]["std_a"]), f"동일 비중 {_pct(e['c']['std_a'])}"),
        ("샤프비율", f"{o['sharpe']:.2f}" if o["sharpe"] == o["sharpe"] else "-",
         f"동일 비중 {e['sharpe']:.2f}" if e["sharpe"] == e["sharpe"] else ""),
        ("최대 낙폭(MDD)", _pct(o["mdd"]), f"동일 비중 {_pct(e['mdd'])}"),
        ("기간 수익률", _pct(o["total"], True), "매일 같은 비중 유지 가정", o["total"]),
        ("베타 (β)", f"{o['c']['beta']:.2f}", bench_name),
        ("젠센 알파 (연)", _pct(o["c"]["alpha"], True), "CAPM 대비 초과수익", o["c"]["alpha"]),
        ("VaR 95% (1일, 과거)", _pct(o["rk"]["var95"]), "하루 손실이 이보다 클 확률 5%"),
    ], cols=4)

    if n > len(R):
        st.warning(
            f"종목 수({n}개)가 공통 거래일 수({len(R)}일)보다 많아 공분산이 불안정합니다. "
            "기간을 늘리거나 종목을 줄이면 결과가 더 믿을 만합니다."
        )

    cw, cf = st.columns([1, 1.35])
    with cw:
        ui.section("투자 비중")
        wdf = pd.DataFrame({"종목": names, **{k: [round(x * 100, 1) for x in w] for k, w in ports.items()}})
        wdf = wdf.sort_values("최대 샤프 (최적)", ascending=False)
        st.dataframe(wdf, use_container_width=True, hide_index=True,
                     column_config={k: st.column_config.ProgressColumn(k, format="%.1f%%", min_value=0, max_value=100) if k == "최대 샤프 (최적)"
                                    else st.column_config.NumberColumn(k, format="%.1f%%") for k in ports})
        st.caption(f"공매도 없이 비중 합 100%, 종목당 최대 {cap_pct}%. 기간이 짧으면 한두 종목에 비중이 쏠릴 수 있습니다.")
    with cf:
        ui.section("위험-수익 평면")
        vol_c, ret_c = res["cloud"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vol_c * 100, y=ret_c * 100, mode="markers", name="무작위 포트폴리오",
                                 marker=dict(size=4, color="#CBD5E0", opacity=0.6), hoverinfo="skip"))
        sd_i = np.sqrt(np.diag(cov))
        fig.add_trace(go.Scatter(x=sd_i * 100, y=mu * 100, mode="markers+text", text=names, textposition="top center",
                                 name="개별 종목", marker=dict(size=9, symbol="diamond", color=SERIES_COLORS[2])))
        markers = {"최대 샤프 (최적)": ("star", 16, SERIES_COLORS[1]), "최소 분산": ("circle", 11, SERIES_COLORS[0]),
                   "동일 비중": ("square", 9, "#718096")}
        for k, w in ports.items():
            sym, size, col = markers[k]
            fig.add_trace(go.Scatter(x=[float(np.sqrt(w @ cov @ w)) * 100], y=[float(w @ mu) * 100], mode="markers", name=k,
                                     marker=dict(size=size, symbol=sym, color=col, line=dict(width=1, color="white"))))
        w_o = ports["최대 샤프 (최적)"]
        s_o, r_o = float(np.sqrt(w_o @ cov @ w_o)), float(w_o @ mu)
        x_max = max(float(vol_c.max()), float(sd_i.max())) * 1.1
        if s_o > 0:
            fig.add_trace(go.Scatter(x=[0, x_max * 100], y=[rf * 100, (rf + (r_o - rf) / s_o * x_max) * 100], mode="lines",
                                     name="자본시장선", line=dict(color=SERIES_COLORS[1], dash="dot", width=1)))
        fig.update_xaxes(title_text="연환산 표준편차(%)", ticksuffix="%")
        fig.update_yaxes(title_text="연환산 기대수익률(%)", ticksuffix="%")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11),
                          legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True)

    # ---- 세부 (더보기)
    exp, op = ui.more("포트폴리오 3종 비교 더보기 · CAPM · VaR · ES", key="pf_more_ports")
    if op:
        with exp:
            prow = {}
            for k, d in pstat.items():
                c, rk = d["c"], d["rk"]
                prow[k] = {
                    "기간 수익률": _pct(d["total"], True),
                    "연환산 기대수익률": _pct(c["mean_a"], True),
                    "연환산 표준편차": _pct(c["std_a"]),
                    "샤프비율": f"{d['sharpe']:.2f}" if d["sharpe"] == d["sharpe"] else "-",
                    "MDD": _pct(d["mdd"]),
                    "베타": f"{c['beta']:.2f}",
                    "상관계수": f"{c['corr']:.2f}",
                    "R²": f"{c['r2']:.2f}",
                    "CAPM 기대수익률": _pct(c["capm"], True),
                    "젠센 알파": _pct(c["alpha"], True),
                    "트레이너": f"{c['treynor']:.3f}" if c["treynor"] == c["treynor"] else "-",
                    "VaR 95% (1일, 과거)": _pct(rk["var95"]),
                    "ES 95% (1일, 과거)": _pct(rk["es95"]),
                    "VaR 99% (1일, 과거)": _pct(rk["var99"]),
                    "ES 99% (1일, 과거)": _pct(rk["es99"]),
                    "VaR 95% (1일, 정규)": _pct(rk["nvar95"]),
                    "VaR 99% (1일, 정규)": _pct(rk["nvar99"]),
                }
            st.dataframe(pd.DataFrame(prow), use_container_width=True)
            st.caption("각 포트폴리오는 위 비중을 기간 내내 매일 유지(일별 리밸런싱)한다고 가정했습니다. 거래비용·세금은 반영하지 않았습니다.")
            st.markdown("**최적 포트폴리오(최대 샤프)의 CAPM**")
            capm_grid(o["c"], bench_name)
            st.caption(capm_note(bench_name))
            st.markdown("**최적 포트폴리오(최대 샤프)의 VaR · ES**")
            risk_grid(o["rk"])
            st.caption(RISK_NOTE)

    exp, op = ui.more("종목별 수치 더보기 · 수익률 · 변동성 · 베타 · 알파 · VaR", key="pf_more_items")
    if op:
        with exp:
            rows = []
            for nm in names:
                r = Rs[nm]
                c = capm_stats(r, m, rf)
                rk = risk_metrics(r)
                px = P[nm]
                mdd = float((px / px.cummax() - 1).min())
                sd = float(r.std(ddof=1)) * TRADING_DAYS ** 0.5
                mean_a = float(r.mean()) * TRADING_DAYS
                rows.append({
                    "종목": nm,
                    "기간 수익률": _pct(float(px.iloc[-1] / px.iloc[0] - 1), True),
                    "연환산 기대수익률": _pct(mean_a, True),
                    "연환산 표준편차": _pct(sd),
                    "샤프비율": f"{(mean_a - rf) / sd:.2f}" if sd > 0 else "-",
                    "베타": f"{c['beta']:.2f}",
                    "상관계수": f"{c['corr']:.2f}",
                    "R²": f"{c['r2']:.2f}",
                    "CAPM 기대수익률": _pct(c["capm"], True),
                    "젠센 알파": _pct(c["alpha"], True),
                    "트레이너": f"{c['treynor']:.3f}" if c["treynor"] == c["treynor"] else "-",
                    "MDD": _pct(mdd),
                    "VaR 95%(1일)": _pct(rk["var95"]),
                    "ES 95%(1일)": _pct(rk["es95"]),
                    "VaR 99%(1일)": _pct(rk["var99"]),
                    "ES 99%(1일)": _pct(rk["es99"]),
                })
            from glossary import col_help

            tdf = pd.DataFrame(rows)
            st.dataframe(tdf, use_container_width=True, hide_index=True, column_config=col_help(tdf.columns))
            st.caption(f"무위험수익률: {rf_label}. 벤치마크: {bench_name}. VaR·ES는 1일 기준 과거 시뮬레이션 손실률입니다.")


# ======================================================================
# 실행
