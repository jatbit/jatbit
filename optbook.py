"""옵션 페이지 '내 포지션' 탭 — 주식·미국 옵션·국내 옵션을 한 장부에 모으고, 그 아래에 헷지 / 손익·위험 / 헷지 시뮬레이션.
세션 키: hedge_stocks_cur(주식, 미국·국내), hedge_opts_cur(미국 옵션), kr_opts_cur(국내 옵션)."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bsm
import kropt
import ui
from options_data import (_empty_opts, _empty_stocks, is_us_ticker, load_chain_raw, load_expiries, load_underlying, load_usdkrw,
                          years_to_expiry)


def _counts():
    s = st.session_state.get("hedge_stocks_cur")
    o = st.session_state.get("hedge_opts_cur")
    k = st.session_state.get("kr_opts_cur")
    ns = 0 if s is None else int(s["종목코드"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan}).notna().sum())
    no = 0 if o is None else len(o.dropna(how="all"))
    nk = 0 if k is None else len(k.dropna(how="all"))
    return ns, no, nk


def _add_us_opt(ticker, expiry):
    ss = st.session_state
    base = ss.get("hedge_opts_cur")
    base = _empty_opts() if base is None else base
    k = ss.get(f"bk_K_{ticker}_{expiry}")
    n = int(ss.get("bk_n", 1))
    kind = ss.get("bk_kind", "콜")
    if k is None or not n:
        return
    row = pd.DataFrame([{"기초자산": ticker, "만기": expiry, "종류": kind, "행사가": float(k), "수량(계약)": n}])
    ss["hedge_opts_cur"] = pd.concat([base, row], ignore_index=True)
    ss["hedge_opts_ver"] = ss.get("hedge_opts_ver", 0) + 1
    ss["bk_msg"] = f"{ticker} {expiry} {kind} {float(k):g} × {n}계약을 추가했습니다."


def _add_stock():
    ss = st.session_state
    q = (ss.get("bk_stock_q") or "").strip()
    n = float(ss.get("bk_stock_n") or 0)
    if not q or not n:
        return
    code, name = q.upper(), q.upper()
    if not is_us_ticker(code):
        hit = kropt.resolve_under(q)
        if not hit:
            ss["bk_msg"] = f"'{q}'를 찾지 못했습니다."
            return
        code, name = hit
    base = ss.get("hedge_stocks_cur")
    base = _empty_stocks() if base is None else base
    row = pd.DataFrame([{"종목코드": code, "종목명": name, "보유 주식 수": n, "비고": "직접 추가"}])
    ss["hedge_stocks_cur"] = pd.concat([base, row], ignore_index=True)
    ss["hedge_stocks_ver"] = ss.get("hedge_stocks_ver", 0) + 1
    ss["bk_msg"] = f"{name} {n:,.0f}주를 추가했습니다."


def add_strategy_us(ticker: str, expiry: str):
    """전략 빌더 → 미국 옵션 장부."""
    def _f(legs):
        ss = st.session_state
        base = ss.get("hedge_opts_cur")
        base = _empty_opts() if base is None else base
        rows, sh = [], 0.0
        for lg in legs:
            if lg["kind"] == "S":
                sh += lg["n"] * 100
            else:
                rows.append({"기초자산": ticker, "만기": expiry, "종류": "콜" if lg["kind"] == "C" else "풋",
                             "행사가": lg["K"], "수량(계약)": int(round(lg["n"]))})
        if rows:
            ss["hedge_opts_cur"] = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
            ss["hedge_opts_ver"] = ss.get("hedge_opts_ver", 0) + 1
        if sh:
            sb = ss.get("hedge_stocks_cur")
            sb = _empty_stocks() if sb is None else sb
            ss["hedge_stocks_cur"] = pd.concat([sb, pd.DataFrame([{"종목코드": ticker, "종목명": ticker, "보유 주식 수": sh,
                                                                   "비고": "전략"}])], ignore_index=True)
            ss["hedge_stocks_ver"] = ss.get("hedge_stocks_ver", 0) + 1
        ss["chain_strat_addmsg"] = f"{ticker} {expiry} 전략 {len(legs)}개 다리를 '내 포지션'에 추가했습니다."
    return _f


def _add_panel(ticker: str | None) -> None:
    mode = st.radio("추가할 것", ["미국 옵션", "국내 옵션", "주식", "최적 포트폴리오"], horizontal=True, key="bk_mode",
                    label_visibility="collapsed")
    if mode == "미국 옵션":
        if not ticker or not is_us_ticker(ticker):
            st.info("맨 위에서 미국 티커를 고르면 그 종목의 만기·행사가 목록에서 바로 고를 수 있습니다.")
            return
        try:
            exps = load_expiries(ticker)
        except Exception as e:  # noqa: BLE001
            st.error(f"만기 목록을 불러오지 못했습니다: {e}")
            return
        if not exps:
            st.warning(f"{ticker}에 상장된 옵션이 없습니다.")
            return
        today = pd.Timestamp.now().normalize()
        c1, c2, c3, c4, c5 = st.columns([1.5, 0.8, 1.2, 1, 1])
        first_i = next((i for i, e in enumerate(exps) if (pd.Timestamp(e) - today).days >= 7), 0)
        e = c1.selectbox("만기", exps, index=first_i, key=f"bk_exp_{ticker}",
                         format_func=lambda x: f"{x} (D-{(pd.Timestamp(x) - today).days})")
        c2.selectbox("종류", ["콜", "풋"], key="bk_kind")
        try:
            raw = load_chain_raw(ticker, e)
            ks = sorted(pd.to_numeric(raw["strike"], errors="coerce").dropna().unique().tolist())
            S = float(load_underlying(ticker)["spot"])
        except Exception as ex:  # noqa: BLE001
            st.error(f"행사가 목록을 불러오지 못했습니다: {ex}")
            return
        atm = int(np.argmin([abs(k - S) for k in ks])) if ks else 0
        c3.selectbox("행사가", ks, index=atm, key=f"bk_K_{ticker}_{e}", format_func=lambda k: f"{k:g}")
        c4.number_input("수량(계약, 매도 −)", value=1, step=1, key="bk_n")
        c5.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        c5.button("추가", key="bk_add_us", on_click=_add_us_opt, args=(ticker, e), use_container_width=True, type="primary")
    elif mode == "국내 옵션":
        kropt.render_kr_add()
    elif mode == "주식":
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.text_input("종목 (미국 티커 또는 국내 종목명·코드)", key="bk_stock_q", placeholder="AAPL, 삼성전자")
        c2.number_input("주식 수", value=100.0, step=10.0, key="bk_stock_n")
        c3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        c3.button("추가", key="bk_add_stock", on_click=_add_stock, use_container_width=True, type="primary")
    else:
        from hedge import _load_from_pf

        res = st.session_state.get("pf_opt_result")
        fxv = load_usdkrw()
        c1, c2, c3 = st.columns([1.6, 1.2, 1.2])
        amount = c1.number_input("투자 금액 (원)", value=100_000_000, step=10_000_000, min_value=0, key="hedge_amount")
        fx = c2.number_input("원/달러 환율", value=round(fxv[0], 2) if fxv else 1400.0, step=1.0, key="hedge_fx")
        c3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        c3.button("불러오기", key="hedge_load_pf", use_container_width=True, disabled=not res, on_click=_load_from_pf, args=(amount, fx))
        st.caption("'포트폴리오' 페이지의 최적 비중 × 투자 금액 ÷ 최근 종가로 주식 수를 채웁니다 (기존 주식 목록을 바꿉니다)."
                   if res else "'포트폴리오' 페이지에서 최적 포트폴리오를 계산하면 켜집니다.")
    msg = st.session_state.pop("bk_msg", None)
    if msg:
        st.success(msg)


def _tables() -> None:
    ss = st.session_state
    stocks = ss.get("hedge_stocks_cur")
    stocks = _empty_stocks() if stocks is None else stocks
    opts = ss.get("hedge_opts_cur")
    opts = _empty_opts() if opts is None else opts
    kr = kropt.kr_book()
    st.markdown("**주식** (미국·국내)")
    ss["hedge_stocks_cur"] = st.data_editor(
        stocks, key=f"hedge_stocks_ed_{ss.get('hedge_stocks_ver', 0)}", num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={"보유 주식 수": st.column_config.NumberColumn(format="%.2f"), "비고": st.column_config.TextColumn(disabled=True)})
    st.markdown("**미국 옵션** (1계약 = 100주, 매수 +, 매도 −)")
    ss["hedge_opts_cur"] = st.data_editor(
        opts, key=f"hedge_opts_ed_{ss.get('hedge_opts_ver', 0)}", num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={"종류": st.column_config.SelectboxColumn(options=["콜", "풋"]), "만기": st.column_config.TextColumn(help="YYYY-MM-DD"),
                       "수량(계약)": st.column_config.NumberColumn(step=1), "행사가": st.column_config.NumberColumn(format="%.2f")})
    st.markdown("**국내 옵션** (주식옵션 1계약 = 10주, 코스피200 = 지수 × 25만 원)")
    ss["kr_opts_cur"] = st.data_editor(
        kr, key=f"kr_opts_ed_{ss.get('kr_opts_ver', 0)}", num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={"기초자산": st.column_config.TextColumn(help="6자리 코드 또는 KOSPI200"),
                       "종류": st.column_config.SelectboxColumn(options=["콜", "풋"]),
                       "만기": st.column_config.TextColumn(help="YYYY-MM-DD (둘째 목요일)"),
                       "수량(계약)": st.column_config.NumberColumn(step=1), "행사가": st.column_config.NumberColumn(format="%.2f"),
                       "가격": st.column_config.NumberColumn(format="%.2f", help="비우면 HV 60일 이론가")})


# ----------------------------------------------------------------------
# 국내 옵션: 그릭스·헷지·손익
# ----------------------------------------------------------------------
def _kr_stock_shares(code: str) -> float:
    s = st.session_state.get("hedge_stocks_cur")
    if s is None or not len(s):
        return 0.0
    m = s[s["종목코드"].astype(str).str.strip() == code]
    return float(pd.to_numeric(m["보유 주식 수"], errors="coerce").fillna(0).sum())


def render_kr_section() -> None:
    kr = kropt.kr_book()
    if kr is None or not len(kr.dropna(how="all")):
        return
    book, warns, r_lab = kropt.price_kr_book(kr)
    for w in warns:
        st.warning(w)
    for code, u in book.items():
        with st.container(border=True):
            S, mult, name = u["S"], u["mult"], u["name"]
            sh = _kr_stock_shares(code) if code != kropt.KS200 else 0.0
            d_tot = sh + sum(lg["n"] * mult * lg["delta"] for lg in u["legs"])
            g_tot = sum(lg["n"] * mult * lg["gamma"] for lg in u["legs"])
            v_tot = sum(lg["n"] * mult * lg["vega"] for lg in u["legs"])
            th_tot = sum(lg["n"] * mult * lg["theta"] for lg in u["legs"])
            val = sh * S + sum(lg["n"] * mult * lg["mkt"] for lg in u["legs"])
            st.markdown(f"**{name}** · 현재가 {S:,.2f} ({u['asof']})")
            unit = "원/포인트" if code == kropt.KS200 else "주"
            ui.cards([("포지션 가치", f"{val:,.0f}원"), ("델타", f"{d_tot:,.1f}{unit}", f"기초자산 1 오르면 {d_tot:,.0f}원"),
                      ("감마", f"{g_tot:,.4f}"), ("베가 (1%p)", f"{v_tot:,.0f}원"), ("세타 (1일)", f"{th_tot:,.0f}원")], cols=5)
            tb = pd.DataFrame([{"옵션": lg["label"], "수량": f"{lg['n']:+g}", "가격": lg["mkt"], "IV(%)": lg["sig"] * 100, "IV 기준": lg["src"],
                                "델타": lg["delta"], "감마": lg["gamma"]} for lg in u["legs"]])
            st.dataframe(tb.round(4), hide_index=True, use_container_width=True)
            # 헷지
            if code == kropt.KS200:
                st.caption(f"델타만 0으로: 코스피200 선물 {-d_tot / 250_000:+,.2f}계약 (선물 승수 25만 원)")
            else:
                st.caption(f"델타만 0으로: 주식 {-d_tot:+,.0f}주 거래")
            ch = kropt.kr_chain_for(code, name)
            if ch is not None and abs(g_tot) > 1e-9:
                r = u["r"]
                exps = sorted(ch["만기"].unique())
                c1, c2, c3 = st.columns(3)
                e = c1.selectbox("감마 헷지 옵션 만기", exps, key=f"krh_e_{code}")
                d = kropt.chain_greeks(ch[ch["만기"] == e], S, r)
                kind = c2.selectbox("종류", ["콜", "풋"], key=f"krh_k_{code}")
                dd = d[(d["종류"] == kind) & np.isfinite(d["IV"])]
                if len(dd):
                    ks = dd["행사가"].tolist()
                    K = c3.selectbox("행사가", ks, index=int(np.argmin([abs(k - S) for k in ks])), key=f"krh_K_{code}_{e}_{kind}")
                    row = dd[dd["행사가"] == K].iloc[0]
                    sol = bsm.solve_hedge_two(d_tot, g_tot, 1.0, 0.0, float(row["delta"]) * mult, float(row["gamma"]) * mult)
                    if sol:
                        n_s, n_o = sol
                        st.success(f"델타·감마 동시 0: {name} {e} {kind} {K:,.0f} {n_o:+,.1f}계약 + 주식 {n_s:+,.0f}주 "
                                   f"(옵션 1계약 {row['종가']:,.2f} · IV {row['IV'] * 100:.1f}%)")
            elif abs(g_tot) > 1e-9:
                st.caption("감마까지 0으로 만들려면 위 '추가할 것 → 국내 옵션'에서 KRX CSV를 올려 헷지에 쓸 옵션 시세를 넣어 주세요.")
            # 손익 그래프
            import strat

            legs = [{"kind": "C" if lg["is_call"] else "P", "K": lg["K"], "n": lg["n"], "px": lg["mkt"], "iv": lg["sig"]} for lg in u["legs"]]
            if sh:
                legs.append({"kind": "S", "K": np.nan, "n": sh / mult, "px": S, "iv": np.nan})
            T_min = min(lg["T"] for lg in u["legs"])
            ctx = {"r": u["r"], "q": 0.0}
            g = np.linspace(S * 0.75, S * 1.25, 241)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=g, y=strat.value_at(legs, g, 0.0, ctx, mult) if len({lg["exp"] for lg in u["legs"]}) == 1
                                     else _multi_exp(u, g, T_min, sh), name="가장 가까운 만기", line=dict(color="#D4AF37", width=2.2)))
            fig.add_trace(go.Scatter(x=g, y=_today(u, g, sh), name="오늘", line=dict(color="#4C9BE8", dash="dash")))
            fig.add_hline(y=0, line_dash="dot", line_color="#888")
            fig.add_vline(x=S, line_dash="dot", line_color="#888")
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="기초자산 가격", yaxis_title="손익(원)",
                              hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)
    st.caption(f"국내 옵션은 유럽형 BSM으로 계산합니다 (무위험금리 {r_lab}, 배당 0 가정). 가격을 넣지 않은 옵션은 HV 60일 이론가입니다.")


def _today(u, g, sh):
    out = sh * (g - u["S"])
    for lg in u["legs"]:
        out = out + lg["n"] * u["mult"] * (bsm.price(g, lg["K"], lg["T"], u["r"], 0.0, lg["sig"], lg["is_call"]) - lg["mkt"])
    return out


def _multi_exp(u, g, T_min, sh):
    """만기가 여러 개면 가장 가까운 만기 시점: 그때 끝나는 옵션은 내재가치, 나머지는 남은 기간으로 재평가."""
    out = sh * (g - u["S"])
    for lg in u["legs"]:
        Tl = lg["T"] - T_min
        v = (np.maximum(g - lg["K"], 0) if lg["is_call"] else np.maximum(lg["K"] - g, 0)) if Tl <= 1e-6 else \
            bsm.price(g, lg["K"], Tl, u["r"], 0.0, lg["sig"], lg["is_call"])
        out = out + lg["n"] * u["mult"] * (v - lg["mkt"])
    return out


# ----------------------------------------------------------------------
# 탭
# ----------------------------------------------------------------------
def render_book(ticker: str | None) -> None:
    ns, no, nk = _counts()
    st.caption("주식·옵션을 한 장부에 모아 그릭스 합계, 자동 헷지, 손익·위험을 계산합니다. 체인·전략 탭에서 '내 포지션에 추가'로 넣어도 됩니다.")
    exp, op = ui.more(f"포지션 추가·수정 (주식 {ns} · 미국 옵션 {no} · 국내 옵션 {nk})", key="bk_more", expanded=(ns + no + nk == 0))
    if op:
        with exp:
            _add_panel(ticker)
            st.divider()
            _tables()
    if ns + no + nk == 0:
        st.info("위 '포지션 추가·수정'에서 주식이나 옵션을 넣으면 아래에 헷지와 손익이 계산됩니다.")
        return
    (t1, o1), (t2, o2), (t3, o3), (t4, o4) = ui.lazy_tabs(["헷지", "손익·위험", "헷지 시뮬레이션", "전체 위험 (채권 포함)"], key="bk_tabs")
    if o1:
        with t1:
            from hedge import render_hedge

            sb = st.session_state.get("hedge_stocks_cur")
            sb = _empty_stocks() if sb is None else sb
            if no or any(is_us_ticker(str(c)) for c in sb["종목코드"].astype(str)):
                render_hedge(editor=False)
            if nk:
                st.markdown("##### 국내 옵션")
                render_kr_section()
    if o2:
        with t2:
            from capm_opt import render_capm_opt

            render_capm_opt()
            if nk:
                st.caption("국내 옵션의 손익 그래프와 그릭스는 '헷지' 탭의 국내 옵션 칸에 있습니다.")
    if o3:
        with t3:
            try:
                import hedgesim

                hedgesim.render(ticker)
            except ImportError:
                st.info("준비 중입니다.")
    if o4:
        with t4:
            import riskall

            riskall.render("ra_o")


__all__ = [n for n in dir() if not n.startswith("__")]
