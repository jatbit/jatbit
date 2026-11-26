"""채권 → 내 채권 탭 (보유 채권 장부 → 평가액·듀레이션·DV01·키 레이트·시나리오·면역화)
+ 계산기 탭의 콜옵션부 채권(헐-화이트 트리, OAS).
세션 키: bd_book (보유 채권 표), bd_book_ver.
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bondmath as bm
import ui

COLS = ["이름", "통화", "액면", "쿠폰(%)", "잔존만기(년)", "이자 횟수", "신용스프레드(bp)", "회수율(%)"]
MATS = [1, 2, 3, 5, 7, 10, 20, 30]
CUR_NAME = {"USD": "미 국채", "KRW": "국고채"}


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object" if c in ("이름", "통화") else "float") for c in COLS})


def _book() -> pd.DataFrame:
    b = st.session_state.get("bd_book")
    return _empty() if b is None else b


def _money(v: float, cur: str) -> str:
    if not np.isfinite(v):
        return "-"
    if cur == "KRW":
        return f"{v:,.0f}원"
    return ("-$" if v < 0 else "$") + f"{abs(v):,.0f}"


def _fallback_curve(cur: str):
    if cur == "KRW":
        try:
            from kr import kr_rate

            r = kr_rate("3Y")
            if r:
                return bm.flat_curve(r[0], f"국고채 3년 {r[0] * 100:.2f}% 평탄 가정")
        except Exception:  # noqa: BLE001
            pass
        return bm.flat_curve(0.03, "3% 평탄 가정")
    return bm.flat_curve(0.04, "4% 평탄 가정")


def _push(row: dict) -> None:
    ss = st.session_state
    ss["bd_book"] = pd.concat([_book(), pd.DataFrame([row])], ignore_index=True)
    ss["bd_book_ver"] = ss.get("bd_book_ver", 0) + 1


def _add_gov(curves):
    ss = st.session_state
    cur = ss.get("bdb_cur", "USD")
    T = float(ss.get("bdb_T", 10))
    face = float(ss.get(f"bdb_face_{cur}") or 0)
    C = curves.get(cur) or _fallback_curve(cur)
    if face <= 0:
        return
    y = C.par_yield(T)
    _push({"이름": f"{CUR_NAME[cur]} {T:g}년", "통화": cur, "액면": face, "쿠폰(%)": round(y * 100, 3), "잔존만기(년)": T,
           "이자 횟수": 2, "신용스프레드(bp)": 0.0, "회수율(%)": 40.0})
    ss["bdb_msg"] = f"{CUR_NAME[cur]} {T:g}년 (쿠폰 {y * 100:.3f}%) 액면 {_money(face, cur)}을 추가했습니다."


def _add_corp():
    ss = st.session_state
    face = float(ss.get("bdc_face") or 0)
    if face <= 0:
        return
    cur = ss.get("bdc_cur", "KRW")
    _push({"이름": (ss.get("bdc_name") or "회사채").strip(), "통화": cur, "액면": face,
           "쿠폰(%)": float(ss.get("bdc_cpn", 4.0)), "잔존만기(년)": float(ss.get("bdc_T", 3.0)),
           "이자 횟수": int(ss.get("bdc_f", 4)), "신용스프레드(bp)": float(ss.get("bdc_spr", 80.0)),
           "회수율(%)": float(ss.get("bdc_rec", 40.0))})
    ss["bdb_msg"] = f"{ss.get('bdc_name') or '회사채'}를 추가했습니다."


def _add_panel(curves) -> None:
    kind = st.radio("추가할 채권", ["국채", "회사채 · 직접 입력"], horizontal=True, key="bdb_kind", label_visibility="collapsed")
    if kind == "국채":
        c1, c2, c3, c4 = st.columns([1.2, 1, 1.4, 0.8])
        c1.selectbox("종류", ["USD", "KRW"], key="bdb_cur", format_func=lambda c: CUR_NAME[c] + ("" if c in curves else " (금리 못 받음)"))
        c2.selectbox("잔존만기", MATS, index=5, key="bdb_T", format_func=lambda x: f"{x}년")
        cur = st.session_state.get("bdb_cur", "USD")
        c3.number_input("액면 금액" + (" (원)" if cur == "KRW" else " ($)"), value=100_000_000.0 if cur == "KRW" else 100_000.0,
                        step=10_000_000.0 if cur == "KRW" else 10_000.0, min_value=0.0, key=f"bdb_face_{cur}")
        c4.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        c4.button("추가", key="bdb_add_gov", on_click=_add_gov, args=(curves,), type="primary", use_container_width=True)
        st.caption("지금 곡선에서 액면가로 발행되는 국채(쿠폰 = 그 만기의 금리, 연 2회 이자)를 넣습니다. 실제 보유 종목은 아래 표에서 쿠폰·만기를 고치세요.")
    else:
        c1, c2, c3, c4 = st.columns([1.6, 0.8, 1.4, 1])
        c1.text_input("이름", key="bdc_name", placeholder="삼성전자 3년 회사채")
        c2.selectbox("통화", ["KRW", "USD"], key="bdc_cur")
        c3.number_input("액면 금액", value=100_000_000.0, step=10_000_000.0, min_value=0.0, key="bdc_face")
        c4.number_input("쿠폰(%)", value=4.0, step=0.05, format="%.3f", key="bdc_cpn")
        c5, c6, c7, c8, c9 = st.columns([1, 1, 1.2, 1, 0.8])
        c5.number_input("잔존만기(년)", value=3.0, min_value=0.1, step=0.5, key="bdc_T")
        c6.selectbox("이자 횟수(연)", [1, 2, 4, 12], index=2, key="bdc_f")
        c7.number_input("신용스프레드(bp)", value=80.0, step=5.0, key="bdc_spr", help="같은 만기 국채 금리 대비 더 받는 금리 (민평 금리 − 국고채)")
        c8.number_input("회수율(%)", value=40.0, step=5.0, min_value=0.0, max_value=100.0, key="bdc_rec", help="부도 시 액면 중 돌려받는 비율")
        c9.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        c9.button("추가", key="bdb_add_corp", on_click=_add_corp, type="primary", use_container_width=True)


def _table() -> None:
    ss = st.session_state
    ss["bd_book"] = st.data_editor(
        _book(), key=f"bd_book_ed_{ss.get('bd_book_ver', 0)}", num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={"통화": st.column_config.SelectboxColumn(options=["KRW", "USD"]),
                       "액면": st.column_config.NumberColumn(format="%.0f"),
                       "쿠폰(%)": st.column_config.NumberColumn(format="%.3f"),
                       "이자 횟수": st.column_config.SelectboxColumn(options=[1, 2, 4, 12]),
                       "신용스프레드(bp)": st.column_config.NumberColumn(format="%.0f", help="국채는 0"),
                       "회수율(%)": st.column_config.NumberColumn(format="%.0f")})


# ----------------------------------------------------------------------
# 평가
# ----------------------------------------------------------------------
def value_book(book: pd.DataFrame, curves: dict):
    """반환: {통화: dict(curve, rows(DataFrame), legs(list), tot, dv01, krd(금액), dur, conv)}"""
    from bonds import ytm_from_price

    out = {}
    b = book.dropna(how="all")
    for cur, g in b.groupby(b["통화"].fillna("KRW").astype(str).str.upper()):
        C = curves.get(cur)
        fb = C is None
        C = C or _fallback_curve(cur)
        rows, legs = [], []
        for _, r in g.iterrows():
            try:
                face = float(r["액면"])
                cpn = float(r["쿠폰(%)"]) / 100
                T = float(r["잔존만기(년)"])
                f = int(r["이자 횟수"]) if pd.notna(r["이자 횟수"]) else 2
                spr = float(r["신용스프레드(bp)"] or 0) / 1e4 if pd.notna(r["신용스프레드(bp)"]) else 0.0
                rec = float(r["회수율(%)"]) / 100 if pd.notna(r["회수율(%)"]) else 0.4
            except (TypeError, ValueError):
                continue
            if not (face > 0 and T > 0):
                continue
            t, cf, acc = bm.cashflows(cpn, T, f)
            R = bm.risk(C, t, cf, spr)
            dirty = R["price"]
            clean = dirty - acc
            ytm = ytm_from_price(clean, cpn, T, f)
            lam = bm.hazard_from_spread(spr, rec)
            k = face / 100
            legs.append({"t": t, "cf": cf * k, "spr": spr, "name": str(r["이름"])})
            rows.append({"이름": r["이름"], "평가액": dirty * k, "가격(액면 100)": clean, "만기수익률(%)": ytm * 100,
                         "듀레이션": R["dur"], "볼록성": R["conv"], "DV01 (1bp)": R["dv01"] * k,
                         "만기까지 부도확률(%)": (1 - math.exp(-lam * T)) * 100 if spr > 0 else 0.0, "_krd": R["krd"] * dirty * k * 1e-4})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        tot = float(df["평가액"].sum())
        out[cur] = {"curve": C, "fallback": fb, "rows": df, "legs": legs, "tot": tot, "dv01": float(df["DV01 (1bp)"].sum()),
                    "dur": float((df["듀레이션"] * df["평가액"]).sum() / tot), "conv": float((df["볼록성"] * df["평가액"]).sum() / tot),
                    "krd": np.sum(np.vstack(df["_krd"].values), axis=0)}
    return out


def _scen(V: dict) -> pd.DataFrame:
    C = V["curve"]
    rows = []
    for nm, dz in bm.SCENARIOS.items():
        new = sum(bm.price(C, lg["t"], lg["cf"], lg["spr"], dz=dz) for lg in V["legs"])
        rows.append({"시나리오": nm, "평가액 변화": new - V["tot"], "변화율(%)": (new / V["tot"] - 1) * 100})
    # 신용스프레드 확대 (회사채만)
    if any(lg["spr"] > 0 for lg in V["legs"]):
        new = sum(bm.price(C, lg["t"], lg["cf"], lg["spr"] + (0.01 if lg["spr"] > 0 else 0.0)) for lg in V["legs"])
        rows.append({"시나리오": "회사채 스프레드 +100bp", "평가액 변화": new - V["tot"], "변화율(%)": (new / V["tot"] - 1) * 100})
    return pd.DataFrame(rows)


def render_book(ts) -> None:
    from curve import curves as get_curves

    curves = get_curves(ts)
    b = _book()
    n = len(b.dropna(how="all"))
    st.caption("보유 채권을 넣으면 지금 금리 곡선으로 평가액·듀레이션·DV01, 만기 구간별 금리 민감도, 곡선 모양이 바뀔 때의 손익을 계산합니다.")
    exp, op = ui.more(f"채권 추가·수정 ({n}건)", key="bdb_more", expanded=(n == 0))
    if op:
        with exp:
            _add_panel(curves)
            st.divider()
            _table()
    msg = st.session_state.pop("bdb_msg", None)
    if msg:
        st.success(msg)
    b = _book()
    if not len(b.dropna(how="all")):
        st.info("위 '채권 추가·수정'에서 국채나 회사채를 넣으면 계산이 시작됩니다.")
        _immunize_panel(curves, None)
        return
    vals = value_book(b, curves)
    for cur, V in vals.items():
        with st.container(border=True):
            st.markdown(f"**{'원화 채권' if cur == 'KRW' else '달러 채권'}** · 곡선: {V['curve'].label}")
            if V["fallback"]:
                st.warning(f"{CUR_NAME.get(cur, cur)} 곡선을 받지 못해 {V['curve'].label}으로 계산했습니다.")
            ui.cards([("평가액 (경과이자 포함)", _money(V["tot"], cur)), ("듀레이션", f"{V['dur']:.2f}", "금리 1%p 오르면 약 이만큼 % 하락"),
                      ("DV01", _money(V["dv01"], cur), "금리가 모든 만기에서 1bp 오를 때 손실"),
                      ("볼록성", f"{V['conv']:.1f}")], cols=4)
            rows = V["rows"].drop(columns=["_krd"])
            st.dataframe(rows.style.format({"평가액": "{:,.0f}", "가격(액면 100)": "{:.3f}", "만기수익률(%)": "{:.3f}", "듀레이션": "{:.2f}",
                                            "볼록성": "{:.1f}", "DV01 (1bp)": "{:,.0f}", "만기까지 부도확률(%)": "{:.2f}"}),
                         hide_index=True, use_container_width=True)
            if (rows["만기까지 부도확률(%)"] > 0).any():
                st.caption("부도확률은 신용스프레드 ≈ 부도강도 × (1 − 회수율)로 구한 위험중립 확률입니다. 실제 부도율보다 높게 나오는 편입니다 "
                           "(스프레드에 유동성·위험 프리미엄이 섞여 있음).")
            c1, c2 = st.columns(2)
            with c1:
                lab = [f"{k:g}년" for k in bm.KEY_T]
                fig = go.Figure(go.Bar(x=lab, y=-V["krd"], marker_color=["#E5484D" if v > 0 else "#3B82F6" for v in V["krd"]],
                                       hovertemplate="%{x} 금리 1bp ↑ → %{y:,.0f}<extra></extra>"))
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), font=dict(size=11),
                                  title=dict(text="만기 구간별 1bp 상승 시 손익 (키 레이트)", font=dict(size=12)), yaxis=dict(title=cur))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                sc = _scen(V)
                st.markdown("**곡선 시나리오** (전부 다시 계산)")
                st.dataframe(sc.style.format({"평가액 변화": "{:,.0f}", "변화율(%)": "{:+.2f}"}), hide_index=True, use_container_width=True)
            st.caption("키 레이트: 1·2·5·10·20·30년 지점의 금리만 1bp 올렸을 때(이웃 지점까지 삼각형으로 줄어듦) 손익입니다. 막대를 다 더하면 DV01과 같습니다.")
    _immunize_panel(curves, vals)
    exp, op = ui.more("주식·옵션까지 합친 전체 위험 (VaR · 스트레스)", key="bdb_more_ra")
    if op:
        with exp:
            import riskall

            riskall.render("ra_b")


def _immunize_panel(curves, vals) -> None:
    exp, op = ui.more("면역화 — 정해진 날 필요한 금액을 금리와 상관없이 맞추기", key="bdb_more_imm")
    if not op:
        return
    with exp:
        c1, c2, c3 = st.columns([1, 1.4, 1])
        cur = c1.selectbox("통화", ["KRW", "USD"], key="imm_cur")
        L = c2.number_input("필요한 금액" + (" (원)" if cur == "KRW" else " ($)"), value=1_000_000_000.0 if cur == "KRW" else 1_000_000.0,
                            step=100_000_000.0 if cur == "KRW" else 100_000.0, min_value=1.0, key=f"imm_L_{cur}")
        H = c3.number_input("몇 년 뒤", value=7.0, min_value=0.5, max_value=29.0, step=0.5, key="imm_H")
        c4, c5 = st.columns(2)
        a = c4.selectbox("짧은 채권", [m for m in MATS if m < 30], index=1, key="imm_a", format_func=lambda x: f"{CUR_NAME[cur]} {x}년")
        bb = c5.selectbox("긴 채권", [m for m in MATS if m > 1], index=5, key="imm_b", format_func=lambda x: f"{CUR_NAME[cur]} {x}년")
        C = curves.get(cur) or _fallback_curve(cur)
        A = bm.cashflows(C.par_yield(a), a, 2)[:2]
        B = bm.cashflows(C.par_yield(bb), bb, 2)[:2]
        R = bm.immunize(C, L, H, A, B)
        if R is None:
            st.warning("목표 시점이 두 채권의 듀레이션 사이에 있어야 합니다. 더 짧은 채권과 더 긴 채권을 고르세요.")
            return
        ui.cards([("지금 필요한 돈 (부채 현재가치)", _money(R["pv_l"], cur)),
                  (f"{a}년 채권 매수 (액면)", _money(R["n_a"] * 100, cur), f"듀레이션 {R['Da']:.2f}"),
                  (f"{bb}년 채권 매수 (액면)", _money(R["n_b"] * 100, cur), f"듀레이션 {R['Db']:.2f}"),
                  ("볼록성: 채권 vs 부채", f"{R['conv_assets']:.1f} vs {R['conv_liab']:.1f}",
                   "채권 쪽이 크면 금리가 어느 쪽으로 움직여도 잉여 ≥ 0")], cols=4)
        st.dataframe(R["table"].style.format({"채권 가치": "{:,.0f}", "부채 현재가치": "{:,.0f}", "잉여": "{:+,.0f}"}),
                     hide_index=True, use_container_width=True)
        msg = ("두 채권의 현재가치 합 = 부채 현재가치, 듀레이션 = 목표 시점이 되도록 나눴습니다(레딩턴 면역). "
               "곡선이 평행 이동하면 잉여가 0 이상으로 남지만, 기울기가 바뀌면 깨질 수 있어 주기적으로 다시 맞춰야 합니다.")
        if vals and cur in vals:
            V = vals[cur]
            msg += f" 지금 보유한 {'원화' if cur == 'KRW' else '달러'} 채권의 듀레이션은 {V['dur']:.2f}로 목표 {H:g}년과 {V['dur'] - H:+.2f}년 차이입니다."
        st.caption(msg)


# ----------------------------------------------------------------------
# 콜옵션부 채권 (계산기 탭)
# ----------------------------------------------------------------------
def render_callable(ts) -> None:
    import hw
    from curve import curves as get_curves

    curves = get_curves(ts)
    c1, c2, c3, c4 = st.columns(4)
    cur = c1.selectbox("할인 곡선", ["USD", "KRW"], key="cb_cur", format_func=lambda c: CUR_NAME[c] + ("" if c in curves else " (평탄 가정)"))
    cpn = c2.number_input("쿠폰(%)", value=5.0, step=0.125, format="%.3f", key="cb_cpn") / 100
    T = c3.number_input("잔존만기(년)", value=10.0, min_value=0.5, max_value=30.0, step=0.5, key="cb_T")
    f = c4.selectbox("이자 횟수(연)", [1, 2, 4], index=1, key="cb_f", format_func=lambda x: f"연 {x}회")
    c5, c6, c7, c8 = st.columns(4)
    kind = c5.selectbox("옵션", ["call", "put"], key="cb_kind", format_func=lambda x: "콜 (발행사가 조기상환)" if x == "call" else "풋 (투자자가 조기상환 요구)")
    K = c6.number_input("행사 가격 (액면 100)", value=100.0, step=0.5, key="cb_K")
    fc = c7.number_input("첫 행사 가능 시점 (몇 년 뒤)", value=min(3.0, float(T)), min_value=0.0, max_value=float(T), step=0.5, key="cb_fc")
    mk = c8.number_input("시장 가격 (깨끗한 가격, 0이면 생략)", value=0.0, step=0.25, key="cb_mkt")
    c9, c10 = st.columns(2)
    a = c9.number_input("평균회귀 속도 a", value=0.05, min_value=0.001, max_value=1.0, step=0.01, format="%.3f", key="cb_a",
                        help="클수록 금리가 장기 평균으로 빨리 돌아옴")
    sig = c10.number_input("금리 변동성 σ (%p, 연)", value=1.0, min_value=0.05, max_value=5.0, step=0.1, key="cb_sig",
                           help="하루가 아니라 1년 동안 단기금리가 움직이는 표준편차. 미 국채는 보통 0.8~1.2%p") / 100
    C = curves.get(cur) or _fallback_curve(cur)
    with st.spinner("헐-화이트 트리로 계산하는 중..."):
        R = hw.callable_bond(C, cpn, T, f, K, fc, a, sig, kind=kind)
        E = hw.eff_risk(C, cpn, T, f, K, fc, a, sig, 0.0, kind=kind)
        s = hw.oas(C, mk + R["accrued"], cpn, T, f, K, fc, a, sig, kind=kind) if mk > 0 else None
    acc = R["accrued"]
    lab = "콜" if kind == "call" else "풋"
    ui.cards([("옵션 없는 채권 가치", f"{R['straight'] - acc:.3f}", "국채 곡선 기준 깨끗한 가격"),
              (f"{lab}옵션부 채권 가치", f"{R['callable'] - acc:.3f}"),
              (f"{lab}옵션 가치", f"{abs(R['option']):.3f}", "발행사가 가진 권리라 채권값에서 빠짐" if kind == "call" else "투자자 권리라 채권값에 더해짐"),
              ("OAS", f"{s * 1e4:,.0f}bp" if s is not None and np.isfinite(s) else "-",
               "시장가와 맞추는 데 필요한 추가 금리 (옵션 효과 뺀 신용·유동성 몫)" if mk > 0 else "시장 가격을 넣으면 계산")], cols=4)
    ui.cards([("유효 듀레이션 (옵션 없음)", f"{E['straight']['dur']:.2f}"), (f"유효 듀레이션 ({lab}옵션부)", f"{E['callable']['dur']:.2f}"),
              ("유효 볼록성 (옵션 없음)", f"{E['straight']['conv']:.1f}"), (f"유효 볼록성 ({lab}옵션부)", f"{E['callable']['conv']:.1f}",
                                                                      "음수면 금리가 내려도 값이 덜 오름" if kind == "call" else "")], cols=4)
    shifts = np.linspace(-0.03, 0.03, 25)
    import copy

    ys, yc = [], []
    for sh in shifts:
        c2_ = copy.copy(C)
        c2_.z = C.z + sh
        r_ = hw.callable_bond(c2_, cpn, T, f, K, fc, a, sig, kind=kind)
        ys.append(r_["straight"] - acc)
        yc.append(r_["callable"] - acc)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shifts * 1e4, y=ys, name="옵션 없는 채권", line=dict(color="#888", dash="dash")))
    fig.add_trace(go.Scatter(x=shifts * 1e4, y=yc, name=f"{lab}옵션부 채권", line=dict(color="#D4AF37", width=2.4)))
    fig.add_hline(y=K, line_dash="dot", line_color="#E5534B", annotation_text="행사 가격")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11), hovermode="x unified",
                      xaxis=dict(title="곡선 평행 이동 (bp)"), yaxis=dict(title="가격 (액면 100)"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"헐-화이트 1요인 모형(dr = (θ(t) − a·r)dt + σ dW)을 {C.label}에 정확히 맞춘 삼항 트리로 계산했습니다 "
               f"(단계 {R['tree'].N}개, 행사 가능일 {R['n_ex']}회 — {fc:g}년 뒤부터 이자 지급일마다). "
               + ("금리가 내려가면 발행사가 싸게 다시 빌리려고 조기상환하므로 가격이 행사 가격 근처에서 막힙니다(음의 볼록성)."
                  if kind == "call" else "금리가 오르면 투자자가 액면에 돌려받을 수 있어 가격 하락이 행사 가격에서 막힙니다."))


__all__ = [n for n in dir() if not n.startswith("__")]
