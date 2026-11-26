"""기업 리서치: 고른 재무비율로 '지표 양호 / 중립 / 지표 부진'를 판단하고, 같은 방법을 과거에 적용했을 때의 결과를 보여준다.

방법 (방법론 해설 문서의 '앱에 넣는다면' 설계)
- 수익성·안정성·효율성·현금흐름 비율: 전년보다 좋아졌으면 +1, 나빠졌으면 −1 (피오트로스키 F-스코어 방식)
- 성장률: 플러스 성장 +1, 마이너스 성장 −1
- 밸류에이션(PER 등): 고른 기간 안에서 그때까지의 값 중 상위 20%면 −1, 하위 20%면 +1 (자기 과거 대비 평균회귀)
- 점수 합 ÷ 근거 수가 +0.25 이상이면 지표 양호, −0.25 이하면 지표 부진
과거 검증: 기간 안의 매달 첫 거래일마다 그때까지 공시된 숫자로 다시 판정하고 그 뒤 1·3·6개월·1년 주가 수익률을 붙인다.
기대수익률: 과거에 지금과 같은 판정이었던 달들의 이후 수익률 평균 (사례가 적으면 기간 전체 평균 쪽으로 당김).
업종 유형(성장·기술 / 경기민감 / 금융 / 리츠 / 방어)에 따라 지표 가중치와 PER을 읽는 방법을 바꾼다 (PROFILES).
추가 신호: 분기 TTM 재무, 모멘텀(12-1개월), 실적 서프라이즈, 동종업계 대비 백분위, 위험 신호 거부권(Altman Z''·Beneish M).
기대수익률 = CAPM(무위험 + 베타 × 시장 위험프리미엄) + 같은 판정이었던 과거의 평균 알파(사례 수만큼만 반영). 워크포워드로 검증.
매수·매도 점수(0~100): CAPM 미포함(6개월 수익률이 플러스였던 비율) · 업종 대비(섹터 ETF를 이긴 비율) · CAPM 대비(CAPM을 이긴 비율) ·
미래 추정치(예상 EPS 성장 + PER 회귀 + 배당 → 6개월 수익률이 플러스일 확률). 80% 구간이 20점보다 넓으면 범위로 표시.
추가 신호(영상 방법론, 업종별 검증 통과분만 EXTRA_ON): 고점 대비 하락 · 현금 품질 · 주주환원 · 외국인 수급(국내).
밸류에이션은 후행이므로 외국인 지분율이 늘고 있으면 PER 등이 비싸도 감점하지 않는다.
"""
# 파일 구성: outlook_rules.py(규칙) → outlook_data.py(데이터) → outlook_signals.py(신호·검증) → outlook_score.py(점수·게이지)
#   → 이 파일(화면). 예전처럼 outlook.이름 으로 모든 함수를 쓸 수 있다.
import numpy as np
import pandas as pd
import streamlit as st

from research_data import RATIOS, _split_factor, annual_eps_table, compute_ratios

from outlook_score import *  # noqa: F401,F403

# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
def render_outlook(sec, R: dict, ticker: str, sel: list, daily: dict, options=None, ms_key=None, domestic=False, fwd=None) -> None:
    """sel: 고른 연간 재무비율 이름, daily: {표시 이름: (일별 시계열, 단위)} — 고른 일별 밸류에이션."""
    from glossary import table_html
    from ui import mgrid

    st.markdown("##### 상승·하락 판단")
    domestic = bool(domestic or "_ksec" in R)
    hist = R.get("hist")
    if sec is None or not isinstance(hist, pd.Series) or not len(hist):
        st.caption("재무제표나 주가가 없어 판단할 수 없습니다.")
        return

    # ---- 업종 유형
    info = R.get("info") or {}
    det = classify(info.get("sector"), info.get("industry"), R.get("_ksec"))
    sic_txt = ""
    if det == "general" and not R.get("_ksec") and not info.get("sector"):  # Yahoo가 막혔으면 SEC 업종 코드로
        try:
            sic, sic_txt = sec_sic(ticker)
            det = classify_sic(sic)
            sic_txt = f"SEC 업종 코드 {sic} {sic_txt}".strip() if sic else ""
        except Exception:  # noqa: BLE001
            sic_txt = ""
    keys = list(PROFILES)
    c1, c2 = st.columns([2, 3])
    prof = c1.selectbox("업종 유형", keys, index=keys.index(det), key=f"rs_prof_{ticker}",
                        format_func=lambda k: PROFILES[k]["name"] + (" (자동 판별)" if k == det else ""))
    P = PROFILES[prof]
    src = " · ".join(x for x in (info.get("sector"), info.get("industry"), R.get("_ksec"), sic_txt) if x)
    c2.caption((f"업종: {src}. " if src else "") + P["why"])
    if options is not None and ms_key:
        rec = [k for k in P["rec"] if k in options]
        if rec:
            names = {"PER (직전 사업연도 EPS)": "PER", "12M 선행 PER (컨센서스 추이)": "12M PER"}
            st.caption("이 업종에서 볼 지표: " + ", ".join(names.get(k, k) for k in rec if k != "주가"))
            st.button("이 지표들로 바꾸기", key=f"rs_rec_{ticker}_{prof}", on_click=_set_rec, args=(ms_key, rec))

    with st.expander("판정에 넣을 신호"):
        o1, o2 = st.columns(2)
        opts = dict(
            quarterly=o1.toggle("분기 실적으로 갱신 (최근 12개월)", value=not domestic, key=f"rs_o_q_{ticker}", disabled=domestic,
                                help="연간 재무제표 대신 분기마다 최근 12개월 합계로 재무비율을 다시 계산합니다 (미국 SEC 공시 종목)."),
            momentum=o1.toggle("모멘텀 (12-1개월 수익률)", value=True, key=f"rs_o_m_{ticker}",
                               help="최근 12개월 수익률(직전 1개월 제외)이 지수보다 5%p 넘게 높으면 상승 쪽, 낮으면 하락 쪽."),
            surprise=o1.toggle("실적 서프라이즈", value=True, key=f"rs_o_s_{ticker}",
                               help="최근 분기 실적이 발표 직전 컨센서스보다 얼마나 좋았나 (발표 후 91일 동안). 이 회사의 과거 서프라이즈와 비교합니다."),
            peers=o2.toggle("동종업계 비교", value=False, key=f"rs_o_p_{ticker}", disabled=domestic,
                            help="고른 재무비율이 동종업계 비교 탭의 회사들 가운데 상위·하위 20%인지. 처음 켜면 회사들 재무제표를 받느라 30초쯤 걸립니다."),
            flags=o2.toggle("위험 신호 거부권 (Altman Z'' · Beneish M)", value=True, key=f"rs_o_f_{ticker}",
                            help="부실 위험(Z'' 1.1 미만)이나 이익 조작 의심(M −1.78 초과)이면 '지표 양호'를 '중립'으로 낮춥니다."),
            extra=o2.toggle("추가 신호 (고점 대비 하락 · 현금 품질 · 주주환원 · 외국인 수급)", value=True,
                            key=f"rs_o_x_{ticker}",
                            help="업종별 대표 종목 검증에서 효과가 있었던 업종에만 씁니다. "
                                 "고점 대비 하락(경기민감형·국내 제외): 52주 고점보다 15% 넘게 빠지고 3개월 마이너스면 부정 신호 · "
                                 "현금 품질(성장·기술형): 영업현금흐름 ÷ 순이익 1.1배 이상 긍정 신호, 0.8배 미만 부정 신호 · "
                                 "주주환원(성장·기술형, 경기민감형): 배당+자사주 매입이 잉여현금흐름의 50% 이상이고 시가총액의 3% 이상이면 긍정 신호 · "
                                 "외국인 수급(국내): 60거래일 외국인 지분율이 0.5%p 넘게 늘면 긍정 신호, 줄면 부정 신호이고, 늘 때는 PER 등이 비싸도 감점 안 함. "
                                 "이익 고점 위치와 PBR ÷ ROE는 검증에서 효과가 없어 뺐습니다. 조건에 걸릴 때만 근거에 들어갑니다."),
        )
        if domestic:
            st.caption("국내 종목은 분기 재무와 동종업계 비교를 아직 지원하지 않습니다 (DART 연간 재무제표 기준).")

    use = [k for k in sel if k in DIR or k in GROWTH]
    no_dir = [k for k in sel if k not in use]
    if not use and not daily and not opts["momentum"]:
        st.info("위 '그래프 항목'에서 재무비율이나 PER 같은 밸류에이션을 하나 이상 고르면 판단합니다.")
        return
    peers = None
    if opts["peers"]:
        try:
            from peers import us_peer_list

            peers, _desc = us_peer_list(ticker, info)
        except Exception:  # noqa: BLE001
            peers = None
    import stats

    rf_now, rf_lab = stats.risk_free_for(domestic, pd.Timestamp.now().date())
    with st.spinner("과거 시점마다 다시 판정하는 중..."):
        c = build_ctx(sec, R, ticker, daily, opts, domestic=domestic, peers=peers)
        rows, signals, skipped, flags = evaluate(c, prof, use, rf_now)
    if not rows:
        st.caption("고른 기간 안에 판정할 수 있는 시점이 없습니다.")
        return

    # ---- 지금 판단
    h = c["h"]
    now_d = min(c["end"], h.index[-1])
    now_sig = signals(now_d)
    sc, n = _score(now_sig)
    vd = _verdict(sc, n)
    if vd is None:
        st.caption("고른 항목의 값이 부족해 판단할 수 없습니다 (전년 값이 없거나 비교할 과거 값이 적음).")
        return
    fl_now = _flags_at(flags, c, now_d) if flags else []
    capped = bool(fl_now) and vd == "지표 양호"
    if capped:
        vd = "중립"
    msg = (f"**{vd}**{' (위험 신호로 낮춤)' if capped else ''} — 점수 {sc:+.1f} / {n:.1f} "
           f"({P['name']} 기준 · {c['freq']} 재무 · {now_d:%Y-%m-%d})")
    __import__("ui").verdict(msg, "up" if vd == "지표 양호" else "down" if vd == "지표 부진" else "")
    if fl_now:
        st.warning("위험 신호: " + ", ".join(fl_now))

    # ---- 미래 추정치 점수 (CAPM 미포함·업종 대비·CAPM 대비 원형 점수는 이 섹션 위에 표시)
    try:
        rf_ = forward_score(c, R, fwd, prof)
    except Exception as e:  # noqa: BLE001
        rf_ = dict(err=f"계산 실패: {e}")
    k1, _k2 = st.columns([1, 2])
    if "err" in rf_:
        k1.metric("미래 추정치 점수 (6개월)", "계산 불가", delta_color="off")
        k1.caption(rf_["err"])
    else:
        _show_score(k1, "미래 추정치 점수 (6개월)", rf_, rf_["reliable"],
                    f"1년 기대수익률 {_p(rf_['mu'])}" + ("" if rf_["consensus"] else " (컨센서스 없음 → 과거 성장률 사용)") + ". 과거 검증 없음.")
    _k2.caption("예상 EPS 성장과 PER의 기준값 회귀로 낸 1년 기대수익률을 변동성으로 나눠, 6개월 수익률이 플러스일 확률로 바꾼 값입니다. "
                "과거 데이터로 계산하는 세 가지 점수(CAPM 미포함·업종 대비·CAPM 대비)는 이 그래프 섹션 바로 위 원형 점수에 있습니다.")
    if "err" not in rf_:
        with st.expander("미래 추정치 계산 내역"):
            mgrid([("현재 PER", f"{rf_['per_now']:.1f}배", f"주가 ÷ EPS({rf_['eps_src']}) {rf_['eps_ttm']:.2f}"),
                   ("기준 PER", f"{rf_['target']:.1f}배" if np.isfinite(rf_["target"]) else "-", "고른 기간의 PER 중앙값"),
                   ("12개월 예상 EPS", f"{rf_['n_eps']:.2f}" if rf_["n_eps"] else "-", "컨센서스"),
                   ("배당수익률", _p(rf_["dy"]).lstrip("+"), "최근 사업연도 배당금 ÷ 시가총액"),
                   ("변동성 (연)", _p(rf_["sig"]).lstrip("+"), "최근 2년 주간 수익률")])
            tb_ = [[x["g"], _p(x["gv"]), "회귀 안 함" if x["k"] is None else f"{x['k']}년", _p(x["mu"]), f"{x['score']:.0f}점"]
                   for x in rf_["scen"]]
            st.markdown(table_html(tb_, ["EPS 성장 가정", "성장률", "PER 회귀 기간", "1년 기대수익률", "점수"]), unsafe_allow_html=True)
            for t_ in rf_.get("notes", []):
                st.caption(t_)
            st.caption("1년 기대수익률 = (1 + EPS 성장률) × (기준 PER ÷ 현재 PER)^(1/회귀 기간) − 1 + 배당수익률 "
                       "(기준 PER ÷ 현재 PER은 0.67~1.5배로 제한). 과거 시점의 컨센서스 기록이 없어 이 점수는 과거 검증을 하지 못했습니다.")
    if c["peers"]:
        st.caption("동종업계: " + ", ".join(p for p, _ in c["peers"]))

    # ---- 향후 기대수익률 = CAPM 기대수익률 + 판정에 따른 보정(과거 같은 판정의 평균 알파, 사례 수만큼만 반영)
    beta_now = _beta(c, now_d)
    st.markdown("**향후 기대수익률**")
    ftb = []
    for lab, days in HORIZONS:
        capm = (1 + rf_now + beta_now * MRP) ** (days / 365) - 1
        same = [r for r in rows if r["verdict"] == vd and np.isfinite(r[lab])]
        a, ns = _alpha_est([r["a_" + lab] for r in same], days)
        base = [r["a_" + lab] for r in same] if len(same) >= 3 else [r["a_" + lab] for r in rows if np.isfinite(r[lab])]
        base = [x for x in base if np.isfinite(x)]
        rng = f"{_p(capm + np.percentile(base, 25))} ~ {_p(capm + np.percentile(base, 75))}" if base else "-"
        up = f"{np.mean([r[lab] > 0 for r in same]) * 100:.0f}%" if same else "-"
        ftb.append([lab, _p(capm + a), _p(capm), _p(a), up, rng, f"{ns}"])
    st.markdown(table_html(ftb, ["기간", "예측 기대수익률", "CAPM 기준", "판정 보정", "상승 확률", "예상 범위", "같은 판정 사례"]),
                unsafe_allow_html=True)
    st.caption(f"CAPM 기준 = 무위험수익률 {rf_now * 100:.2f}% + 베타 {beta_now:.2f} × 시장 위험프리미엄 {MRP * 100:.0f}% (가정). "
               f"판정 보정 = 과거 판정이 '{vd}'인 달들의 시장 대비 초과수익(알파) 평균을, 사례가 적을수록 0 쪽으로 당긴 값. "
               "예상 범위 = 그 알파의 하위 25%~상위 25%를 CAPM 기준에 더한 값. 무위험수익률: " + rf_lab)

    # ---- 워크포워드 성적
    wf = walk_forward(rows)
    wtb = []
    for lab, _days in HORIZONS:
        w = wf[lab]
        if not w["n"]:
            continue
        wtb.append([lab, _p(w["mae_m"]).lstrip("+"), _p(w["mae_c"]).lstrip("+"),
                    f"{w['hit']}/{w['nh']} ({w['hit'] / w['nh'] * 100:.0f}%)" if w["nh"] else "-", f"{w['n']}"])
    if wtb:
        st.markdown("**워크포워드 성적** — 각 시점에 그때까지 결과가 나온 사례만으로 예측했을 때")
        st.markdown(table_html(wtb, ["기간", "예측 오차 (이 방법)", "예측 오차 (CAPM만)", "초과수익 방향 적중", "시점 수"]),
                    unsafe_allow_html=True)
        st.caption("예측 오차 = 예측 기대수익률과 실제 수익률 차이의 평균 (작을수록 좋음). 이 방법의 오차가 CAPM만 쓴 오차보다 작아야 판정이 쓸모 있습니다.")

    past = [r for r in rows if np.isfinite(r["1년"])]
    hit, nh = _hit_rate(rows)
    items = [("점수 방향 적중률", f"{hit}/{nh} ({hit / nh * 100:.0f}%)" if nh else "-",
              f"{P['name']} 기준 · 점수 부호와 이후 1년 방향이 같았던 달의 비율")]
    if prof != "general":
        rows_g = evaluate(c, "general", use, rf_now)[0]
        bh, bn_ = _hit_rate(rows_g)
        items.append(("업종 반영 없이 적중률", f"{bh}/{bn_} ({bh / bn_ * 100:.0f}%)" if bn_ else "-", "같은 지표를 일반 기준으로 판정했을 때"))
    items.append(("베타", f"{beta_now:.2f}", f"최근 2년 주간 수익률 · {c['mname'] or '지수'} 기준 (블룸 조정)"))
    fs = fscore(sec, R.get("splits"))
    fs.index = [pd.Timestamp(x) for x in fs.index]
    if len(fs) and fs["항목"].iloc[-1] >= 5:
        items.append(("F-스코어", f"{int(fs['점수'].iloc[-1])}/{int(fs['항목'].iloc[-1])}", "재무 상태 개선 항목 수 (참고)"))
    mgrid(items)

    # ---- 근거별 신호
    trs = []
    for k, (s, desc, wt) in now_sig.items():
        hp_ = [r for r in past if k in r["sig"] and r["sig"][k][0] != 0 and r["1년"] != 0]
        hk = sum(1 for r in hp_ if np.sign(r["sig"][k][0]) == np.sign(r["1년"]))
        trs.append([k, desc, _sig_txt(s), _w(wt), f"{hk}/{len(hp_)}" if hp_ else "-"])
    st.markdown(table_html(trs, ["근거", "값", "신호", "가중치", "과거 적중 (1년)"]), unsafe_allow_html=True)
    notes = []
    if any("(성장 대비)" in k for k in now_sig):
        notes.append("(성장 대비) = PER ÷ 최근 3년 EPS 연평균 성장률(%). 이익이 늘지 않았으면 계산하지 않습니다.")
    if "평균이익 PER" in now_sig:
        notes.append("경기민감형이라 PER 대신 '평균이익 PER'(주가 ÷ 최근 최대 10년 평균 EPS)로 판단했습니다. 이익 고점의 낮은 PER에 속지 않기 위해서입니다.")
    for k, why in skipped.items():
        notes.append(f"{k}: 판단에서 뺐습니다 — {why}.")
    if no_dir:
        notes.append(f"{', '.join(no_dir)}: 높고 낮음의 좋고 나쁨이 정해져 있지 않아 판단에서 뺐습니다.")
    tr = R.get("eps_trend")
    if isinstance(tr, pd.DataFrame) and {"current", "90daysAgo"} <= set(tr.columns):
        parts = []
        for per, nm in (("0y", "올해"), ("+1y", "내년")):
            try:
                a_, b_ = float(tr.loc[per, "current"]), float(tr.loc[per, "90daysAgo"])
                if b_:
                    parts.append(f"{nm} {(a_ / b_ - 1) * 100:+.1f}%")
            except Exception:  # noqa: BLE001
                pass
        if parts:
            notes.append("실적 추정치 변화 (최근 90일 컨센서스): " + ", ".join(parts)
                         + ". 과거 시점의 추정치 기록이 없어 점수와 검증에는 넣지 않고 참고로만 보여줍니다.")
    if any(k.split(" (")[0] in VALN or "PER" in k or "PBR" in k or k in daily for k in now_sig):
        notes.append("밸류에이션(PER·PBR 등)은 후행 지표입니다. 이익이 바닥일 때는 비싸 보이고 고점일 때는 싸 보입니다."
                     + (" 그래서 외국인 지분율이 늘고 있으면 비싸 보여도 감점하지 않습니다." if domestic and opts.get("extra") else ""))
    if opts.get("extra") and domestic and c.get("frgn") is None:
        notes.append(f"외국인 수급 자료를 받지 못해 이 신호는 뺐습니다 ({c.get('frgn_err') or '자료 없음'}).")
    for t in notes:
        st.caption(t)

    # ---- 판정별 평균과 과거 판정 내역
    with st.expander("과거 판정과 이후 수익률"):
        sm = []
        for v_ in ("지표 양호", "중립", "지표 부진"):
            g_ = [r for r in rows if r["verdict"] == v_]
            line = [v_, f"{len(g_)}"]
            for lab, _d in HORIZONS:
                x = [r[lab] for r in g_ if np.isfinite(r[lab])]
                line.append(f"{_p(np.mean(x))} ({sum(1 for y in x if y > 0)}/{len(x)})" if x else "-")
            sm.append(line)
        st.markdown(table_html(sm, ["판정", "횟수"] + [f"{lab} 평균" for lab, _ in HORIZONS]), unsafe_allow_html=True)
        st.caption("괄호는 오른 횟수/전체 횟수입니다.")
        df = pd.DataFrame([[f"{r['d']:%Y-%m-%d}", f"{r['score']:+.1f}/{r['n']:.1f}", r["verdict"] + (" ⚠" if r["flags"] else "")]
                           + [_p(r[lab]) for lab, _ in HORIZONS] for r in reversed(rows)],
                          columns=["판정일", "점수", "판정"] + [lab for lab, _ in HORIZONS])
        st.dataframe(df, hide_index=True, use_container_width=True, height=320)
        st.caption(f"매달 첫 거래일에 그때까지 공시된 숫자({c['freq']})와 그날의 주가로 다시 판정했습니다. ⚠ = 위험 신호. "
                   "수익률은 판정일 종가부터 1·3·6개월, 1년 뒤 종가까지입니다 (배당 제외).")
        own = sig_tally(rows)
        if own:
            stb = [[k, f"{v[0]}/{v[1]}", f"{v[0] / v[1] * 100:.0f}%", _p(v[2] / v[1])]
                   for k, v in sorted(own.items(), key=lambda kv: -kv[1][1]) if v[1]]
            st.markdown("**이 종목의 신호별 성적** — 신호가 켜진 달의 이후 6개월 CAPM 대비 초과수익 방향")
            st.markdown(table_html(stb, ["신호", "적중/사례", "적중률", "평균 초과수익 (신호 방향)"]), unsafe_allow_html=True)

    # ---- 여러 종목 검증
    with st.expander(f"여러 종목으로 '{P['name']}' 규칙 검증"):
        st.caption(f"대표 종목 {', '.join(VAL_TICKERS[prof])}에 이 업종에 쓰는 지표·분기 재무·모멘텀·위험 신호를 똑같이 적용합니다 "
                   "(최근 10년, 동종업계 비교·실적 서프라이즈 제외). 처음 실행은 1~2분 걸리고 결과는 1주일 저장됩니다.")
        run_key = f"rs_val_{prof}"
        st.button("검증 실행", key=f"rs_valbtn_{ticker}_{prof}", on_click=_val_on, args=(prof,))
        if st.session_state.get(run_key):
            with st.spinner("대표 종목들을 다시 판정하는 중..."):
                res = validate_profile(prof)
            ok = [r for r in res if "err" not in r]
            tb = [[r["종목"], f"{r['hit_p']}/{r['n_p']}", f"{r['hit_g']}/{r['n_g']}", _p(r["mae_m"]).lstrip("+"),
                   _p(r["mae_c"]).lstrip("+")] for r in ok]
            if ok:
                sp = sum(r["hit_p"] for r in ok), sum(r["n_p"] for r in ok)
                sg_ = sum(r["hit_g"] for r in ok), sum(r["n_g"] for r in ok)
                mm = np.nanmean([r["mae_m"] for r in ok])
                mc = np.nanmean([r["mae_c"] for r in ok])
                tb.append(["합계·평균", f"{sp[0]}/{sp[1]} ({sp[0] / max(sp[1], 1) * 100:.0f}%)",
                           f"{sg_[0]}/{sg_[1]} ({sg_[0] / max(sg_[1], 1) * 100:.0f}%)", _p(mm).lstrip("+"), _p(mc).lstrip("+")])
                st.markdown(table_html(tb, ["종목", f"적중 ({P['name']})", "적중 (일반 기준)", "1년 예측 오차 (이 방법)",
                                            "1년 예측 오차 (CAPM만)"]), unsafe_allow_html=True)
            agg = {}
            for r in ok:
                for k, (hk, nk, sa) in (r.get("sig") or {}).items():
                    t = agg.setdefault(k, [0, 0, 0.0])
                    t[0], t[1], t[2] = t[0] + hk, t[1] + nk, t[2] + sa
            if agg:
                stb = [[k, f"{v[0]}/{v[1]}", f"{v[0] / v[1] * 100:.0f}%", _p(v[2] / v[1])]
                       for k, v in sorted(agg.items(), key=lambda kv: -kv[1][1]) if v[1]]
                st.markdown("**신호별 성적** — 신호가 켜진 달의 이후 6개월 CAPM 대비 초과수익 방향")
                st.markdown(table_html(stb, ["신호", "적중/사례", "적중률", "평균 초과수익 (신호 방향)"]), unsafe_allow_html=True)
                st.caption("적중률이 50% 근처이고 평균 초과수익이 0 근처면 그 신호는 이 업종에서 효과가 없다는 뜻입니다. "
                           "매달 판정하므로 이웃한 달끼리 기간이 겹쳐 실제 독립 사례는 더 적습니다.")
            bad = [r for r in res if "err" in r]
            if bad:
                st.caption("불러오지 못한 종목: " + ", ".join(f"{r['종목']}({r['err']})" for r in bad))

    st.caption("과거 통계로 낸 추정이며 예측을 보장하지 않습니다. 매달 판정하므로 이웃한 달의 수익률은 기간이 겹쳐, "
               "실제로 독립된 사례 수는 표의 횟수보다 적습니다.")
