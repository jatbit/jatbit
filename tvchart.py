"""시계열 차트를 트레이딩뷰처럼 조작하게 만든다.

- 끌면 좌우 이동(팬), 마우스 휠은 가로로만 확대·축소(커서 위치 기준)
- 보이는 구간이 바뀌면 세로축을 그 구간 데이터에 맞춰 자동 조정
- 아래에 스크롤바(레인지 슬라이더): 손잡이로 구간 길이 조절, 가운데를 끌어 이동
- 더블클릭하면 전체 기간으로 복귀

Streamlit 기본 차트는 이벤트에 자바스크립트를 붙일 수 없어서, 날짜 축 차트만 Plotly.js를 직접 띄운 작은 iframe으로 그린다.
ui.py가 불러올 때 st.plotly_chart를 감싸서 앱 전체 시계열 차트에 자동 적용된다.
"""
from __future__ import annotations

import datetime as _dt
import json
import re

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:  # noqa: BLE001
    go = None

_NO_X = {"pie", "treemap", "sunburst", "icicle", "funnelarea", "sankey", "table", "parcoords", "parcats",
         "indicator", "choropleth", "scattergeo", "scatterpolar", "barpolar", "scatter3d", "surface", "mesh3d"}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _js_version() -> str:
    try:
        from plotly.offline import get_plotlyjs_version
        v = get_plotlyjs_version()
        if v:
            return v
    except Exception:  # noqa: BLE001
        pass
    return "2.35.2"


def _dateish(v) -> bool:
    if v is None:
        return False
    try:
        if len(v) == 0:
            return False
        first = v[0]
    except Exception:  # noqa: BLE001
        return False
    if isinstance(first, (pd.Timestamp, np.datetime64, _dt.date, _dt.datetime)):
        return True
    return isinstance(first, str) and bool(_DATE_RE.match(first))


def is_time_figure(fig) -> bool:
    """x축이 날짜인 선·봉 차트인가 (원형·트리맵 등은 제외)."""
    try:
        data = list(fig.data)
    except Exception:  # noqa: BLE001
        return False
    if not data:
        return False
    n_date = 0
    for t in data:
        tp = getattr(t, "type", "") or ""
        if tp in _NO_X:
            return False
        if getattr(t, "orientation", None) == "h":
            return False
        if _dateish(getattr(t, "x", None)):
            n_date += 1
    if n_date == 0:
        return False
    # 점이 아주 적으면(연간 막대 몇 개 등) 굳이 바꾸지 않는다
    longest = max((len(t.x) for t in data if getattr(t, "x", None) is not None and _dateish(t.x)), default=0)
    return longest >= 30


def _axis_keys(lay: dict, prefix: str) -> list:
    keys = [k for k in lay if re.fullmatch(prefix + r"\d*", k)]
    return keys or [prefix]


def _bottom_xaxis(lay: dict) -> str:
    """가장 아래 줄 그래프의 x축 (스크롤바를 붙일 곳)."""
    best, best_y = "xaxis", 9.0
    for xk in _axis_keys(lay, "xaxis"):
        anchor = (lay.get(xk) or {}).get("anchor") or "y"
        if not str(anchor).startswith("y"):
            continue
        yk = "yaxis" + anchor[1:]
        dom = (lay.get(yk) or {}).get("domain") or [0, 1]
        if dom[0] < best_y:
            best, best_y = xk, dom[0]
    return best


THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Pretendard, 'Noto Sans KR', sans-serif", color="#ECE6D8"),
    hoverlabel=dict(bgcolor="#1A1918", bordercolor="#C9A44C", font=dict(color="#ECE6D8")),
)


def render(fig, height: int | None = None) -> None:
    import streamlit.components.v1 as components

    f = go.Figure(fig)
    lay0 = f.layout.to_plotly_json()
    h = int(height or lay0.get("height") or 450)
    xkeys = _axis_keys(lay0, "xaxis")
    ykeys = _axis_keys(lay0, "yaxis")
    # 세로축 맞춤 방식: 코드에서 눈금까지 정한 축(RSI 0~100 등)은 고정, 범위만 정한 축은 극단값을 뺀 분위수로, 나머지는 최소~최대
    modes = {}
    for yk in ykeys:
        y = lay0.get(yk) or {}
        if y.get("range") is not None and y.get("tickvals") is not None:
            modes[yk] = "fixed"
        elif y.get("range") is not None:
            modes[yk] = "robust"
        else:
            modes[yk] = "minmax"
    bx = _bottom_xaxis(lay0)
    slider_h = 46
    f.update_layout(template="plotly_dark", dragmode="pan", height=h + slider_h, **THEME)
    for xk in xkeys:
        f.layout[xk].update(gridcolor="#2A2620", zeroline=False, showspikes=True, spikemode="across", spikesnap="cursor",
                            spikethickness=1, spikecolor="#8C877C", spikedash="dot",
                            rangeslider=dict(visible=False))
    f.layout[bx].update(rangeslider=dict(visible=True, thickness=0.06, bgcolor="#141413", bordercolor="#2E2A22", borderwidth=1))
    for yk in ykeys:
        f.layout[yk].update(fixedrange=True, gridcolor="#2A2620", zerolinecolor="#3A352B")
    m = dict(f.layout.margin.to_plotly_json()) if f.layout.margin else {}
    f.update_layout(margin=dict(l=m.get("l", 10), r=m.get("r", 10), t=m.get("t", 10), b=max(m.get("b", 10), 10)))

    spec = f.to_json()
    ver = _js_version()
    total = h + slider_h + 6
    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<style>html,body{margin:0;padding:0;background:transparent;overflow:hidden}#c{width:100%;height:__H__px}
</style>
<script src="https://cdn.plot.ly/plotly-__VER__.min.js"></script></head>
<body><div id="c"></div>
<script>
(function(){
const SPEC = __SPEC__;
const MODES = __MODES__;
const gd = document.getElementById('c');
const ms = v => { if (v === null || v === undefined) return NaN; if (typeof v === 'number') return v;
  const s = String(v); return Date.parse(s.length > 10 ? s.replace(' ', 'T') : s); };
let cache = null, fitting = false;
function prep(){
  cache = (gd._fullData || []).map(t => {
    if (t.visible !== true) return null;
    const xs = t.x ? Array.from(t.x, ms) : null;
    if (!xs) return null;
    const ys = [];
    if (t.type === 'candlestick' || t.type === 'ohlc') { ys.push(Array.from(t.high || [])); ys.push(Array.from(t.low || [])); }
    else if (t.y) { ys.push(Array.from(t.y)); }
    return {xa: 'xaxis' + (t.xaxis || 'x').slice(1), ya: 'yaxis' + (t.yaxis || 'y').slice(1), xs, ys, bar: t.type === 'bar'};
  });
}
function q(a, p){ const i = (a.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i); return a[lo] + (a[hi] - a[lo]) * (i - lo); }
function fit(){
  if (!cache) prep();
  const L = gd._fullLayout, upd = {}, vals = {};
  cache.forEach(c => {
    if (!c || !MODES[c.ya] || MODES[c.ya] === 'fixed') return;
    const xa = L[c.xa]; if (!xa || !xa.range) return;
    const a = ms(xa.range[0]), b = ms(xa.range[1]);
    const arr = vals[c.ya] || (vals[c.ya] = []);
    if (c.bar) arr.push(0);
    for (let i = 0; i < c.xs.length; i++) {
      const x = c.xs[i]; if (!(x >= a && x <= b)) continue;
      for (const ys of c.ys) { const y = +ys[i]; if (isFinite(y)) arr.push(y); }
    }
  });
  for (const ya in vals) {
    let v = vals[ya]; if (v.length < 2) continue;
    const log = (L[ya] && L[ya].type === 'log');
    if (log) v = v.filter(x => x > 0).map(x => Math.log10(x));
    if (v.length < 2) continue;
    let lo, hi;
    if (MODES[ya] === 'robust') { v.sort((p, r) => p - r); lo = q(v, 0.01); hi = q(v, 0.99); }
    else { lo = Math.min.apply(null, v); hi = Math.max.apply(null, v); }
    const pad = (hi - lo) * 0.07 || Math.abs(hi) * 0.05 || 1;
    upd[ya + '.range'] = [lo - pad, hi + pad];
  }
  if (Object.keys(upd).length) { fitting = true; Plotly.relayout(gd, upd).then(() => { fitting = false; }, () => { fitting = false; }); }
}
let timer = null;
Plotly.newPlot(gd, SPEC.data, SPEC.layout, {scrollZoom: true, displaylogo: false, responsive: true, doubleClick: 'reset',
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'zoomIn2d', 'zoomOut2d', 'toggleSpikelines']}).then(() => {
  prep();
  gd.on('plotly_relayout', ev => {
    if (fitting) return;
    if (!Object.keys(ev).some(k => k.indexOf('xaxis') === 0)) return;
    clearTimeout(timer); timer = setTimeout(fit, 30);
  });
  gd.on('plotly_doubleclick', () => { setTimeout(fit, 60); });
});
})();
</script></body></html>"""
    html = (html.replace("__H__", str(h + slider_h)).replace("__VER__", ver)
            .replace("__SPEC__", spec.replace("</", "<\\/")).replace("__MODES__", json.dumps(modes)))
    components.html(html, height=total, scrolling=False)


def install() -> None:
    """st.plotly_chart를 감싸서 날짜 축 차트는 위 방식으로 그린다 (그 밖의 차트는 원래대로)."""
    if go is None:
        return
    orig = getattr(st, "_tv_orig_plotly_chart", None) or st.plotly_chart
    st._tv_orig_plotly_chart = orig

    def plotly_chart(figure_or_data, *args, **kwargs):
        try:
            fig = figure_or_data if isinstance(figure_or_data, go.Figure) else go.Figure(figure_or_data)
            if not kwargs.get("on_select") and is_time_figure(fig):
                return render(fig)
        except Exception:  # noqa: BLE001  문제가 생기면 기본 차트로
            pass
        return orig(figure_or_data, *args, **kwargs)

    st.plotly_chart = plotly_chart


__all__ = [n for n in dir() if not n.startswith("__")]
