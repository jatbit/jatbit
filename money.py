"""통화별 금액 표시 (리서치 탭: 미국 종목은 달러, 국내 종목은 원).
지금 보고 있는 종목의 통화는 세션마다 st.session_state['rs_cur']에 저장된다."""
import numpy as np

try:
    import streamlit as st
except Exception:  # noqa: BLE001
    st = None


def cur() -> str:
    try:
        return st.session_state.get("rs_cur", "USD") if st is not None else "USD"
    except Exception:  # noqa: BLE001
        return "USD"


def set_cur(c: str) -> None:
    try:
        st.session_state["rs_cur"] = c or "USD"
    except Exception:  # noqa: BLE001
        pass


def is_krw(c: str | None = None) -> bool:
    return (c or cur()) == "KRW"


def big(x, c: str | None = None) -> str:
    """큰 금액: 달러는 $1.23조/십억/백만, 원은 1.2조원/3,456억원."""
    try:
        x = float(x)
    except Exception:  # noqa: BLE001
        return "-"
    if not np.isfinite(x):
        return "-"
    c = c or cur()
    a, sg = abs(x), "-" if x < 0 else ""
    if c == "KRW":
        if a >= 1e12:
            return f"{sg}{a / 1e12:,.2f}조원"
        if a >= 1e8:
            return f"{sg}{a / 1e8:,.0f}억원"
        if a >= 1e4:
            return f"{sg}{a / 1e4:,.0f}만원"
        return f"{sg}{a:,.0f}원"
    sym = "$" if c == "USD" else ""
    tail = "" if c in ("USD", "") else f" {c}"
    for d, u in ((1e12, "조"), (1e9, "십억"), (1e6, "백만")):
        if a >= d:
            return f"{sg}{sym}{a / d:,.2f}{u}{tail}"
    return f"{sg}{sym}{a:,.0f}{tail}"


def price(x, c: str | None = None) -> str:
    try:
        x = float(x)
    except Exception:  # noqa: BLE001
        return "-"
    if not np.isfinite(x):
        return "-"
    c = c or cur()
    if c == "KRW":
        return f"{x:,.0f}원"
    return f"${x:,.2f}" if c == "USD" else f"{x:,.2f} {c}"


def unit(c: str | None = None) -> tuple:
    """입력칸 단위 (배수, 이름): 달러는 백만$, 원은 억원."""
    return (1e8, "억원") if (c or cur()) == "KRW" else (1e6, "백만$")


def hv(c: str | None = None, fmt: str = ",.0f") -> str:
    """plotly hovertemplate용 금액 형식."""
    return f"%{{y:{fmt}}}원" if (c or cur()) == "KRW" else f"%{{y:${fmt}}}"


def sym(c: str | None = None) -> str:
    return "원" if (c or cur()) == "KRW" else "$"
