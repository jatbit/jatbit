"""GitHub에 새 코드를 올리면 새로고침만으로 반영되게 하는 장치.

Streamlit Community Cloud는 GitHub에 커밋이 올라오면 파일은 바로 갱신하지만, 이미 불러온 모듈(options.py 등)은
메모리에 남아 있어 앱을 재부팅해야 바뀌었다. app.py가 실행될 때마다 이 모듈의 refresh()가 각 파일의 수정 시각·크기를
확인하고, 하나라도 바뀌었으면 의존 순서대로 모듈을 다시 불러온다.
"""
import importlib
import importlib.util
import os
import sys

# 다른 모듈을 가져다 쓰는 모듈일수록 뒤에 둔다 (먼저 다시 불러온 모듈을 뒤의 모듈이 가져가도록)
ORDER = ["bsm", "glossary", "ui", "money", "rates", "options", "volmodels", "capm_opt", "bonds", "news", "valuation", "kr", "dcf", "insider",
         "credit", "optvol", "american", "risklab", "backtest", "peers", "earnings", "research", "market"]
_BASE = os.path.dirname(os.path.abspath(__file__))
_SELF = os.path.abspath(__file__)
_seen: dict = {}


def _sig(path: str):
    s = os.stat(path)
    return s.st_mtime_ns, s.st_size


def _self_sig():
    try:
        return _sig(_SELF)
    except OSError:
        return None


_MY_SIG = _self_sig()


def refresh(force: bool = False) -> list:
    """바뀐 파일 이름 목록을 돌려준다. 바뀐 게 있으면 등록된 모듈을 순서대로 모두 다시 불러온다.
    이 파일(autoreload.py) 자체가 바뀌면 먼저 자신을 다시 불러온 뒤 모든 모듈을 새로 불러온다."""
    if not force and _self_sig() != _MY_SIG:
        try:
            os.remove(importlib.util.cache_from_source(_SELF))
        except OSError:
            pass
        try:
            mod = importlib.reload(sys.modules[__name__])
            return mod.refresh(force=True)
        except Exception:  # noqa: BLE001
            pass
    changed = ["*"] if force else []
    for name in ORDER:
        path = os.path.join(_BASE, name + ".py")
        if not os.path.exists(path):
            continue
        sig = _sig(path)
        if name not in _seen:  # 처음 보는 파일: 이번 실행에서 최신 내용으로 불러오므로 기록만
            _seen[name] = sig
            continue
        if sig != _seen[name]:
            _seen[name] = sig
            if name in sys.modules:
                changed.append(name)
    if changed:
        importlib.invalidate_caches()
        for name in ORDER:
            mod = sys.modules.get(name)
            if mod is None:
                continue
            p = os.path.join(_BASE, name + ".py")
            try:  # 같은 초에 같은 크기로 바뀐 파일도 새로 읽도록 바이트코드 캐시를 지움
                os.remove(importlib.util.cache_from_source(p))
            except OSError:
                pass
            try:
                importlib.reload(mod)
                if os.path.exists(p):
                    _seen[name] = _sig(p)
            except Exception:  # noqa: BLE001  (새 코드에 오류가 있으면 이전 모듈을 그대로 쓴다)
                pass
    return changed
