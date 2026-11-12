"""기업 리서치 탭: 종목 관련 최신 뉴스 3건 찾기
1) 수집: Google 뉴스 RSS 검색(회사명·티커, 최근 7일) + Yahoo Finance 종목 RSS + yfinance 뉴스
2) 관련도: 제목에 회사명·티커가 들어 있는지 점수화하고, 여러 종목을 묶은 목록형 기사는 감점
3) 중복 제거: 같은 사건을 여러 매체가 쓴 경우 제목 단어가 많이 겹치면 하나만 남김
4) 정렬: 관련 기사 중 발행 시각이 가장 최근인 3건 (관련 기사가 모자라면 기준을 낮춰 채움)
"""
import email.utils
import html
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import streamlit as st

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
KST = ZoneInfo("Asia/Seoul")
SUFFIX = re.compile(r"(,?\s+(inc\.?|incorporated|corp\.?|corporation|co\.?|company|ltd\.?|limited|plc|n\.?v\.?|s\.?a\.?|ag|se|"
                    r"holdings?|group|class [abc]|the))+\s*$", re.I)
GENERIC_FIRST = {"the", "first", "american", "general", "united", "bank", "national", "international", "global", "new", "royal",
                 "spdr", "ishares", "invesco", "vanguard", "select", "direxion", "proshares"}
ALIASES = {"GOOGL": ["Google"], "GOOG": ["Google"], "META": ["Facebook", "Meta"], "BRK-B": ["Berkshire"], "BRK-A": ["Berkshire"],
           "TSM": ["TSMC"], "SPY": ["S&P 500"], "QQQ": ["Nasdaq 100", "Nasdaq-100"], "DIA": ["Dow Jones"], "IWM": ["Russell 2000"]}
LISTICLE = re.compile(r"\b(\d+|top|best|these|several)\b[^.]{0,40}\bstocks?\b|\bstocks? to (buy|watch|sell)\b|\bstock market today\b", re.I)


def keywords(name: str, ticker: str) -> list:
    base = SUFFIX.sub("", (name or "").replace("&amp;", "&")).strip(" ,.")
    kws = [base] if base else []
    first = base.split()[0] if base else ""
    if first and first.lower() not in GENERIC_FIRST and len(first) >= 3 and first != base:
        kws.append(first)
    kws += ALIASES.get(ticker.upper(), [])
    return list(dict.fromkeys(k for k in kws if k))


def _get(url: str, timeout: int = 10) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", s or "")).replace("\xa0", " ").strip()


def _parse_rss(xml: bytes, src_tag: str) -> list:
    out = []
    root = ET.fromstring(xml)
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        src_el = it.find("source")
        source = (src_el.text or "").strip() if src_el is not None and src_el.text else ""
        if src_tag == "google" and source and title.endswith(" - " + source):
            title = title[: -len(" - " + source)].strip()  # Google 뉴스 제목 끝의 " - 매체명" 제거
        pub = it.findtext("pubDate")
        try:
            dt = email.utils.parsedate_to_datetime(pub) if pub else None
            if dt is not None and dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:  # noqa: BLE001
            dt = None
        desc = _strip_tags(it.findtext("description") or "")
        if desc and (desc.startswith(title) or title.startswith(desc[:40])):
            desc = ""  # Google RSS 설명은 보통 제목 반복
        if title and link:
            out.append({"title": html.unescape(title), "url": link, "source": source or ("Yahoo Finance" if src_tag == "yahoo" else ""),
                        "time": dt, "summary": desc[:400], "via": src_tag})
    return out


def _from_yf(items) -> list:
    out = []
    for n in items or []:
        c = n.get("content") if isinstance(n, dict) and isinstance(n.get("content"), dict) else n
        if not isinstance(c, dict) or not c.get("title"):
            continue
        url = None
        for k in ("canonicalUrl", "clickThroughUrl"):
            if isinstance(c.get(k), dict) and c[k].get("url"):
                url = c[k]["url"]
                break
        url = url or c.get("link")
        prov = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else c.get("publisher")
        dt = None
        if c.get("pubDate") or c.get("displayTime"):
            try:
                dt = datetime.fromisoformat(str(c.get("pubDate") or c.get("displayTime")).replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                dt = None
        elif c.get("providerPublishTime"):
            dt = datetime.fromtimestamp(int(c["providerPublishTime"]), tz=timezone.utc)
        if url:
            out.append({"title": c["title"].strip(), "url": url, "source": prov or "Yahoo Finance", "time": dt,
                        "summary": _strip_tags(c.get("summary") or c.get("description") or "")[:400], "via": "yfinance"})
    return out


def relevance(item: dict, kws: list, ticker: str) -> int:
    t, s = item["title"], item.get("summary") or ""
    sc = 0
    tk = re.escape(ticker.upper())
    strong = rf"(\${tk}\b|\({tk}\)|\b(NASDAQ|NYSE|AMEX)\s*:\s*{tk}\b)"
    if re.search(strong, t, re.I) or (len(ticker) >= 2 and re.search(rf"(?<![A-Za-z]){tk}(?![A-Za-z])", t)):
        sc += 3
    if any(re.search(rf"(?<![A-Za-z]){re.escape(k)}(?![A-Za-z])", t, re.I) for k in kws):
        sc += 3
    if any(re.search(rf"(?<![A-Za-z]){re.escape(k)}(?![A-Za-z])", s, re.I) for k in kws) or re.search(strong, s, re.I):
        sc += 1
    if LISTICLE.search(t):
        sc -= 2
    return sc


def _tokens(title: str) -> set:
    w = re.findall(r"[a-z0-9]+", title.lower())
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "as", "is", "are", "at", "by", "with", "its", "after", "from"}
    return {x for x in w if x not in stop}


def dedupe(items: list, thr: float = 0.6) -> list:
    kept = []
    for it in items:
        tok = _tokens(it["title"])
        if any(len(tok & k["_tok"]) / max(1, len(tok | k["_tok"])) >= thr for k in kept):
            continue
        it["_tok"] = tok
        kept.append(it)
    for k in kept:
        k.pop("_tok", None)
    return kept


@st.cache_data(ttl=15 * 60, show_spinner="최신 뉴스를 검색하는 중...")
def latest_news(ticker: str, name: str, yf_items: list, n: int = 3, days: int = 7) -> dict:
    kws = keywords(name, ticker)
    q = " OR ".join([f'"{k}"' for k in kws[:3]] + [ticker]) + f" when:{days}d"
    pool, errs = [], []
    g_url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    y_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(ticker)}&region=US&lang=en-US"
    for url, tag in ((g_url, "google"), (y_url, "yahoo")):
        for _ in range(2):
            try:
                pool += _parse_rss(_get(url), tag)
                break
            except Exception as e:  # noqa: BLE001
                err = f"{tag}: {e}"
        else:
            errs.append(err)
    pool += _from_yf(yf_items)
    now = datetime.now(timezone.utc)
    pool = [p for p in pool if p["time"] is None or (now - p["time"]).days <= max(days, 14)]
    for p in pool:
        p["score"] = relevance(p, kws, ticker)
    oldest = datetime(1970, 1, 1, tzinfo=timezone.utc)
    pool.sort(key=lambda p: p["time"] or oldest, reverse=True)
    pool = dedupe(pool)
    picked = [p for p in pool if p["score"] >= 3]
    if len(picked) < n:  # 제목에 회사명이 없는 기사라도 본문 요약에 언급되면 보충
        picked += [p for p in pool if 1 <= p["score"] < 3][: n - len(picked)]
    if len(picked) < n:
        picked += [p for p in pool if p["score"] < 1 and p["via"] != "google"][: n - len(picked)]
    picked = sorted(picked, key=lambda p: p["time"] or oldest, reverse=True)[:n]
    return {"items": picked, "keywords": kws, "query": q, "pool": len(pool), "errors": errs}


def _ago(dt) -> str:
    if dt is None:
        return ""
    sec = (datetime.now(timezone.utc) - dt).total_seconds()
    if sec < 3600:
        rel = f"{max(1, int(sec // 60))}분 전"
    elif sec < 86400:
        rel = f"{int(sec // 3600)}시간 전"
    else:
        rel = f"{int(sec // 86400)}일 전"
    return f"{rel} · {dt.astimezone(KST):%m-%d %H:%M} KST"


def render_latest_news(ticker: str, name: str, yf_items: list, ko: bool, translate) -> None:
    st.markdown("##### 최신 뉴스 3")
    try:
        res = latest_news(ticker, name, yf_items or [])
    except Exception as e:  # noqa: BLE001
        st.caption(f"뉴스를 검색하지 못했습니다: {e}")
        return
    items = res["items"]
    if not items:
        st.caption("최근 7일 안에 이 종목과 관련된 뉴스를 찾지 못했습니다.")
        return
    cards = []
    for i, it in enumerate(items, 1):
        t_ko = translate(it["title"]) if ko else None
        s_ko = translate(it["summary"]) if ko and it.get("summary") else None
        main = (t_ko or it["title"]).strip()
        link = f'<a href="{html.escape(it["url"], quote=True)}" target="_blank" style="font-weight:600">{html.escape(main)}</a>'
        orig = f'<div style="font-size:0.8rem;opacity:.55">{html.escape(it["title"])}</div>' if t_ko and t_ko.strip() != it["title"] else ""
        summ = s_ko or it.get("summary") or ""
        summ = f'<div style="font-size:0.85rem;opacity:.8;margin-top:2px">{html.escape(summ[:220])}{"…" if len(summ) > 220 else ""}</div>' if summ else ""
        meta = " · ".join(x for x in [it.get("source"), _ago(it.get("time"))] if x)
        cards.append(f'<div style="border:1px solid rgba(128,128,128,.25);border-radius:8px;padding:8px 10px;margin:0 0 8px">'
                     f'<span style="opacity:.6;margin-right:6px">{i}</span>{link}{orig}{summ}'
                     f'<div style="font-size:0.78rem;opacity:.6;margin-top:3px">{html.escape(meta)}</div></div>')
    st.markdown("".join(cards).replace("$", "&#36;"), unsafe_allow_html=True)
    st.caption(f"Google 뉴스·Yahoo Finance에서 최근 7일 기사 {res['pool']}건을 모아 제목에 '{'·'.join(res['keywords'])}' 또는 '{ticker}'가 들어간 기사를 "
               "우선 고르고, 같은 사건을 다룬 중복 기사와 여러 종목을 묶은 목록형 기사는 걸러낸 뒤 가장 최근 3건을 보여줍니다. 15분마다 새로 검색합니다."
               + (f" (일부 출처 실패: {', '.join(res['errors'])})" if res["errors"] else ""))
