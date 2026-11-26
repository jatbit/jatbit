"""나라 지도 데이터: 종목 본사 위치, 구역(미국 주 · 한국 시도) 경계, 지도에 칠할 값(수익률·거래대금·PER·점수) 계산.

화면은 geomap.py. 여기는 화면 코드 없이 데이터만 다룬다.
- 미국: 본사 = 위키백과 'List of S&P 500 companies' 표의 Headquarters Location,
  좌표 = 그 도시 문서의 위키백과 좌표 (한 번에 50곳씩 조회, 7일 캐시). 좌표가 없으면 주 중심.
- 한국: 본사 시도 = KRX 상장회사 정보(지역). OpenDART 키가 있으면 기업개황 주소로 시·군·구까지.
- 시도 경계: southkorea-maps (KOSTAT) GeoJSON/TopoJSON.
"""

from __future__ import annotations

import html as _html
import json
import math
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st


UA = {"User-Agent": "jatbit-stock-app/1.0 (streamlit; educational)"}
WIKI_SP = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_API = "https://en.wikipedia.org/w/api.php"
KR_GEO_URLS = [
    "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_provinces_geo_simple.json",
    "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2018/json/skorea-provinces-2018-geo.json",
    "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2018/json/skorea-provinces-2018-topo-simple.json",
    "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_provinces_topo_simple.json",
]

# 미국 주: 영문 이름 → (약어, 한글, 위도, 경도)
US_STATES = {
    "Alabama": ("AL", "앨라배마", 32.8, -86.8), "Alaska": ("AK", "알래스카", 61.4, -150.0), "Arizona": ("AZ", "애리조나", 34.3, -111.7),
    "Arkansas": ("AR", "아칸소", 34.9, -92.4), "California": ("CA", "캘리포니아", 37.2, -119.5), "Colorado": ("CO", "콜로라도", 39.0, -105.5),
    "Connecticut": ("CT", "코네티컷", 41.6, -72.7), "Delaware": ("DE", "델라웨어", 39.0, -75.5),
    "District of Columbia": ("DC", "워싱턴 D.C.", 38.9, -77.03), "Florida": ("FL", "플로리다", 28.6, -82.4),
    "Georgia": ("GA", "조지아", 32.7, -83.4), "Hawaii": ("HI", "하와이", 21.3, -157.8), "Idaho": ("ID", "아이다호", 44.4, -114.6),
    "Illinois": ("IL", "일리노이", 40.0, -89.2), "Indiana": ("IN", "인디애나", 39.9, -86.3), "Iowa": ("IA", "아이오와", 42.1, -93.5),
    "Kansas": ("KS", "캔자스", 38.5, -98.4), "Kentucky": ("KY", "켄터키", 37.5, -85.3), "Louisiana": ("LA", "루이지애나", 31.1, -92.0),
    "Maine": ("ME", "메인", 45.4, -69.2), "Maryland": ("MD", "메릴랜드", 39.0, -76.8), "Massachusetts": ("MA", "매사추세츠", 42.3, -71.8),
    "Michigan": ("MI", "미시간", 43.3, -84.6), "Minnesota": ("MN", "미네소타", 46.3, -94.3), "Mississippi": ("MS", "미시시피", 32.7, -89.7),
    "Missouri": ("MO", "미주리", 38.4, -92.5), "Montana": ("MT", "몬태나", 47.0, -109.6), "Nebraska": ("NE", "네브래스카", 41.5, -99.8),
    "Nevada": ("NV", "네바다", 39.3, -116.6), "New Hampshire": ("NH", "뉴햄프셔", 43.7, -71.6), "New Jersey": ("NJ", "뉴저지", 40.2, -74.6),
    "New Mexico": ("NM", "뉴멕시코", 34.4, -106.1), "New York": ("NY", "뉴욕", 42.9, -75.5),
    "North Carolina": ("NC", "노스캐롤라이나", 35.6, -79.4), "North Dakota": ("ND", "노스다코타", 47.5, -100.5),
    "Ohio": ("OH", "오하이오", 40.3, -82.8), "Oklahoma": ("OK", "오클라호마", 35.6, -97.5), "Oregon": ("OR", "오리건", 43.9, -120.6),
    "Pennsylvania": ("PA", "펜실베이니아", 40.9, -77.8), "Rhode Island": ("RI", "로드아일랜드", 41.7, -71.5),
    "South Carolina": ("SC", "사우스캐롤라이나", 33.9, -80.9), "South Dakota": ("SD", "사우스다코타", 44.4, -100.2),
    "Tennessee": ("TN", "테네시", 35.9, -86.4), "Texas": ("TX", "텍사스", 31.5, -99.3), "Utah": ("UT", "유타", 39.3, -111.7),
    "Vermont": ("VT", "버몬트", 44.1, -72.7), "Virginia": ("VA", "버지니아", 37.5, -78.9), "Washington": ("WA", "워싱턴주", 47.4, -120.5),
    "West Virginia": ("WV", "웨스트버지니아", 38.6, -80.6), "Wisconsin": ("WI", "위스콘신", 44.6, -89.9), "Wyoming": ("WY", "와이오밍", 43.0, -107.6),
}
US_ABBR = {v[0]: k for k, v in US_STATES.items()}
COUNTRY_KO = {"Ireland": "아일랜드", "United Kingdom": "영국", "England": "영국", "Switzerland": "스위스", "Netherlands": "네덜란드",
              "Bermuda": "버뮤다", "Canada": "캐나다", "Jersey": "저지섬", "Israel": "이스라엘", "Singapore": "싱가포르",
              "Germany": "독일", "France": "프랑스", "Luxembourg": "룩셈부르크", "Cayman Islands": "케이맨 제도", "Bahamas": "바하마",
              "Puerto Rico": "푸에르토리코", "Taiwan": "대만", "Japan": "일본", "Australia": "호주", "Mexico": "멕시코"}

# 한국 시도: 짧은 이름 → (정식 이름, 위도, 경도, 흩어 찍을 반지름(도))
SIDO = {
    "서울": ("서울특별시", 37.5665, 126.978, 0.06), "부산": ("부산광역시", 35.18, 129.075, 0.06), "대구": ("대구광역시", 35.871, 128.601, 0.05),
    "인천": ("인천광역시", 37.456, 126.705, 0.05), "광주": ("광주광역시", 35.160, 126.851, 0.04), "대전": ("대전광역시", 36.350, 127.385, 0.04),
    "울산": ("울산광역시", 35.538, 129.311, 0.05), "세종": ("세종특별자치시", 36.480, 127.289, 0.03), "경기": ("경기도", 37.40, 127.20, 0.22),
    "강원": ("강원특별자치도", 37.75, 128.25, 0.3), "충북": ("충청북도", 36.75, 127.70, 0.2), "충남": ("충청남도", 36.55, 126.85, 0.2),
    "전북": ("전북특별자치도", 35.72, 127.15, 0.2), "전남": ("전라남도", 34.88, 126.95, 0.25), "경북": ("경상북도", 36.35, 128.75, 0.3),
    "경남": ("경상남도", 35.35, 128.25, 0.22), "제주": ("제주특별자치도", 33.38, 126.55, 0.1),
}
_SIDO_ALIAS = [("충청북", "충북"), ("충청남", "충남"), ("전라북", "전북"), ("전라남", "전남"), ("경상북", "경북"), ("경상남", "경남"),
               ("Chungcheongbuk", "충북"), ("Chungcheongnam", "충남"), ("Jeollabuk", "전북"), ("Jeonbuk", "전북"), ("Jeollanam", "전남"),
               ("Gyeongsangbuk", "경북"), ("Gyeongsangnam", "경남"), ("Seoul", "서울"), ("Busan", "부산"), ("Daegu", "대구"),
               ("Incheon", "인천"), ("Gwangju", "광주"), ("Daejeon", "대전"), ("Ulsan", "울산"), ("Sejong", "세종"),
               ("Gyeonggi", "경기"), ("Gangwon", "강원"), ("Jeju", "제주")]

# 본사가 많은 시·군·구 좌표 (대략적인 중심). 'DART 주소 → 시군구'를 여기서 찾고, 없으면 시도 안에 흩어 찍는다.
SIGUNGU = {
    "서울 강남구": (37.517, 127.047), "서울 서초구": (37.484, 127.033), "서울 송파구": (37.514, 127.106), "서울 영등포구": (37.526, 126.896),
    "서울 중구": (37.564, 126.998), "서울 종로구": (37.573, 126.979), "서울 마포구": (37.566, 126.902), "서울 용산구": (37.532, 126.990),
    "서울 구로구": (37.495, 126.888), "서울 금천구": (37.457, 126.896), "서울 성동구": (37.563, 127.037), "서울 강서구": (37.551, 126.850),
    "서울 동대문구": (37.574, 127.040), "서울 광진구": (37.538, 127.082), "서울 강동구": (37.530, 127.124), "서울 서대문구": (37.579, 126.937),
    "서울 성북구": (37.589, 127.017), "서울 노원구": (37.654, 127.056), "서울 관악구": (37.478, 126.951), "서울 동작구": (37.512, 126.940),
    "서울 양천구": (37.517, 126.866), "서울 은평구": (37.603, 126.929), "서울 도봉구": (37.669, 127.047), "서울 강북구": (37.640, 127.025),
    "서울 중랑구": (37.606, 127.093),
    "경기 성남시": (37.420, 127.127), "경기 성남시 분당구": (37.383, 127.119), "경기 성남시 수정구": (37.450, 127.146),
    "경기 성남시 중원구": (37.430, 127.137), "경기 수원시": (37.263, 127.029), "경기 수원시 영통구": (37.259, 127.047),
    "경기 용인시": (37.241, 127.178), "경기 용인시 기흥구": (37.280, 127.115), "경기 용인시 수지구": (37.322, 127.098),
    "경기 화성시": (37.200, 126.831), "경기 이천시": (37.272, 127.435), "경기 평택시": (36.992, 127.113), "경기 안산시": (37.322, 126.831),
    "경기 안양시": (37.394, 126.957), "경기 고양시": (37.658, 126.832), "경기 부천시": (37.503, 126.766), "경기 시흥시": (37.380, 126.803),
    "경기 파주시": (37.760, 126.780), "경기 광명시": (37.478, 126.865), "경기 김포시": (37.615, 126.716), "경기 의왕시": (37.345, 126.968),
    "경기 군포시": (37.362, 126.935), "경기 과천시": (37.429, 126.988), "경기 하남시": (37.539, 127.215), "경기 오산시": (37.150, 127.077),
    "경기 안성시": (37.008, 127.280), "경기 광주시": (37.429, 127.255), "경기 남양주시": (37.636, 127.216), "경기 의정부시": (37.738, 127.034),
    "경기 구리시": (37.594, 127.130), "경기 포천시": (37.895, 127.200), "경기 양주시": (37.785, 127.046), "경기 여주시": (37.298, 127.637),
    "인천 연수구": (37.410, 126.678), "인천 남동구": (37.447, 126.731), "인천 서구": (37.545, 126.676), "인천 중구": (37.474, 126.621),
    "인천 부평구": (37.507, 126.722), "인천 미추홀구": (37.464, 126.650),
    "부산 해운대구": (35.163, 129.164), "부산 강서구": (35.212, 128.981), "부산 남구": (35.137, 129.084), "부산 사상구": (35.153, 128.991),
    "부산 부산진구": (35.163, 129.053), "부산 사하구": (35.104, 128.975), "부산 기장군": (35.244, 129.222), "부산 중구": (35.106, 129.032),
    "대전 유성구": (36.362, 127.356), "대전 대덕구": (36.347, 127.416), "대전 서구": (36.355, 127.384),
    "울산 남구": (35.544, 129.330), "울산 북구": (35.583, 129.361), "울산 울주군": (35.522, 129.243), "울산 동구": (35.505, 129.417),
    "대구 달서구": (35.830, 128.533), "대구 북구": (35.886, 128.583), "대구 수성구": (35.858, 128.631), "대구 달성군": (35.775, 128.431),
    "광주 광산구": (35.140, 126.794), "광주 북구": (35.174, 126.912), "광주 서구": (35.152, 126.890),
    "충북 청주시": (36.642, 127.489), "충북 음성군": (36.940, 127.690), "충북 진천군": (36.855, 127.436), "충북 충주시": (36.991, 127.926),
    "충남 천안시": (36.815, 127.114), "충남 아산시": (36.790, 127.002), "충남 당진시": (36.890, 126.628), "충남 서산시": (36.785, 126.450),
    "충남 공주시": (36.446, 127.119), "충남 논산시": (36.187, 127.099),
    "경북 구미시": (36.120, 128.344), "경북 포항시": (36.019, 129.343), "경북 경산시": (35.825, 128.741), "경북 경주시": (35.856, 129.225),
    "경북 김천시": (36.140, 128.114), "경북 영천시": (35.973, 128.939), "경북 칠곡군": (35.995, 128.402),
    "경남 창원시": (35.228, 128.681), "경남 김해시": (35.229, 128.889), "경남 거제시": (34.880, 128.621), "경남 양산시": (35.335, 129.037),
    "경남 진주시": (35.180, 128.108), "경남 사천시": (35.004, 128.064), "경남 함안군": (35.272, 128.406), "경남 밀양시": (35.504, 128.746),
    "전북 전주시": (35.824, 127.148), "전북 익산시": (35.948, 126.958), "전북 군산시": (35.968, 126.737), "전북 완주군": (35.904, 127.162),
    "전북 정읍시": (35.570, 126.856),
    "전남 여수시": (34.760, 127.662), "전남 광양시": (34.940, 127.696), "전남 나주시": (35.016, 126.711), "전남 순천시": (34.951, 127.487),
    "전남 목포시": (34.812, 126.392), "전남 영암군": (34.800, 126.697), "전남 화순군": (35.064, 126.987),
    "강원 춘천시": (37.881, 127.730), "강원 원주시": (37.342, 127.920), "강원 강릉시": (37.752, 128.876), "강원 홍천군": (37.697, 127.889),
    "세종": (36.480, 127.289), "제주 제주시": (33.499, 126.531), "제주 서귀포시": (33.254, 126.560),
}


def sido_key(s) -> str | None:
    """'서울특별시' · '경상북도' · 'Seoul' → 짧은 시도 이름. 모르면 None."""
    t = str(s or "").strip()
    if not t or t.lower() == "nan":
        return None
    for a, k in _SIDO_ALIAS:
        if t.startswith(a):
            return k
    k = t[:2]
    return k if k in SIDO else None


def _get(url: str, timeout: int = 25) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def _strip(s: str) -> str:
    s = re.sub(r"<sup.*?</sup>", "", s, flags=re.S)
    return _html.unescape(re.sub(r"<[^>]+>", "", s)).replace("\xa0", " ").strip()


def _spread(lat: float, lon: float, k: int, step: float) -> tuple:
    """같은 곳에 여러 점이 겹치지 않게 해바라기 모양으로 펼친다 (k번째 점)."""
    if k == 0:
        return lat, lon
    r, a = step * math.sqrt(k), k * 2.39996
    return lat + r * math.sin(a), lon + r * math.cos(a) / max(math.cos(math.radians(lat)), 0.3)


# ----------------------------------------------------------------------
# 미국: 본사 위치
# ----------------------------------------------------------------------
def parse_sp500_hq(page: str) -> pd.DataFrame:
    """위키백과 S&P 500 표 HTML → Symbol, HQ(글자), Link(도시 문서 제목)."""
    i = page.find('id="constituents"')
    if i < 0:
        raise RuntimeError("S&P 500 표를 찾지 못했습니다")
    tbl = page[i: page.find("</table>", i)]
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, flags=re.S)
    hdr = [_strip(c) for c in re.findall(r"<th[^>]*>(.*?)</th>", rows[0], flags=re.S)]
    isym = next(j for j, h in enumerate(hdr) if "Symbol" in h)
    ihq = next(j for j, h in enumerate(hdr) if "Headquarters" in h)
    out = []
    for r in rows[1:]:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S)
        if len(cells) <= max(isym, ihq):
            continue
        hq = cells[ihq]
        links = [_html.unescape(t) for t in re.findall(r'<a [^>]*title="([^"]+)"', hq)]
        out.append({"Symbol": _strip(cells[isym]), "HQ": _strip(hq), "Link": links[0] if links else ""})
    if not out:
        raise RuntimeError("S&P 500 표가 비어 있습니다")
    return pd.DataFrame(out)


def split_hq(hq: str) -> tuple:
    """'Saint Paul, Minnesota' → (도시, 주 영문 이름, 나라). 미국이 아니면 주 = None."""
    parts = [p.strip() for p in str(hq or "").split(",") if p.strip()]
    if not parts:
        return "", None, None
    last = parts[-1]
    city = ", ".join(parts[:-1]) or last
    if last in ("D.C.", "DC", "District of Columbia") or str(hq).startswith("Washington, D.C"):
        return "Washington", "District of Columbia", "United States"
    if last in US_STATES:
        return city, last, "United States"
    if last.upper() in US_ABBR:
        return city, US_ABBR[last.upper()], "United States"
    if last in ("U.S.", "US", "USA", "United States") and len(parts) >= 3 and parts[-2] in US_STATES:
        return ", ".join(parts[:-2]), parts[-2], "United States"
    return city, None, last


def wiki_coords(titles: list) -> dict:
    """위키백과 문서 제목 → (위도, 경도). 50개씩 묶어서 묻는다."""
    out = {}
    titles = [t for t in dict.fromkeys(titles) if t]
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        q = urllib.parse.urlencode({"action": "query", "prop": "coordinates", "format": "json", "redirects": 1,
                                    "colimit": "max", "titles": "|".join(chunk)})
        try:
            j = json.loads(_get(f"{WIKI_API}?{q}").decode("utf-8"))
        except Exception:  # noqa: BLE001
            continue
        qd = j.get("query") or {}
        fwd = {}
        for m in (qd.get("normalized") or []) + (qd.get("redirects") or []):
            fwd[m.get("from")] = m.get("to")
        got = {}
        for p in (qd.get("pages") or {}).values():
            c = p.get("coordinates") or []
            if c:
                got[p.get("title")] = (float(c[0]["lat"]), float(c[0]["lon"]))
        for t in chunk:
            u, seen = t, 0
            while u in fwd and seen < 4:
                u, seen = fwd[u], seen + 1
            if u in got:
                out[t] = got[u]
    return out


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def us_hq() -> pd.DataFrame:
    """S&P 500 본사: key, Symbol, City, State(영문), StateKo, Abbr, Country, lat, lon, Exact(도시 좌표 여부)."""
    import sectors

    d = parse_sp500_hq(_get(WIKI_SP).decode("utf-8", "ignore"))
    sp = d.apply(lambda r: split_hq(r["HQ"]), axis=1, result_type="expand")
    d["City"], d["State"], d["Country"] = sp[0], sp[1].where(sp[1].notna(), None), sp[2]
    coords = wiki_coords(list(d["Link"]))
    lat, lon, exact = [], [], []
    for r in d.itertuples():
        c = coords.get(r.Link)
        us = isinstance(r.State, str) and r.State in US_STATES
        if c and us:  # 미국 안의 좌표만 믿는다 (링크가 주 문서일 때도 주 안이라 괜찮음)
            lat.append(c[0]), lon.append(c[1]), exact.append(True)
        elif us:
            s = US_STATES[r.State]
            lat.append(s[2]), lon.append(s[3]), exact.append(False)
        else:
            lat.append(np.nan), lon.append(np.nan), exact.append(False)
    d["lat"], d["lon"], d["Exact"] = lat, lon, exact
    d["Abbr"] = d["State"].map(lambda s: US_STATES[s][0] if s in US_STATES else None)
    d["StateKo"] = d["State"].map(lambda s: US_STATES[s][1] if s in US_STATES else None)
    d["key"] = d["Symbol"].map(sectors._norm)
    return d


# ----------------------------------------------------------------------
# 한국: 본사 시도 · 주소
# ----------------------------------------------------------------------
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def kr_regions() -> pd.DataFrame:
    """종목코드 → 본사 시도 (KRX 상장회사 정보의 '지역')."""
    from data_load import krx_desc_raw

    desc = krx_desc_raw().copy()
    if "Symbol" in desc.columns and "Code" not in desc.columns:
        desc = desc.rename(columns={"Symbol": "Code"})
    if "Region" not in desc.columns:
        raise RuntimeError("KRX 상장회사 정보에 지역이 없습니다")
    out = pd.DataFrame({"Code": desc["Code"].astype(str).str.zfill(6), "RegionRaw": desc["Region"].astype(str)})
    out["Sido"] = out["RegionRaw"].map(sido_key)
    if out["Sido"].notna().sum() == 0:  # 지역 열이 비어 있으면 결과를 기억하지 않는다 (다음에 다시 받음)
        raise RuntimeError("KRX 지역 값 없음 · 열 " + ",".join(map(str, desc.columns))[:120] + " · 예시 " + "|".join(out["RegionRaw"].head(3)))
    return out.drop_duplicates("Code")


@st.cache_resource(show_spinner=False)
def _addr_store() -> dict:
    return {}


def _dart_addr(corp: str, key: str) -> str:
    q = urllib.parse.urlencode({"crtfc_key": key, "corp_code": corp})
    j = json.loads(_get(f"https://opendart.fss.or.kr/api/company.json?{q}", timeout=15).decode("utf-8"))
    if j.get("status") in ("010", "011", "012", "020", "800", "900", "901"):  # 키 오류·한도 초과 등은 '주소 없음'으로 기억하지 않는다 (다음에 다시 받음)
        raise RuntimeError("DART " + str(j.get("status")))
    return str(j.get("adres") or "") if j.get("status") == "000" else ""


def kr_addresses(codes: list, key: str) -> dict:
    """종목코드 → DART 기업개황 주소 (한 번 받은 주소는 앱이 켜져 있는 동안 기억)."""
    import kr

    store = _addr_store()
    need = [c for c in codes if c not in store]
    if need:
        cmap = kr.dart_corp_map(key)
        jobs = [(c, cmap.get(c)) for c in need]

        def one(job):
            c, corp = job
            if not corp:
                return c, ""
            try:
                return c, _dart_addr(corp, key)
            except Exception:  # noqa: BLE001
                return c, None

        with ThreadPoolExecutor(max_workers=8) as ex:
            for c, a in ex.map(one, jobs):
                if a is not None:
                    store[c] = a
    return {c: store.get(c, "") for c in codes}


def addr_point(addr: str) -> tuple:
    """'경기도 성남시 분당구 …' → (시도, 시군구 이름, 위도, 경도). 시군구 좌표를 모르면 위도·경도 = None."""
    toks = str(addr or "").replace(",", " ").split()
    if not toks:
        return None, "", None, None
    sd = sido_key(toks[0])
    rest = toks[1:]
    if sd is None and toks[0].endswith(("시", "구", "군")):
        rest = toks
    if sd == "세종":
        return sd, "세종", *SIGUNGU["세종"]
    if not rest:
        return sd, "", None, None
    g1 = rest[0]
    g2 = rest[1] if len(rest) > 1 and g1.endswith("시") and rest[1].endswith("구") else ""
    for k in ([f"{sd} {g1} {g2}"] if g2 else []) + [f"{sd} {g1}"]:
        if k in SIGUNGU:
            return sd, (f"{g1} {g2}".strip()), *SIGUNGU[k]
    return sd, (f"{g1} {g2}".strip() if g1.endswith(("시", "구", "군")) else ""), None, None


# ----------------------------------------------------------------------
# 한국 시도 경계 (GeoJSON)
# ----------------------------------------------------------------------
def topo_to_geo(t: dict) -> dict:
    """TopoJSON → GeoJSON (폴리곤만)."""
    tr = t.get("transform")
    sc, tl = (tr["scale"], tr["translate"]) if tr else ((1, 1), (0, 0))
    arcs = []
    for a in t.get("arcs", []):
        x = y = 0
        pts = []
        for p in a:
            if tr:
                x, y = x + p[0], y + p[1]
                pts.append([x * sc[0] + tl[0], y * sc[1] + tl[1]])
            else:
                pts.append([p[0], p[1]])
        arcs.append(pts)

    def ring(ids):
        out = []
        for k, i in enumerate(ids):
            a = arcs[i] if i >= 0 else arcs[~i][::-1]
            out.extend(a if k == 0 else a[1:])
        return out

    feats = []
    for obj in (t.get("objects") or {}).values():
        for g in obj.get("geometries", []):
            if g.get("type") == "Polygon":
                coords = [ring(r) for r in g["arcs"]]
            elif g.get("type") == "MultiPolygon":
                coords = [[ring(r) for r in p] for p in g["arcs"]]
            else:
                continue
            feats.append({"type": "Feature", "properties": g.get("properties") or {}, "geometry": {"type": g["type"], "coordinates": coords}})
    return {"type": "FeatureCollection", "features": feats}


def _area(r) -> float:
    return 0.5 * sum(r[i][0] * r[i + 1][1] - r[i + 1][0] * r[i][1] for i in range(len(r) - 1))


def rewind(gj: dict) -> dict:
    """지도 그리기(d3) 규칙에 맞춰 바깥 고리는 시계 방향, 구멍은 반시계 방향으로 맞춘다 (반대면 지구 전체가 칠해짐)."""
    def fix(poly):
        return [r if (_area(r) < 0) == (k == 0) else r[::-1] for k, r in enumerate(poly)]

    for f in gj.get("features", []):
        g = f.get("geometry") or {}
        if g.get("type") == "Polygon":
            g["coordinates"] = fix(g["coordinates"])
        elif g.get("type") == "MultiPolygon":
            g["coordinates"] = [fix(p) for p in g["coordinates"]]
    return gj


def label_sido(gj: dict) -> dict:
    """각 시도 경계에 properties._k = 짧은 시도 이름을 붙인다 (모르는 경계는 버림)."""
    feats = []
    for f in gj.get("features", []):
        p = f.get("properties") or {}
        k = None
        for v in list(p.values()):
            k = sido_key(v)
            if k:
                break
        if k:
            f["properties"] = dict(p, _k=k)
            feats.append(f)
    return {"type": "FeatureCollection", "features": feats}


@st.cache_data(ttl=30 * 24 * 60 * 60, show_spinner=False)
def kr_geojson() -> dict | None:
    for u in KR_GEO_URLS:
        try:
            j = json.loads(_get(u, timeout=20).decode("utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if j.get("type") == "Topology":
            j = topo_to_geo(j)
        gj = label_sido(j)
        if len({f["properties"]["_k"] for f in gj["features"]}) >= 15:
            return rewind(gj)
    return None


# ----------------------------------------------------------------------
# 지도에 올릴 종목 표 (미국 · 한국 공통 열)
#   Key, Label(점 이름), Name, Region(주 영문 이름 / 시도), Place(본사 글자), lat, lon(본사), plat, plon(겹침을 편 점 위치),
#   Exact(도시·시군구 좌표인지), Marcap, Close, ChangesRatio, Amount, Sector, Industry, ysym(Yahoo 심볼),
#   Query(차트 페이지 입력), RsKey(기업 리서치 입력), SectorEn·IndustryEn(미국 점수 계산용)
# ----------------------------------------------------------------------
def _spread_all(d: pd.DataFrame, radius_fn) -> pd.DataFrame:
    """같은 좌표에 모인 점들을 해바라기 모양으로 펼친다. radius_fn(행) = 그 무리가 넘지 말아야 할 반지름(도).
    점이 많을수록 촘촘하게 (반지름 ≈ 간격 × √개수 를 넘지 않게), 큰 종목일수록 가운데."""
    d = d.sort_values("Marcap", ascending=False).reset_index(drop=True)
    g = d.groupby([d["lat"].round(3), d["lon"].round(3)])
    d["_k"], d["_n"] = g.cumcount(), g["Marcap"].transform("size")
    xy = []
    for a, b, k, n, r in zip(d["lat"], d["lon"], d["_k"], d["_n"], d.to_dict("records")):
        if a != a:
            xy.append((np.nan, np.nan))
            continue
        R, step = radius_fn(r)
        xy.append(_spread(a, b, int(k), min(step, R / math.sqrt(max(int(n), 1)))))
    d["plat"], d["plon"] = [x[0] for x in xy], [x[1] for x in xy]
    return d.drop(columns=["_k", "_n"])


@st.cache_data(ttl=300, show_spinner=False)
def us_frame() -> tuple:
    """(지도 안 종목 표, 해외 본사 종목 표, 도시 좌표를 찾은 수)."""
    import sectors

    sp, _allq, _src = sectors.us_snapshot()
    hq = us_hq()
    d = sp.merge(hq[["key", "City", "State", "Country", "lat", "lon", "Exact", "HQ"]], on="key", how="left")
    d = d.dropna(subset=["Marcap"]).copy()
    d["ysym"] = d["Symbol"].astype(str).str.replace(".", "-", regex=False)
    d["Key"], d["Label"], d["Query"], d["RsKey"] = d["Symbol"].astype(str), d["Symbol"].astype(str), d["Symbol"], d["Symbol"]
    d["SectorEn"] = d["Sector"].astype(str) if "Sector" in d.columns else ""
    d["Sector"] = d["SectorKo"].fillna("분류 없음") if "SectorKo" in d.columns else "분류 없음"
    try:  # 점수 계산용 영문 세부 업종
        import screener

        u = screener.us_universe()
        d["IndustryEn"] = d["ysym"].map(dict(zip(u["Symbol"], u["IndustryEn"])))
    except Exception:  # noqa: BLE001
        d["IndustryEn"] = ""
    d["Region"] = d["State"]
    d["Place"] = [f"{c}, {US_STATES[s][0]}" if s in US_STATES else (f"{c}, {COUNTRY_KO.get(n, n)}" if isinstance(n, str) and n else "알 수 없음")
                  for c, s, n in zip(d["City"].fillna(""), d["State"], d["Country"])]
    d = _spread_all(d, lambda r: (0.6, 0.10) if r.get("Exact") is True else (1.2, 0.35))
    inside = d[d["Region"].isin(list(US_STATES))].copy()
    outside = d[~d["Region"].isin(list(US_STATES)) & d["HQ"].notna()].assign(Region="해외 본사")
    return inside, outside, int(inside["Exact"].fillna(False).astype(bool).sum())


@st.cache_data(ttl=300, show_spinner=False)
def kr_frame(snap: pd.DataFrame, n: int | None, dart_key: str | None) -> tuple:
    """(지도 안 종목 표, 해외·지역 미상 표, 시군구 좌표를 찾은 수). snap: KRX 당일 시세."""
    import sectors

    m = snap.dropna(subset=["Marcap"]).copy()
    m["Code"] = m["Code"].astype(str).str.zfill(6)
    m = m[m["Code"].str.endswith("0")]  # 우선주 제외
    if n:
        m = m.nlargest(int(n), "Marcap")
    m = m.merge(kr_regions()[["Code", "Sido", "RegionRaw"]], on="Code", how="left")
    try:
        smap, _src = sectors.kr_sector_map()
        m = m.merge(smap, on="Code", how="left")
    except Exception:  # noqa: BLE001
        m["Sector"], m["Industry"] = "분류 없음", "분류 없음"
    m["Sector"] = m["Sector"].fillna("분류 없음")
    addr = kr_addresses(list(m["Code"]), dart_key) if dart_key else {}
    sido, place, lat, lon, exact = [], [], [], [], []
    for r in m.itertuples():
        sd, gu, a, b = addr_point(addr.get(r.Code, "")) if addr.get(r.Code) else (None, "", None, None)
        sd = sd or (r.Sido if isinstance(r.Sido, str) else None)
        sido.append(sd)
        exact.append(a is not None)
        if a is not None:
            lat.append(a), lon.append(b), place.append(f"{SIDO[sd][0]} {gu}" if sd in SIDO else gu)
        elif sd in SIDO:
            lat.append(SIDO[sd][1]), lon.append(SIDO[sd][2]), place.append(SIDO[sd][0] + (f" {gu}" if gu else ""))
        else:
            lat.append(np.nan), lon.append(np.nan), place.append(str(r.RegionRaw) if isinstance(r.RegionRaw, str) else "알 수 없음")
    m["Region"], m["Place"], m["lat"], m["lon"], m["Exact"] = sido, place, lat, lon, exact
    kq = m["Market"].astype(str).str.upper().str.startswith("KOSDAQ") if "Market" in m.columns else False
    m["ysym"] = m["Code"] + np.where(kq, ".KQ", ".KS")
    m["Key"], m["Label"], m["Query"], m["RsKey"] = m["Code"], m["Name"], m["Name"], m["Code"]
    m["SectorEn"], m["IndustryEn"] = "", ""
    m = _spread_all(m, lambda r: (0.04, 0.006) if r["Exact"] else ((SIDO[r["Region"]][3], 0.03) if r["Region"] in SIDO else (0.1, 0.03)))
    inside = m[m["Region"].isin(list(SIDO))].copy()
    outside = m[~m["Region"].isin(list(SIDO))].assign(Region="해외 본사")
    return inside, outside, int(inside["Exact"].sum())


def sectors_by_cap(*frames) -> list:
    d = pd.concat([f for f in frames if f is not None and len(f)])
    return list(d.groupby("Sector")["Marcap"].sum().sort_values(ascending=False).index.astype(str))


# ----------------------------------------------------------------------
# 일봉 이력 · 기간 수익률 · 거래대금 증가율
# ----------------------------------------------------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def history(symbols: tuple) -> tuple:
    """(종가, 거래량) 최근 2년 일봉. 열 = Yahoo 심볼."""
    import yfinance as yf

    d = yf.download(list(symbols), period="2y", interval="1d", auto_adjust=True, progress=False, threads=True)
    if d is None or d.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(d.columns, pd.MultiIndex):
        c, v = d["Close"], d["Volume"] if "Volume" in d.columns.get_level_values(0) else pd.DataFrame()
    else:
        c = d[["Close"]].rename(columns={"Close": symbols[0]})
        v = d[["Volume"]].rename(columns={"Volume": symbols[0]}) if "Volume" in d.columns else pd.DataFrame()
    c.index = pd.to_datetime(c.index)
    return c, v


UNIVERSE: dict = {}  # 나라 → 그 나라 지도 전체 종목의 Yahoo 심볼 튜플 (섹터를 골라도 같은 일봉 캐시를 쓰려고)


@st.cache_resource(show_spinner=False)
def _locks() -> dict:
    return {}


def _lock(key):
    import threading

    L = _locks()
    if key not in L:
        L[key] = threading.Lock()
    return L[key]


def set_universe(iso: str, syms) -> tuple:
    u = tuple(sorted(set(str(x) for x in syms)))
    UNIVERSE[iso] = u
    return u


def hist(syms, iso: str | None = None) -> tuple:
    """일봉 (종가, 거래량). 나라 전체 종목을 한 번에 받아 두고(1시간 캐시) 필요한 열만 꺼낸다.
    같은 묶음을 두 곳에서 동시에 받지 않도록 잠근다 (미리 받기와 겹칠 때 기다렸다 캐시를 씀)."""
    syms = [str(x) for x in syms]
    u = UNIVERSE.get(iso) if iso else None
    key = u if u and set(syms) <= set(u) else tuple(syms)
    with _lock(key):
        c, v = history(key)
    cols = [x for x in syms if x in c.columns]
    return c[cols] if len(c) else c, (v[[x for x in cols if x in v.columns]] if len(v) else v)


@st.cache_resource(show_spinner=False)
def _prewarm_started() -> dict:
    return {}


def prewarm(iso: str) -> None:
    """나라 전체 일봉을 뒤에서 미리 받아 둔다 (기간을 바꾸거나 날짜별로 볼 때 기다리지 않게). 1시간에 한 번."""
    import threading
    import time

    u = UNIVERSE.get(iso)
    if not u:
        return
    started = _prewarm_started()
    if time.time() - started.get(u, 0) < 3600:
        return
    started[u] = time.time()

    def run():
        try:
            with _lock(u):
                history(u)
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=run, daemon=True).start()


def period_ret(s: pd.Series, per: str) -> float:
    import market

    return market._period_ret(s.dropna(), per) if s is not None else np.nan


def returns_for(d: pd.DataFrame, pers: list, iso: str | None = None) -> pd.DataFrame:
    """종목별 기간 수익률(%) 표 (열 = 기간). 1일은 시세표의 등락률."""
    out = pd.DataFrame(index=d.index)
    need = [p for p in pers if p != "1일"]
    close = hist(d["ysym"], iso)[0] if need else pd.DataFrame()
    for p in pers:
        if p == "1일":
            out[p] = pd.to_numeric(d["ChangesRatio"], errors="coerce")
        else:
            out[p] = [period_ret(close[s], p) if s in close.columns else np.nan for s in d["ysym"].astype(str)]
    return out


def amount_ratio(d: pd.DataFrame, iso: str | None = None) -> pd.DataFrame:
    """최근 거래일 거래대금(A1)과 직전 20거래일 평균(A0). 거래대금 = 종가 × 거래량."""
    close, vol = hist(d["ysym"], iso)
    a1, a0 = [], []
    for s in d["ysym"].astype(str):
        if s in close.columns and s in vol.columns:
            a = (close[s] * vol[s]).dropna()
            a = a[a > 0]
            if len(a) >= 10:
                a1.append(float(a.iloc[-1])), a0.append(float(a.iloc[-21:-1].mean()))
                continue
        a1.append(np.nan), a0.append(np.nan)
    return pd.DataFrame({"A1": a1, "A0": a0}, index=d.index)


def index_returns(close: pd.DataFrame | None, sym: str, pers: list) -> dict:
    """대표지수의 기간 수익률(%)."""
    if close is None or sym not in getattr(close, "columns", []):
        return {p: np.nan for p in pers}
    return {p: period_ret(close[sym], p) for p in pers}


# ----------------------------------------------------------------------
# 날짜별 누적 수익률 (지도 슬라이더용)
# ----------------------------------------------------------------------
PATH_RANGES = {"1개월": 31, "3개월": 92, "6개월": 183, "연초 이후": "ytd", "1년": 366, "2년": 730}
MAX_FRAMES = 70


def _path_dates(idx: pd.DatetimeIndex, rng: str) -> tuple:
    """(시작 기준일, 장면 날짜들). 장면은 많아야 MAX_FRAMES개, 마지막 거래일은 꼭 넣는다."""
    last = idx[-1]
    v = PATH_RANGES[rng]
    start = pd.Timestamp(last.year, 1, 1) - pd.Timedelta(days=1) if v == "ytd" else last - pd.Timedelta(days=v)
    ds = idx[idx > start]
    if len(ds) > MAX_FRAMES:
        step = int(np.ceil(len(ds) / MAX_FRAMES))
        ds = ds[::-1][::step][::-1]
    if len(ds) == 0 or ds[-1] != last:
        ds = ds.append(pd.DatetimeIndex([last]))
    return start, ds


def cum_path(close: pd.DataFrame, cols: list, rng: str) -> tuple:
    """(장면 날짜들, 표[행 = cols, 열 = 날짜] 시작 기준일 대비 누적 수익률 %)."""
    if close is None or close.empty:
        return pd.DatetimeIndex([]), pd.DataFrame(index=cols)
    c = close.reindex(columns=cols).sort_index().ffill()
    start, ds = _path_dates(c.index, rng)
    before = c[c.index <= start]
    base = before.iloc[-1] if len(before) else c.bfill().iloc[0]
    base = base.where(base.notna(), c.bfill().iloc[0])
    cum = (c.loc[ds] / base - 1) * 100
    return ds, cum.T


def stock_path(d: pd.DataFrame, rng: str, iso: str | None = None) -> tuple:
    """d 종목들의 날짜별 누적 수익률. 반환 (날짜들, 표[행 = d.index, 열 = 날짜])."""
    close = hist(d["ysym"], iso)[0]
    ds, cum = cum_path(close, list(d["ysym"].astype(str)), rng)
    cum.index = d.index
    return ds, cum


def index_path(close: pd.DataFrame | None, sym: str, rng: str, dates) -> pd.Series:
    """대표지수의 같은 범위 누적 수익률(%)을 종목 장면 날짜들에 맞춘다."""
    if close is None or sym not in getattr(close, "columns", []) or len(dates) == 0:
        return pd.Series(np.nan, index=dates)
    c = close[[sym]].dropna()
    c = c.reindex(c.index.union(dates)).sort_index().ffill()
    start, _ds = _path_dates(c.index, rng)
    before = c[c.index <= start][sym]
    base = float(before.iloc[-1]) if len(before) else float(c[sym].dropna().iloc[0])
    return (c[sym].reindex(dates) / base - 1) * 100


# ----------------------------------------------------------------------
# PER (후행) — Yahoo 스크리너로 한 번에, 빠진 종목은 하나씩
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _pe_store() -> dict:
    return {}  # Yahoo 심볼 → (PER 또는 nan, 받은 시각)


def _pe_bulk(region: str) -> dict:
    import yfinance as yf

    out = {}
    try:
        q = yf.EquityQuery("eq", ["region", region])
    except Exception:  # noqa: BLE001
        return out
    for off in (0, 250, 500):
        try:
            r = yf.screen(q, offset=off, size=250, sortField="intradaymarketcap", sortAsc=False)
        except Exception:  # noqa: BLE001
            break
        qs = (r or {}).get("quotes") or []
        for x in qs:
            pe = x.get("trailingPE")
            if x.get("symbol"):
                out[str(x["symbol"])] = float(pe) if isinstance(pe, (int, float)) else np.nan
        if len(qs) < 250:
            break
    return out


def _pe_one(sym: str) -> float:
    import yfinance as yf

    try:
        pe = (yf.Ticker(sym).info or {}).get("trailingPE")
        return float(pe) if isinstance(pe, (int, float)) else np.nan
    except Exception:  # noqa: BLE001
        return np.nan


def pe_missing(symbols: list, max_age: float = 24 * 3600) -> list:
    import time

    st_ = _pe_store()
    now = time.time()
    return [s for s in symbols if s not in st_ or now - st_[s][1] > max_age]


def fetch_pe(symbols: list, domestic: bool, progress=None) -> None:
    """PER을 받아 저장소에 넣는다 (하루 동안 다시 쓰고, 모든 사용자가 같이 씀). progress(비율, 글자)."""
    import time

    st_ = _pe_store()
    now = time.time()
    need = pe_missing(symbols)
    if not need:
        return
    if progress:
        progress(0.05, "Yahoo 스크리너에서 PER을 한 번에 받는 중...")
    bulk = _pe_bulk("kr" if domestic else "us")
    for s in need:
        if s in bulk:
            st_[s] = (bulk[s], now)
    rest = [s for s in need if s not in bulk]
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for s, pe in zip(rest, ex.map(_pe_one, rest)):
            st_[s] = (pe, time.time())
            done += 1
            if progress and (done % 10 == 0 or done == len(rest)):
                progress(0.1 + 0.9 * done / max(len(rest), 1), f"남은 종목 하나씩 받는 중 ({done}/{len(rest)})")


def pe_values(symbols) -> list:
    st_ = _pe_store()
    return [st_[s][0] if s in st_ else np.nan for s in symbols]


# ----------------------------------------------------------------------
# 색칠할 값 (색 기준별)
# ----------------------------------------------------------------------
BASES = ["수익률", "지수 대비", "거래대금 증가율", "PER", "종합 점수"]
PERIOD_BASES = ("수익률", "지수 대비")


def region_values(d: pd.DataFrame, basis: str) -> pd.DataFrame:
    """구역별 값 V와 종목 수·시가총액·큰 종목. d: Region, Marcap, Label, V (+ 거래대금은 A1·A0, PER은 PE).
    - 수익률·지수 대비·점수: 시가총액 가중 평균
    - 거래대금 증가율: 구역 합계 기준 (ΣA1 ÷ ΣA0 − 1)
    - PER: 구역 합산 PER (Σ시가총액 ÷ Σ순이익, 적자 종목 제외)"""
    v = d.dropna(subset=["Region", "Marcap"])
    v = v[v["Marcap"] > 0]
    g = v.groupby("Region").agg(Cap=("Marcap", "sum"), N=("Marcap", "size"))
    if basis == "거래대금 증가율":
        ok = v.dropna(subset=["A1", "A0"])
        w = ok.groupby("Region").agg(A1=("A1", "sum"), A0=("A0", "sum"))
        g["V"] = ((w["A1"] / w["A0"] - 1) * 100).reindex(g.index)
    elif basis == "PER":
        ok = v[(v["PE"] > 0) & v["PE"].notna()]
        w = ok.assign(E=ok["Marcap"] / ok["PE"]).groupby("Region").agg(M=("Marcap", "sum"), E=("E", "sum"))
        g["V"] = (w["M"] / w["E"]).reindex(g.index)
    else:
        ok = v.dropna(subset=["V"])
        w = ok.assign(_w=ok["V"] * ok["Marcap"]).groupby("Region").agg(W=("_w", "sum"), M=("Marcap", "sum"))
        g["V"] = (w["W"] / w["M"]).reindex(g.index)
    g["Top"] = v.sort_values("Marcap", ascending=False).groupby("Region")["Label"].apply(
        lambda s: ", ".join(s.astype(str).head(3))).reindex(g.index)
    return g.sort_values("Cap", ascending=False)


def region_table(d: pd.DataFrame) -> pd.DataFrame:
    """(예전 이름) 구역별 시가총액 가중 수익률: Ret 열을 V로 보고 계산, 결과 열 이름은 Ret."""
    r = region_values(d.assign(V=d["Ret"]), "수익률")
    return r.rename(columns={"V": "Ret"})


def agg_value(d: pd.DataFrame, basis: str) -> float:
    """종목 묶음 전체 하나의 값 (구역 카드용)."""
    r = region_values(d.assign(Region="_all"), basis)
    return float(r["V"].iloc[0]) if len(r) else np.nan


__all__ = [n for n in dir() if not n.startswith("__")]
