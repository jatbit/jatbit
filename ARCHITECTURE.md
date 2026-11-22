# 파일 지도

주가 차트 · 시장 분석 앱 (Streamlit). 고칠 곳을 빨리 찾기 위한 문서입니다. 파일은 모두 저장소 맨 위에 있고, 이름 앞부분으로 파트를 나눕니다.

## 1. 뼈대 · 공통

| 파일 | 하는 일 |
|---|---|
| `app.py` | 시작점. 메뉴(페이지 9개)를 만들고 열린 페이지 하나만 실행 |
| `autoreload.py` | GitHub에 올린 코드를 새로고침만으로 반영. **새 모듈은 `ORDER`에 의존 순서대로 추가** (가져다 쓰는 쪽이 뒤) |
| `ui.py` | 테마, 페이지 이동 후 입력값 유지, 지연 탭·더보기, 시세 헤더, 카드, 상호작용 도우미 (아래 표) |
| `glossary.py` | 지표 이름 옆 ⓘ 설명, 표 머리글 설명 |
| `tvchart.py` | 날짜 축 차트를 트레이딩뷰처럼 (끌기·휠 확대) |
| `money.py` | 통화별 금액 표시 |
| `common.py` | `data_load` · `period` · `stats` 묶음 (페이지가 `from common import *`) |
| `data_load.py` / `period.py` / `stats.py` | 종목 목록·주가 / 기간 입력 / 수익률·CAPM·위험 지표 |
| `rates.py` · `ta.py` · `drawings.py` | 금리 겹쳐 보기 · 기술적 지표 · 차트 그리기 도구 |

### ui.py 공통 도우미 (새 화면을 만들 때 먼저 쓰기)

| 함수 | 용도 |
|---|---|
| `ui.section(제목, 설명)` · `ui.cards(items, cols)` · `ui.mgrid(items, cols)` | 소제목 · 값 카드(색 있는 등락) · 지표 격자(ⓘ 설명) |
| `ui.more(라벨, key)` · `ui.lazy_tabs(라벨들, key)` | 펼쳤을 때만 계산하는 더보기 · 열린 탭만 계산 |
| `ui.seg(라벨, 선택지, default, key)` | 버튼 줄 하나 고르기 (선택이 풀리면 None) |
| `ui.go(페이지, 입력키, 값)` · `ui.stock_buttons(이름, 리서치키, 차트입력, key)` | 다른 페이지로 이동하며 입력칸 채우기 · '기업 리서치 / 차트 보기' 버튼 줄 |
| `ui.chart_points(fig, key)` + `ui.new_click(points, key)` | plotly 그림 클릭 받기 → `(trace 번호, 점 번호)` (같은 선택 반복은 무시) |
| `ui.table_pick(df, key, **kw)` | 표에서 고른 행 번호 |
| `ui.with_ctx(fn)` | 작업 스레드에서 Streamlit 캐시를 쓸 때 함수에 씌우기 |
| `ui.UP` / `ui.DOWN` / `ui.FLAT` | 상승 빨강 · 하락 파랑 · 보합 |

## 2. 파트별 파일

**시장 (첫 화면)**: `market.py` → 세계 지도 → `geomap.py`(나라 지도 화면) + `geodata.py`(본사 위치·경계·값 계산). 업종 분류는 `sectors.py`

**주가 조회 · 비교 · 포트폴리오 · 백테스트**: `page_single.py` · `page_compare.py` · `page_portfolio.py` · `backtest.py` + `risklab.py`

**기업 리서치**: `research.py`(탭 묶기) →
- 데이터: `research_data.py`(Yahoo·SEC 10-K, 재무비율, PER 계열), `kr.py`(국내: KRX·DART·국고채), `valuation.py`, `data_check.py`
- 판단·점수: `outlook_rules.py` → `outlook_data.py` → `outlook_signals.py` → `outlook_score.py` → `outlook.py`(화면).
  뒤 파일이 앞 파일을 `from … import *`로 이어받으므로, 다른 곳에서는 예전처럼 `outlook.이름`으로 쓴다
- 하위 탭: `research_chart.py`(겹쳐 그리기·게이지) · `earnings.py` · `peers.py` · `dcf.py` · `credit.py`(+ 계산 `credit_model.py`) · `insider.py` · `news.py`

**스크리너**: `screener.py` (점수는 `outlook.screen_score`, 저장소 `st.session_state["sc_scores"]`를 나라 지도와 같이 씀)

**옵션**: `options.py`(탭 묶기) →
- 데이터·가격: `options_data.py`(체인·기초자산·금리), `bsm.py`, `american.py`, `garch.py`(변동성 예측), `volmodels.py`
- 화면: `options_chain.py` · `strat.py`(전략·백테스트) · `optvol.py` · `optbook.py`(내 포지션) → `hedge.py` · `hedgesim.py` · `capm_opt.py` · `kropt.py`(국내 옵션)

**채권**: `bonds.py`(탭 묶기) → `curve.py` · `bondbook.py` · `bondmath.py` · `hw.py`(헐-화이트) · `riskall.py`(주식·옵션·채권 전체 위험)

## 3. 세션 키 접두어 (위젯 key)

| 접두어 | 파일 | 접두어 | 파일 |
|---|---|---|---|
| `mk_` · `us_map_` | market | `gm_usa_` · `gm_kor_` | geomap |
| `single_` | page_single | `cmp_` · `pf_` | page_compare · page_portfolio (`cmp_` 종목·기간은 두 페이지가 같이 씀) |
| `bt_` | backtest | `sc_` | screener |
| `rs_` · `dart_` | research · outlook · research_chart | `cr_` · `cds_` | credit |
| `ins_` | insider | `opt_` · `sb_` · `ov_` · `cal_` · `am_` | 옵션 체인·전략·변동성·미국식 |
| `hedge_` · `hm_` · `hs_` · `sd_` · `rnd_` · `dgv_` · `capm_` | 헷지·시뮬·CAPM | `kr_` · `krh_` · `bk_` | 국내 옵션 · 내 포지션 (kropt · optbook) |
| `bd_` · `bdb_` · `bdc_` · `cb_` · `cv_` · `imm_` | 채권 | | |

새 위젯을 만들 때 파일 접두어를 붙이면 다른 화면과 겹치지 않는다. 밑줄로 시작하는 키(`_…`)는 페이지 이동 후 되살리지 않는다.

## 4. 데이터 출처 · 캐시

| 출처 | 쓰는 곳 | 캐시 |
|---|---|---|
| Yahoo Finance (yfinance) | 주가·재무·옵션·PER | 시세 5분, 일봉 1시간, 재무 6~24시간 |
| FinanceDataReader (KRX 시세·상장 정보, S&P 500 목록) | 시장·스크리너·지도 | 5분 / 1일 |
| 나스닥 스크리너 API | 미국 시세 | 5분 |
| SEC EDGAR | 미국 10개년 재무 | 1일 |
| OpenDART (키는 Streamlit Secrets `DART_API_KEY`) | 국내 재무·지분·본사 주소 | 1일~7일 |
| FRED · ECOS(샘플 키) | 금리 곡선 | 1시간~1일 |
| 위키백과 (S&P 500 본사, 도시 좌표) | 미국 지도 | 7일 |
| southkorea-maps (KOSTAT) | 한국 시도 경계 | 30일 |

## 5. 화면 규칙

- 상승 빨강 · 하락 파랑 (국내 관행). 금액은 원/달러 단위를 붙여 표시
- 라벨은 쉬운 말로, 요청하지 않은 항목은 넣지 않는다. 투자 유의 문구는 작게 한 줄
- 무거운 계산(점수·PER 등)은 버튼을 눌러 따로 불러오고 진행률을 보여준다
- 가려진 탭·더보기는 펼쳤을 때만 계산한다 (`ui.lazy_tabs`, `ui.more`)

## 6. 고칠 때 순서

1. 이 문서에서 파일을 찾고, 그 파일 머리말(docstring)에서 구성을 본다
2. 새 파일을 만들면 `autoreload.ORDER`에 넣고, 이 문서 2절에 한 줄 추가
3. 여러 파일이 같이 쓰는 기능은 `ui.py`(화면) 또는 그 파트의 데이터 파일에 둔다. 페이지 파일(`options.py` 같은 탭 묶기)에서 가져다 쓰지 않는다
4. GitHub에 올리면 앱을 새로고침할 때 `autoreload`가 바뀐 파일을 다시 불러온다. 앱 맨 위에 노란 경고가 뜨면 그 파일을 불러오다 실패한 것
