# BERT를 활용한 트럼프의 관세정책에 대한 각 언론사별 사람들의 반응 분석

---
![img.png](img.png)

<!--
badge 아이콘 참고 사이트
https://github.com/danmadeira/simple-icon-badges
-->

<img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white" />
<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/tensorflow-%23FF6F00.svg?&style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/numpy-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/pandas-%23150458.svg?&style=for-the-badge&logo=pandas&logoColor=white" />

---

## 1. 이 주제는 왜?
트럼프 전 미국 대통령이 2018년부터 시행한 일련의 관세 정책은 미국 중심의 보호무역주의를 전면에 내세운 대표적인 정책으로 평가됩니다. 그는 특히 중국을 중심으로 한 주요 무역국에 대해 고율의 관세를 부과하며, 자국 산업 보호와 무역 적자 해소를 강조했습니다. 이 정책은 단순한 경제 조치에 그치지 않고, 국제 정치, 글로벌 공급망, 환율 정책, 소비자 물가 등 다양한 분야에 중대한 파급 효과를 가져왔습니다.

하지만 이러한 관세 정책은 미국 내에서도 **정치적 논쟁의 중심**에 있었으며, 언론사 및 정파에 따라 **상반된 평가**를 받았습니다. 보수 성향의 언론은 자국 산업 보호라는 측면에서 긍정적으로 평가한 반면, 진보 성향의 언론은 글로벌 공급망 교란과 소비자 피해, 무역 전쟁의 부작용 등을 비판했습니다. 이에 따라 대중 역시 해당 정책에 대해 상이한 태도와 의견을 보여주었습니다.

특히 유튜브와 같은 플랫폼은 언론사가 콘텐츠를 배포하는 창구로 기능할 뿐 아니라, **대중의 실시간 의견이 반영되는 공간**이기도 합니다. 뉴스 클립, 풍자 프로그램, 인터뷰 영상 등에 남겨진 댓글을 통해 시청자의 감정, 입장, 그리고 사회적 분위기를 직관적으로 파악할 수 있습니다. 본 프로젝트는 이러한 유튜브 댓글을 분석함으로써, 언론사가 전달하는 메시지와 그것에 대한 대중의 반응 사이의 관계를 구체적으로 살펴보고자 합니다.

분석 대상은 다음 세 언론사로 구성됩니다:

| 언론사 | 성향 | 주요 특징 |
|--------|------|------------|
| **CNBC** | 중립 ~ 보수 | 경제 전문 채널로 정책의 **실효성** 및 **시장 반응**에 중점 |
| **MSNBC** | 진보 | 정책의 **정치적 영향** 및 **윤리적 정당성**에 대한 비판적 보도 |
| **The Late Show with Stephen Colbert** | 진보 (풍자 중심) | **풍자와 유머**를 통한 간접적 비판, **정치 풍자 토크쇼** 형식 |

### 언론사 선정기준 
본 프로젝트는 유튜브에서 **트럼프의 관세 정책**을 주제로 한 영상들 중, **조회수와 댓글 수가 많아 대중 반응이 활발한 콘텐츠**를 기준으로 언론사를 선정하였습니다.

| 언론사 | 영상 제목 | 조회수   | 댓글 수    |
|--------|---------------------------|-------|---------|
| **CNBC** | How Companies Are Dodging Trump Tariffs On Canada, Mexico And China | 420만회 | 12,644개 | 
| **MSNBC** | Lawrence: Canada's Trudeau humiliates 'cowardly' Trump who backs down on tariffs. Again. | 902만회 | 31,340개 |
| **The Late Show with Stephen Colbert** | Dumbest Trade War In History Effects Of Trump's Tariffs Already Being Felt Chicken Rental Deals | 415만회 | 8,618개  |

본 분석을 통해 다음과 같은 질문에 답하고자 했습니다,:

- 언론사별로 트럼프의 관세 정책에 대한 대중 반응은 어떻게 달랐는가?
- 언론사의 성향이 시청자나 대중들의 반응에 영향이 있는가?
- 보도 방식과 시청자 의견 간에는 어떤 연관성이 있는가?
- 감성 분석 결과를 통해 미국 내 여론 지형과 정치 성향을 예측할 수 있는가?

이 연구는 단순한 여론 조사나 댓글 통계를 넘어서, **딥러닝 기반 자연어 처리 모델(BERT)**을 활용하여 **텍스트의 맥락 속 감정과 입장을 정량적으로 분석**하고, **정치 담론의 확산 구조를 탐색**하는 의의를 두었습니다.

---
## 2. 데이터 수집 및 전처리

###  데이터 출처
가장 대중적이고 즉각적인 반응을 확인할 수 있는 유튜브 플랫폼을 활용하였습니다. 다음 3개 언론사의 유튜브 영상에서 댓글을 수집하였습니다:

-  CNBC: [https://www.youtube.com/watch?v=h5P8WHBrQvo](https://www.youtube.com/watch?v=h5P8WHBrQvo)  
-  MSNBC: [https://www.youtube.com/watch?v=arHHAfYbM-M](https://www.youtube.com/watch?v=arHHAfYbM-M)  
-  The Late Show with Stephen Colbert: [https://www.youtube.com/watch?v=F90YWg11UAU](https://www.youtube.com/watch?v=F90YWg11UAU)

###  수집 방법
- Python 기반 크롤링 코드 활용 (Google Colab에서 실행)  
- 참고 블로그: [Naver 블로그 링크](https://m.blog.naver.com/galaxyworldinfo/223615648013)
- 수집된 항목: 댓글 본문, 작성 시점, 해당 언론사 정보
- ### 수집한 유튜브 댓글 예시

| text | time | channel | label |
|------|------|---------|---|
| This is absolutely mind-blowing! | 14 hours ago | CNBC | 1 |
| Companies should start price labelling their products with the (RTT) Republican Trump Tax. e.g. $4.99 + RTT $12.23 | 3 days ago | CNBC | 1 |
| Tariffs are paid by the companies. Consumers pay for products at retail, and if that company increases a price then they may lose that sale to a competitor. If one company wants to gain market-share then they can eat more of the costs while contracting existing US shoe makers to tool up for making their shoes. None of the companies making products like clothing have the monopoly needed for passing the costs completely to the customer. | 8 days ago | CNBC | 2 |


###  데이터 탐색 (EDA)

- 총 수집된 댓글 수: **54,302건**
- 각 댓글에 대해 언론사 라벨과 작성 날짜 정보 포함
- 영어 댓글만 수집되어 자연어 처리에 적합
---
## 3. 학습 데이터 구축

###  목표

총 54,302개의 댓글 중 일부 샘플을 정제 및 라벨링하여 BERT 모델 학습을 위한 데이터셋을 구성하였습니다.

###  라벨링 기준

| 라벨 | 의미 | 설명 |
|------|------|------|
| **0 (긍정)** | 지지 | 트럼프의 관세 정책에 대해 **명확히 긍정적이거나 지지하는 입장** 표현 |
| **1 (부정)** | 반대 | 정책에 대해 **비판적**, **조롱**, **반대 의견**을 표현 |
| **2 (중립)** | 감정 없음 | **정보 전달**, **유머**, **맥락 없는 언급** 등 **감정적 입장이 드러나지 않는 경우** |

### 세부 사항

- 전체 댓글 중 **무작위로 2,000건을 추출**하여 수작업으로 라벨링 진행
- 각 언론사별로 약 **600여 건씩 균형 있게** 포함
- 데이터의 라벨 분포는 모델 학습 시 클래스 불균형을 고려하여 분석

### Case 2. 수집된 데이터 활용
  - 10~20% 추출에 대한 기준
    + 긍정이 70%(35,000), 부정 30%(15,000) 이라면
    + (1안) 10%를 추출 : 70%(3,500), 부정 30%(1,500)
    + (2안) 10%를 추출 : 50%(2,500), 부정 50%(2,500)
    + 추가 고려도 가능하다 (분류, 시점)

## 4. MobileBERT Finetuning(재학습, 미세조정) 결과
 - 학습 데이터 전체의 수가 5,000건 이라면, 실제 학습을 할 때는 검증 데이터를 일부 추출해야 한다. 학습:검증은 8:2나 7:3 정도를 쓴다.
 - 4,000:1,000 혹은 3,500:1,500 으로 학습:검증 데이터를 나누고
 - MobileBERT를 학습시킨 후 1. training loss, 2. training accuracy & validation accuracy 두 개의 그래프(x축 epoch)를 그리고 값을 표로 제공한다.
 - 수집된 데이터가 있는 경우에는 전체 데이터셋에 inference하여 test accuracy 수치를 작성한다.
 - 직접 수집한 경우에는 라벨이 없으므로 문장 분류 예측 결과를 나타낸다. (라벨별로 데이터 분포를 말한다.)

## 5. 결론 및 느낀점