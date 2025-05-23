# BERT를 활용한 트럼프의 관세정책에 대한 각 언론사별 사람들의 반응 분석

---
![img.png](img.png)

<!--
badge 아이콘 참고 사이트
https://github.com/danmadeira/simple-icon-badges
-->

<img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white" /> <img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white" />

---
## 1. 이 주제는 왜?
트럼프 전 미국 대통령의 관세정책은 자국 산업 보호와 무역 불균형 해소를 명분으로 시행되었으며, 
세계 경제에 큰 파장을 일으켰다. 특히 미국 내에서도 이 정책에 대한 반응은 언론사별로 상이했으며, 
대중의 의견 역시 다양하게 나타났다. 
본 조사는 미국 주요 언론사인 CNBC, MSNBC, 그리고 The Late Show with Stephen Colbert의 유튜브 채널에 게시된 관련 영상의 댓글을 수집하고, 
각 언론사별로 시청자 반응이 중립적인지, 지지적인지, 혹은 비판적인지를 분석하고자 한다. 
이를 통해 우리는 미국 내 주요 언론이 트럼프의 관세정책을 어떤 시각에서 다루었고, 
이에 대한 시청자들의 수용 태도는 어떠했는지를 확인할 수 있다. 
또한 정치적 성향에 따라 언론사와 대중의 반응에 차이가 존재하는지를 파악함으로써, 
언론과 여론 간 상호작용의 구조를 이해하는 데 기여할 수 있다. 
궁극적으로는 정책 담론 형성 과정에서 언론과 대중 의견의 관계를 밝히는 데 의미가 있다.

---
## 2. 데이터 수집
우선 트럼프의 관세 정책에 대한 반응을 가장 빠르게 확인할 수 있는곳은 유튜브라고 판단해 유튜브 댓글을 수집하였다.

- 유튜브 댓글 수집 코드는 코랩으로 실행하였고 https://m.blog.naver.com/galaxyworldinfo/223615648013 해당 블로그를 참고해 크롤링을 진행하였다. 
- 수집한 영상 목록
    + https://www.youtube.com/watch?v=h5P8WHBrQvo (CNBC),
    + https://www.youtube.com/watch?v=arHHAfYbM-M (MSNBC),
    + https://www.youtube.com/watch?v=F90YWg11UAU (The Late Show with Stephen Colbert)

 - 데이터에 대한 EDA
   + 총 데이터는 54302건의 댓글을 수집
   + 각 댓글이 어떤 언론사의 유튜브에 달린 댓글인지 라벨링, 작성 시점도 함께 수집하여 자료 분석에 활용하였음
---
## 3. 학습 데이터 구축
총 데이터의 수 54,302건 중에서 각 언론사별로 600개정도의 데이터를 추출해서 2000건의 데이터를 따로 분류후
긍정 0, 부정 1, 중립 2 로 라벨링하였음
### Case 1. 직접수집
 - 라벨링
   + 유튜브에서 전부 영어로 된 댓글들만 수집하였으며
   + 총 라벨링 건수 등
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