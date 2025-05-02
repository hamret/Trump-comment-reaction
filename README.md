# MobileBERT를 활용한 트럼프의 관세정책에 대한 각 언론사별 사람들의 반응

---
![img.png](img.png)

<!--
badge 아이콘 참고 사이트
https://github.com/danmadeira/simple-icon-badges
-->

<img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white" /> <img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white" />

---
## 1. 이 주제는 왜?

    트럼프 전 대통령의 관세정책은 세계 경제에 큰 파장을 일으켰으며, 한국 경제에도 적지 않은 영향을 미쳤다. 
    이러한 정책에 대해 한국 언론사별로 유튜브 댓글에서 어떤 반응이 주를 이루는지를 분석하는 것은 언론의 영향력과 대중 인식의 흐름을 파악하는 데 
    중요한 의미가 있다. 특히 언론사에 따라 중립적, 지지적, 혹은 비판적 태도의 차이가 존재한다면,
    이는 뉴스 소비자에게 전달되는 정보의 방향성과 여론 형성에 미치는 영향을 보여줄 수 있다. 
    본 조사를 통해 우리는 언론사별 보도방향과 시청자 반응 간의 연관성을 확인하고자 하며, 
    이를 통해 미디어의 역할과 책임에 대한 논의의 기반을 마련하고자 한다.
---
## 2. 데이터 수집
우선 트럼프의 관세 정책에 대한 반응을 가장 빠르게 확인할 수 있는곳은 유튜브라고 판단해 유튜브 댓글을 수집하였다.

- 유튜브 댓글 수집 코드는 코랩으로 실행하였고 https://m.blog.naver.com/galaxyworldinfo/223615648013 해당 블로그를 참고해 크롤링을 진행하였다. 
- 수집한 영상 목록
    + https://www.youtube.com/watch?v=h5P8WHBrQvo (CNBC),
    + https://www.youtube.com/watch?v=arHHAfYbM-M (MSNBC),
    + https://www.youtube.com/watch?v=F90YWg11UAU (The Late Show with Stephen Colbert)
)
 - 데이터에 대한 EDA
   + 총 데이터는 약 54302건의 댓글을 수집
   + 각 댓글이 어떤 언론사의 유튜브에 달린 댓글인지 라벨링, 작성 시점도 같이 수집하였음
---
## 3. 학습 데이터 구축
총 데이터의 수가 50,000건 이라고 한다면 이 중에서 10%~20%를 추출하여 학습데이털 만든다. **학습 데이터 구축의 핵심은 전체 데이터에서 일부를 그저 랜덤하게 추출하는 것이 아닌, 전체 데이터의 분포를 고려한 학습 데이터이어야 일반화의 가능성이 높아진다. --> 예측 성능이 좋아진다.**
### Case 1. 직접수집
 - 라벨링
   + 기준, 예시
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