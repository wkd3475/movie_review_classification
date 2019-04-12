목표
1. 자연어 처리 수업에서 배운 내용들을 실제로 적용해본다
2. 영화 리뷰가 긍정을 의미하는지 부정을 의미하는지 분리를 해본다
3. 최신 기술들을 적용해본다(사용에도 의미를 두고, 최적화에도 의미를 둠)

해결 과제
1. 함수 이름이 깔끔하지 않은 상태... => 주석으로 설명 달아두자
2. CNN or RNN 어려움...
3. fasttext 시도해보고 싶음
4. hierarchical softmax 해보면 좋을 것 같음
5. 시도해볼 것들 정리가 필요함(시도해보고 싶은 최신 기술)
6. review 값이 깔끔하지 않은 상태... 필터링이 필요함

###2019/03/29
- 1차 회의 : 목표 설정

###~2019/04/08
- 자연어 처리 과제를 진행하며 공부

###2019/04/09
- 2차 회의 : dataset 정하기, 스터디

###2019/04/10 PM 11:05
- 프로젝트 개발 시작

###2019/04/11 AM 12:16
- get_data() 구현
- word2vec() 구현

###2019/04/12 AM 3:51
- 각 단계에서 학습 후에 결과를 파일로 저장을 하도록 수정했음
- num_review(실제 데이터 셋에서 review를 사용한 갯수)를 50, 3000으로 CBOW, SG 각각 학습해둠



dataset : {'review', 'label(0 or 1)'}
splited_dataset : {[r1, r2, r3, ... , rn], 'label(0 or 1)'} *예정*

1. dataset 만들기
(1) review 정제가 필요함
=> <br /> \' () 같이 필요없어보이는 값들은 삭제할 필요가 있음

(2) training set, learning set 나누기 필요함

(3) 문장을 벡터화 하기 위한 준비
- get_splited_reviews()

2. 전체 review를 이용한 Word2Vec
다양하게 시도해볼 수 있을 듯
- basic softmax (v)
- hierarchical softmax
- negative sampling (v)
- subsampling
- fasttext

++ Word2Vec 말고 GloVe의 distributed representation도 있다고함

3. 문장 인코더-디코더
- RNN
- bidirectional 인코더 등등

좋은거 찾아서 써보면 좋을 듯!!
=> 조사가 필요함!

4. 위에서 만든 벡터들을 이용해서 Pos/Neg 학습

5. 실험