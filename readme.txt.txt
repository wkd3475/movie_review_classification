dataset : {'review', 'label(0 or 1)'}

1. dataset 만들기
(1) review 정제가 필요함
=> <br /> \' () 같이 필요없어보이는 값들은 삭제할 필요가 있음

(2) training set, learning set 나누기

2. 전체 review를 이용한 Word2Vec
다양하게 시도해볼 수 있을 듯
- basic softmax
- hierarchical softmax
- negative sampling
- subsampling
- fasttext

3. 문장 인코더-디코더
- RNN
- bidirectional 인코더 등등

좋은거 찾아서 써보면 좋을 듯!!
=> 조사가 필요함!

4. 위에서 만든 벡터들을 이용해서 Pos/Neg 학습

5. 실험