get_dataset()
���� ù ���ڷ� movie_data.csv�� ��� ��θ� �־��ָ� ��

2019/04/11 AM 12:16

2019/04/12 AM 3:51
�� �ܰ迡�� �н� �Ŀ� ����� ���Ϸ� ������ �ϵ��� ��������
num_review(���� ������ �¿��� review�� ����� ����)�� 50, 3000���� CBOW, SG ���� �н��ص�



dataset : {'review', 'label(0 or 1)'}

1. dataset �����
(1) review ������ �ʿ���
=> <br /> \' () ���� �ʿ����̴� ������ ������ �ʿ䰡 ����

(2) training set, learning set ������

2. ��ü review�� �̿��� Word2Vec
�پ��ϰ� �õ��غ� �� ���� ��
- basic softmax
- hierarchical softmax
- negative sampling
- subsampling
- fasttext

++ Word2Vec ���� GloVe�� distributed representation�� �ִٰ���

3. ���� ���ڴ�-���ڴ�
- RNN
- bidirectional ���ڴ� ���

������ ã�Ƽ� �Ẹ�� ���� ��!!
=> ���簡 �ʿ���!

4. ������ ���� ���͵��� �̿��ؼ� Pos/Neg �н�

5. ����