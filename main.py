# -*- coding: cp949 -*-
import get_data as gd
import word2vec as w2v
import classification

#default
num_review = 50

numwords = 4209
mode = "SG"
dimension = 64

def data_extract():
    global numwords
    print("1. Start data_extract")
    print('Load review (%d)...' %(num_review))
    datas = gd.get_dataset('../dataset/movie_data.csv', 'UTF8', num_review)
    w2i, i2w, freqtable, train_set = gd.get_unigram_voca(datas, num_review)
    numwords = len(w2i)
    
def word2vec():
    print("\n2. Start word2vec")
    w2i, i2w, freqtable, train_set = gd.load_data_extract(num_review, numwords)
    #Set word2vec variables

    print('Training (mode : %s)...' %(mode))
    
    #word2vec
    W_in, W_out = w2v.word2vec_trainer(train_set, numwords=numwords, stats=freqtable, mode=mode, dimension=dimension, epoch=1, learning_rate=0.05, using_W_files = 0, num_review=num_review)
    w2v.save_word2vec(W_in, W_out, mode, num_review, numwords, dimension)
    
    #Test
    testwords = ["it"]
    for tw in testwords:
    	w2v.sim(tw,w2i,i2w,W_in)

def classification():
    datas = gd.get_dataset('../dataset/movie_data.csv', 'UTF8', num_review)
    


def CNN():
    #Load word2vec
    W_in, W_out= w2v.load_word2vec(num_review, numwords, mode, dimension)

    print(gd.get_splited_reviews(datas, w2i)[0])

#data_extract()
word2vec()
