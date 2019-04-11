import get_data as gd
import word2vec as w2v

def main():
    
    #Set num_review and get data set
    num_review = 500
    print('load dataset (%d)...' %(num_review))
    datas = gd.get_dataset('../dataset/movie_data.csv', 'UTF8', num_review)
    words, freqtable, train_set, w2i, i2w = gd.get_unigram_voca(datas)

    #Set word2vec variables
    mode = "CBOW"
    using_W_files = 1
    numwords = len(w2i)
    dimension = 64

    print('training (mode : %s)...' %(mode))
    """
    W_in, W_out = w2v.word2vec_trainer(train_set, numwords=numwords, stats=freqtable, mode=mode, dimension=dimension, epoch=1, learning_rate=0.05, using_W_files = using_W_files, num_review=num_review)
    w2v.save_weight(W_in, W_out, mode, num_review, numwords, dimension)

    #Test
    testwords = ["good", "bad"]
    for tw in testwords:
    	w2v.sim(tw,w2i,i2w,W_in)
    """
    W_in, W_out = w2v.load_weight(mode, num_review, numwords, dimension)
    
main()
