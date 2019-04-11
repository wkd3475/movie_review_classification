import word2vec as word2vec
import get_data as gd

def main():
    print('load dataset...')
    datas = gd.get_dataset('../dataset/movie_data.csv', 'UTF8', 100)
    words, freqtable, train_set, w2i, i2w = gd.get_unigram_voca(datas)
    mode = "CBOW"
    print('training (mode : %s)...' %(mode))
    W_in, W_out = word2vec.word2vec_trainer(train_set, len(w2i), freqtable, mode=mode, dimension=64, epoch=3, learning_rate=0.05)

    testwords = ["good", "it"]
    for tw in testwords:
    	word2vec.sim(tw,w2i,i2w,W_in)
main()
