import csv
import pickle
import os
from collections import Counter

#movie_data.csv => encoding = 'UTF8'

def get_dataset(file_, encoding_, type_):
    datas = []
    
    f = open(file_, 'r', encoding=encoding_)

    lists = csv.reader(f)
    i = 0
    if not type_ == 'full':
        for list_ in lists:
            if i == 0:
                pass
            elif i <= type_:
                datas.append({'review':list_[0], 'sentiment':list_[1]})
            else:
                break
            i += 1
    elif type_ == 'full':
        for list_ in lists:
            if not i == 0:
                datas.append({'review':list_[0], 'sentiment':list_[1]})
            i += 1

    return datas

def get_unigram_voca(datas, num_review):
    corpus = []
    for data in datas:
        for voca in data['review'].split():
            corpus.append(voca)

    stats = Counter(corpus)
    words = []
    #Discard rare words
    for word in corpus:
        if stats[word]>0:
            words.append(word)

    freqtable = []
    vocab = set(words)

    #Give an index number to a word
    w2i = {}
    w2i[" "]=0
    i = 1
    for word in vocab:
        w2i[word] = i
        i += 1
    i2w = {}
    for k,v in w2i.items():
        i2w[v]=k

    #Frequency table for negative sampling
    for k,v in stats.items():
        f = int(v**0.75)
        for _ in range(f):
            if k in w2i.keys():
                freqtable.append(w2i[k])

    train_set = []
    window_size = 5
    for j in range(len(words)):
        if j<window_size:
            contextlist = [0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)]
            train_set.append((w2i[words[j]],contextlist))
        elif j>=len(words)-window_size:
            contextlist = [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)]
            train_set.append((w2i[words[j]],contextlist))
        else:
            contextlist = [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)]
            train_set.append((w2i[words[j]],[w2i[words[j-1]],w2i[words[j+1]]]))

    save_data_extract(w2i, i2w, freqtable, train_set, num_review, len(w2i))
    return w2i, i2w, freqtable, train_set

def get_splited_reviews(datas, w2i):
    splited_reviews =[]
    i = 0
    for data in datas:
        splited_reviews.append({'review':[], 'sentiment':None})
        for voca in data['review'].split():
            if not w2i[voca] == None:
                splited_reviews[i]['review'].append(voca)
        splited_reviews[i]['sentiment'] = data['sentiment']
        i += 1

    return splited_reviews

def save_data_extract(w2i, i2w, freqtable, train_set, num_review, numwords):
    dir_ = '%d_data_extract_%d' %(num_review, numwords)
    try:
        if not os.path.exists(os.path.dirname(dir_)):
            os.makedirs(os.path.join(dir_))
    except:
        pass
    
    file_ = open(dir_+'/w2i.pkl', 'wb')
    pickle.dump(w2i, file_)
    file_.close()

    file_ = open(dir_+'/i2w.pkl', 'wb')
    pickle.dump(i2w, file_)
    file_.close()

    file_ = open(dir_+'/freqtable.pkl', 'wb')
    pickle.dump(freqtable, file_)
    file_.close()

    file_ = open(dir_+'/train_set.pkl', 'wb')
    pickle.dump(train_set, file_)
    file_.close()

    print("Save data_extract in %s" %(dir_))
                    
def load_data_extract(num_review, numwords):
    dir_ = '%d_data_extract_%d' %(num_review, numwords)
    try:
        file_ = open(dir_+'/w2i.pkl', 'rb')
        w2i = pickle.load(file_)
        file_.close()
        
        file_ = open(dir_+'/i2w.pkl', 'rb')
        i2w = pickle.load(file_)
        file_.close()
        
        file_ = open(dir_+'/freqtable.pkl', 'rb')
        freqtable = pickle.load(file_)
        file_.close()
        
        file_ = open(dir_+'/train_set.pkl', 'rb')
        train_set = pickle.load(file_)
        file_.close()
    except:
        print('There are no (%d)_(%d) files.' %(num_review, numwords))
        return -1, -1, -1, -1

    print("Load data_extract in %s" %(dir_))
    return w2i, i2w, freqtable, train_set
