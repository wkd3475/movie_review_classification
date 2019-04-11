# -*- coding: cp949 -*-
import csv
from collections import Counter

#movie_data.csv => encoding = 'UTF8'

#데이터 값들이 정제된 상태라고 가정
def get_dataset(file, encoding_, type_):
    datas = []
    
    f = open(file, 'r', encoding=encoding_)

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

def get_unigram_voca(datas):
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

    return words, freqtable, train_set, w2i, i2w
