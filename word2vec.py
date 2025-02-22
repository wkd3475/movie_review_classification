import torch
import os
import csv
from random import shuffle
from collections import Counter

def skipgram(centerWord, contextWord, inputMatrix, outputMatrix):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWord : Index of a contextword (type:int)                       #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
#########################################################################
    V = inputMatrix.shape[0]
    D = inputMatrix.shape[1]

    h = inputMatrix[centerWord].reshape(D, 1)
    #h.shape = (D,1)

    o = torch.mm(outputMatrix, h)
    e = torch.exp(o - torch.max(o))
    softmax = e / torch.sum(e)
    #softmax.shape = (V, 1)

    loss = -torch.log(softmax[contextWord]+0.00001)
    softmax[contextWord] = softmax[contextWord] - 1
    
    grad_in = torch.mm(softmax.reshape(1, V), outputMatrix)
    #grad_in.shape = (1, D)
    grad_out = torch.mm(softmax, h.reshape(1, D))
    #grad_out.shape = (V, D)
###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))            #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################

    return loss, grad_in, grad_out

def CBOW(centerWord, contextWords, inputMatrix, outputMatrix):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWords : Indices of contextwords (type:list(int))               #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
#########################################################################
    V = inputMatrix.shape[0]
    D = inputMatrix.shape[1]

    h = torch.zeros(1, D)
    for word in contextWords:
        h = h + inputMatrix[word]
    h = h.reshape(D, 1)
    #h.shape = (D,1)
    
    o = torch.mm(outputMatrix, h)
    e = torch.exp(o - torch.max(o))
    softmax = e / torch.sum(e)
    #softmax.shape = (V, 1)
    
    loss = -torch.log(softmax[centerWord]+0.00001)
    softmax[centerWord] = softmax[centerWord] - 1
    
    grad_in = torch.mm(softmax.reshape(1, V), outputMatrix)
    #grad_in.shape = (1, D)
    grad_out = torch.mm(softmax, h.reshape(1, D))
    #grad_out.shape = (V, D)
###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))            #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################

    return loss, grad_in, grad_out


def word2vec_trainer(train_seq, numwords, stats, mode="CBOW", dimension=100, learning_rate=0.0025, epoch=3, using_W_files = 0, num_review=500):
# train_seq : list(tuple(int, list(int))
    
# Xavier initialization of weight matrices
    W_in = None
    W_out = None
    
    if using_W_files == 0:
        print("Init weight random...")
        W_in = torch.randn(numwords, dimension) / (dimension**0.5)
        W_out = torch.randn(numwords, dimension) / (dimension**0.5)
    elif using_W_files == 1:
        print("Use existing weight...")
        try:
            W_in, W_out = load_weight(mode, num_reivew, numwords, dimension)
        except:
            print("No weight files...")
            print("New weight...")
            W_in = torch.randn(numwords, dimension) / (dimension**0.5)
            W_out = torch.randn(numwords, dimension) / (dimension**0.5)
            
    i=0
    losses=[]

    print("# of training samples")
    if mode=="CBOW":
    	print(len(train_seq))
    elif mode=="SG":
    	print(len(train_seq)*len(train_seq[0][1]))
    print()

    for _ in range(epoch):
        #Random shuffle of training data
        shuffle(train_seq)
        #Training word2vec using SGD(Batch size : 1)
        for center, contexts in train_seq:
            i+=1
            centerInd = center
            contextInds = contexts
            if mode=="CBOW":
                L, G_in, G_out = CBOW(centerInd, contextInds, W_in, W_out)
                
                W_in[contextInds] -= learning_rate*G_in
                W_out -= learning_rate*G_out

                losses.append(L.item())
            elif mode=="SG":
            	for contextInd in contextInds:
	                L, G_in, G_out = skipgram(centerInd, contextInd, W_in, W_out)
	                W_in[centerInd] -= learning_rate*G_in.squeeze()
	                W_out -= learning_rate*G_out

	                losses.append(L.item())
            else:
                print("Unkwnown mode : "+mode)
                exit()

            if i%1000==0:
            	avg_loss=sum(losses)/len(losses)
            	print("%d  Loss : %f" %(i, avg_loss,))
            	losses=[]
    
    return W_in, W_out

def sim(testword, word2ind, ind2word, matrix):
    length = (matrix*matrix).sum(1)**0.5
    wi = word2ind[testword]
    inputVector = matrix[wi].reshape(1,-1)/length[wi]
    sim = (inputVector@matrix.t())[0]/length
    values, indices = sim.squeeze().topk(5)
    
    print()
    print("===============================================")
    print("The most similar words to \"" + testword + "\"")
    for ind, val in zip(indices,values):
        print(ind2word[ind.item()]+":%.3f"%(val,))
    print("===============================================")
    print()

def check_file(file_path):
    return os.path.exists(file_path)

def save_word2vec(W_in, W_out, mode, num_review, numwords, dimension):
    dir_ = '%d_word2vec_%s_%dx%d' %(num_review, mode, numwords, dimension)
    try:
        if not os.path.exists(os.path.dirname(dir_)):
            os.makedirs(os.path.join(dir_))
    except:
        pass
    
    torch.save(W_in, dir_+'/W_in.pth')
    torch.save(W_out, dir_+'/W_out.pth')

    print("Save word2vec in %s" %(dir_))

def load_word2vec(num_review, numwords, mode, dimension):
    dir_ = '%d_word2vec_%s_d_%d' %(num_review, mode, numwords, dimension)
    try:
        W_in = torch.load(dir_+'/W_in.pth')
        W_out = torch.load(dir_+'/W_out.pth')
    except:
        print('There are no %d_%s_%dx%d files.' %(num_review, mode, numwords, dimension))
        return -1, -1

    print("Load word2vec in %s" %(dir_))
    return W_in, W_out
