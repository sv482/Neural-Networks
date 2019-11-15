import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os

import time
from tqdm import tqdm
from data_loader import fetch_data

import gensim
from gensim.models import Word2Vec

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,n_layers): # Add relevant parameters
        super(RNN, self).__init__()
        # Fill in relevant parameters
        self.rnn=nn.RNN(input_dim,hidden_dim, n_layers)
        self.fc=nn.Linear(hidden_dim,5)
        
        # Ensure parameters are initialized to small values, see PyTorch documentation for guidance
        self.softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)  
    
    def forward(self, input_vector): 
        h0 = torch.zeros(1,1,32)
        n_words = input_vector.size(0)
        n_embed = input_vector.size(1)
        input_vector = input_vector.reshape(n_words, 1, n_embed)
        output,hidden = self.rnn(input_vector,h0)
        #out = output.contiguous().view(-1, self.hidden_dim)
        
        hidden = hidden.squeeze(0).squeeze(1)
        out=self.fc(hidden)
        predicted_vector = self.softmax(out)
        return predicted_vector
    
# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = []
    for document, _ in data:
        for i in range(len(document)):
            v=document[i].split()
            vocab.append(v)
    vocab.append([unk])
    return vocab 
    
def unique_words(data):
    vocab_set = set()
    for document, _ in data:
        for word in document:
            vocab_set.add(word.lower())
    vocab_set.add(unk)
    return vocab_set 

def vectorization(document,Vocab,vocab_set):
    model=Word2Vec(Vocab,size=100,min_count=1)
    OurTensor=torch.zeros(len(document),100)
    for word in document:
        for i in range(0,len(document)):
            if word in vocab_set:
                OurTensor[i] = torch.FloatTensor(model[word])
            else:
                OurTensor[i]=torch.FloatTensor(model[unk])
    return OurTensor          

def main(hidden_dim, number_of_epochs):
    print('Fetching data')
    train_data, valid_data = fetch_data() # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab_set= unique_words(train_data)
    print("Vectorized data")
    
    a = vectorization(train_data[0][0],vocab, vocab_set)
    print(a.size())
    
    model = RNN(input_dim =100, hidden_dim= hidden_dim, n_layers=1)
    optimizer=optim.Adam(model.parameters())
    print("Training for {} epochs".format(number_of_epochs))
    for epoch in range(number_of_epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16 
        N = len(train_data)
#        N = 10
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                document, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_vector=vectorization(document,vocab,vocab_set)
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        random.shuffle(valid_data) # Good practice to shuffle order of validation data
        minibatch_size = 16
        model.eval()
        N = len(valid_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                document, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                input_vector=vectorization(document,vocab,vocab_set)
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            #loss.backward()
            #optimizer.step()
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

main(32, 1)

    