# -*- coding: utf-8 -*-
# !nvcc --version
# !pip install mxnet-cu100

import re
import os
import math
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn
import time
import random
import pickle

def load_data(directory):
    """
    directory: where the cleaned data.txt file is located    
    
    """
    data_dir="./data/"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
      
    with open(directory) as file:
        data = file.read()
    
    clean_data = []
    for poetry in data.split('\n'):
        pattern = r'[\u4e00-\u9fa5]+' # matching all Chinese, remove others string
        text = re.findall(pattern, poetry)
        drop = False
        for sent in text:
            if len(sent)!=5:
                drop = True
                break
        if drop == False:
            clean_data.append(poetry)
    
    pattern = r'[\u4e00-\u9fa5]+' # matching all Chinese, remove others string
    corpus_chars = re.findall(pattern, ''.join(clean_data))
    corpus_chars = ''.join(corpus_chars)
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    
    with open(f"./data/char_to_idx.pkl", "wb") as fp:
        pickle.dump(char_to_idx, fp) 
    with open(f"./data/idx_to_char.pkl", "wb") as fp:
        pickle.dump(idx_to_char, fp) 
    with open(f"./data/vocab_size.pkl", "wb") as fp:
        pickle.dump(vocab_size, fp)   
    
    """
    corpus_indices => list of number(idx) in the corpus
    char_to_idx => dict[char] = number
    idx_to_char => list[char]
    vocab_size => len(set(list[char]))

    """
    
    return (corpus_indices, char_to_idx, idx_to_char, vocab_size)

class RNNModel(nn.Block):
    """
    create the model instance
    
    """
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        """
        vocab_size: num of unique character in the string
        rnn_layer: can pass in RNN, GRU, LSTM
        
        """
        
        super(RNNModel, self).__init__(**kwargs) # inherit from parent
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size) # final output layer

    def forward(self, inputs, state):
        """
        inputs: (batch_size, num_steps)
        state: previous hidden state
        
        """
        one_hot_X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(one_hot_X, state) # update state and make prediction
        """
        Y: the output before dense layer, size = (num_steps, batch_size, num_hidden)
        batch_size: how many sample in one batch
        num_hidden: number of neurons in one layer
        num_steps: how many step in one sample
        
        """
        output = self.dense(Y.reshape((-1, Y.shape[-1])))  # output size: (num_steps * batch_size, vocab_size)
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs) # initial the state

def predict(prefix, num_chars, model, vocab_size, ctx, idx_to_char,
                      char_to_idx):
    
    """
    prefix: the prefix the start the poetry
    num_chars: how many chars in the prediction
    ctx: whether the computation based on CPU or GPU
    
    """
    state = model.begin_state(batch_size=1, ctx=ctx) # make prediction one by one, so the batch_size = 1
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1): # we need to make (num_chars + len(prefix) - 1) predictions
        X = nd.array([output[-1]], ctx=ctx)
        (Y, state) = model(X.reshape((1, 1)), state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]]) # if we still have prefix, no need to predict
        else:
            output.append(int(Y.argmax(axis=1).asscalar())) # make prediction
    return ''.join([idx_to_char[i] for i in output])

def data_iter(corpus_indices, batch_size, num_steps, ctx = mx.cpu()):
    """
    randomly select the data, create data iterator for SGD
    
    """
    num_examples = (len(corpus_indices) - 1) // num_steps  
    example_indices = list(range(num_examples))
    random.shuffle(example_indices) # shuffle
    
    """
    num_examples: how many example available, no overlap, because of the input is poetry
    example_indices: inx correaponds with each example
    
    
    """
    for ind in range(num_examples // batch_size):
        batch_indices = example_indices[ind * batch_size: ind * batch_size + batch_size]
        inputs = [corpus_indices[j * num_steps: j * num_steps + num_steps] for j in batch_indices]
        outputs = [corpus_indices[j * num_steps + 1: j * num_steps + num_steps + 1] for j in batch_indices]
        yield nd.array(inputs, ctx), nd.array(outputs, ctx)

def gradient_clipping(params, theta, ctx = mx.cpu()):
    """
    grading_clipping to avoid gradient vanishing (and exploding)
    theta: the clipping parameters
    
    """  
    accumulator = 0
    for param in params:
        accumulator = accumulator + (param.grad ** 2).sum().asscalar() # accumulate all the gradient
    norm = nd.array([accumulator], ctx).sqrt().asscalar()
    if norm > theta: # if exceed the limit
        for param in params:
            param.grad[:] *= theta / norm # scaling

def train_and_predict(model_ind, num_hiddens, num_layer, bidirectional, ctx, lr, num_epochs, vocab_size, 
                      data_iteror, num_steps, corpus_indices, batch_size, idx_to_char, 
                      char_to_idx, clipping_theta,
                      pred_period, prefix):
    """
    model_ind: what type of model, e.g.: "RNN", "LSTM", "GRU"
    num_hiddens: how mamy hidden neurons in each layer
    num_layer: number of RNN/LSTM/GRU layer
    bidirectional: whether each layer is bidirectional
    ctx: train on cpu or gpu
    lr: learning rate
    num_epochs: how many epochs for training
    vocal_size: how many unique character in the train string
    data_iteror: the data iteror to generate input for each epoch
    num_steps: number of stepsfor each sample
    corpus_indices: list of number(idx) in the corpus
    batch_size: number of sample in one batch
    idx_to_char: idx_to_char => list[char]
    char_to_idx => dict[char] = number
    clipping_theta: gradient clipper parameter
    pred_period: the len to print result
    prefix: the prefix for the poetry generator
    
    """
    
    if model_ind == 'RNN':
        layer = rnn.RNN(hidden_size = num_hiddens, num_layers = num_layer, bidirectional = bidirectional)
        layer.initialize()
    elif model_ind == 'GRU':
        layer = rnn.GRU(hidden_size = num_hiddens, num_layers = num_layer, bidirectional = bidirectional)
        layer.initialize()
    elif model_ind == 'LSTM':
        layer = rnn.LSTM(hidden_size = num_hiddens, num_layers = num_layer, bidirectional = bidirectional)
        layer.initialize()
    else:
        raise ValueError('Model unspecified')
    model = RNNModel(layer, vocab_size)
    
    
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0, 'wd': 0})
    for epoch in range(num_epochs):
        loss_sum, num_sample, start = 0.0, 0, time.time()  # we will also record the training time for each epoch
        data_iter = data_iteror(
            corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            "Becasue we are using random iter, os we need to initialize the model before each batch"
            state = model.begin_state(batch_size=batch_size, ctx=ctx)
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(output, y).mean()
            l.backward()
            "gradent clipping to avoid gradient vanishing (and exploding)"
            params = [p.data() for p in model.collect_params().values()]
            gradient_clipping(params, clipping_theta, ctx)
            "gradent descend"
            trainer.step(1)
            loss_sum = loss_sum + l.asscalar() * y.size
            num_sample = num_sample + y.size

        if (epoch + 1) % pred_period == 0:
            try:
              perplexity = math.exp(loss_sum / num_sample)
            except OverflowError:
              perplexity = float('inf')
            
            print('epoch %d, perplexity %f, loss %f, time %.2f sec' % (
                epoch + 1, perplexity, loss_sum / num_sample, time.time() - start))
            print('Chinese Tang Poetry Generator')
            print(f'作者：{model_ind} \n')
            print('Copy Right: David \n') 
            
            "first sentence"
            poe = predict(
                prefix[0], 5, model, vocab_size, ctx, idx_to_char,
                char_to_idx)
            print(poe[0:5]+'，')
            try:
                pre = prefix[1]
                poe = predict(
                    pre, 5, model, vocab_size, ctx, idx_to_char,
                    char_to_idx)
                print(poe[0:5]+'。')
            except:
                poe = predict(
                    poe[-1], 6, model, vocab_size, ctx, idx_to_char,
                    char_to_idx)                
                print(poe[1:6]+'。')            
            
            "second sentence"
            try:
                pre = prefix[2]
                poe = predict(
                    pre, 5, model, vocab_size, ctx, idx_to_char,
                    char_to_idx)
                print(poe[0:5]+'，')
            except:
                poe = predict(
                    poe[-1], 6, model, vocab_size, ctx, idx_to_char,
                    char_to_idx)                
                print(poe[1:6]+'，')                
            
            try:
                pre = prefix[3]
                poe = predict(
                    pre, 5, model, vocab_size, ctx, idx_to_char,
                    char_to_idx)
                print(poe[0:5]+'。')
            except:
                poe = predict(
                    poe[-1], 6, model, vocab_size, ctx, idx_to_char,
                    char_to_idx)                
                print(poe[1:6]+'。')                             
            
            "third sentence"
            try:
                pre = prefix[4]
                poe = predict(
                    pre, 5, model, vocab_size, ctx, idx_to_char,
                    char_to_idx)
                print(poe[0:5]+'，')
            except:
                poe = predict(
                    poe[-1], 6, model, vocab_size, ctx, idx_to_char,
                    char_to_idx)                
                print(poe[1:6]+'，')                             
            
            try:
                pre = prefix[5]
                poe = predict(
                    pre, 5, model, vocab_size, ctx, idx_to_char,
                    char_to_idx)
                print(poe[0:5]+'。')
            except:
                poe = predict(
                    poe[-1], 6, model, vocab_size, ctx, idx_to_char,
                    char_to_idx)                
                print(poe[1:6]+'。')              

            "fourth sentence"
            try:
                pre = prefix[6]
                poe = predict(
                    pre, 5, model, vocab_size, ctx, idx_to_char,
                    char_to_idx)
                print(poe[0:5]+'，')
            except:
                poe = predict(
                    poe[-1], 6, model, vocab_size, ctx, idx_to_char,
                    char_to_idx)                
                print(poe[1:6]+'，')                              
            
            try:
                pre = prefix[7]
                poe = predict(
                    pre, 5, model, vocab_size, ctx, idx_to_char,
                    char_to_idx)
                print(poe[0:5]+'。')
            except:
                poe = predict(
                    poe[-1], 6, model, vocab_size, ctx, idx_to_char,
                    char_to_idx)                
                print(poe[1:6]+'。') 
            print('\n')            
            
            model.save_parameters(f'./data/params_{model_ind}_{epoch}')

def predict_new(model_ind, model_directory, num_hiddens, num_layer, bidirectional, vocab_size, 
                idx_to_char, char_to_idx, prefix, ctx):
    """
    model_ind: what type of model, e.g.: "RNN", "LSTM", "GRU"
    model_directory: where the parameters are saved
    num_hiddens: how mamy hidden neurons in each layer
    num_layer: number of RNN/LSTM/GRU layer
    bidirectional: whether each layer is bidirectional
    vocal_size: how many unique character in the train string
    idx_to_char: idx_to_char => list[char]
    char_to_idx => dict[char] = number
    prefix: the prefix for the poetry generator
    ctx: train on cpu or gpu
    
    """

    if model_ind == 'RNN':
        layer = rnn.RNN(hidden_size = num_hiddens, num_layers = num_layer, bidirectional = bidirectional)
        layer.initialize()
    elif model_ind == 'GRU':
        layer = rnn.GRU(hidden_size = num_hiddens, num_layers = num_layer, bidirectional = bidirectional)
        layer.initialize()
    elif model_ind == 'LSTM':
        layer = rnn.LSTM(hidden_size = num_hiddens, num_layers = num_layer, bidirectional = bidirectional)
        layer.initialize()
    else:
        raise ValueError('Model unspecified')
    model = RNNModel(layer, vocab_size)
    model.load_parameters(model_directory)
    
    # start predicting
    print('Chinese Tang Poetry Generator')
    print(f'作者：{model_ind} \n')
    
    "first sentence"
    poe = predict(
        prefix[0], 5, model, vocab_size, ctx, idx_to_char,
        char_to_idx)
    print(poe[0:5]+'，')
    try:
        pre = prefix[1]
        poe = predict(
            pre, 5, model, vocab_size, ctx, idx_to_char,
            char_to_idx)
        print(poe[0:5]+'。')
    except:
        poe = predict(
            poe[-1], 6, model, vocab_size, ctx, idx_to_char,
            char_to_idx)                
        print(poe[1:6]+'。')            
    
    "second sentence"
    try:
        pre = prefix[2]
        poe = predict(
            pre, 5, model, vocab_size, ctx, idx_to_char,
            char_to_idx)
        print(poe[0:5]+'，')
    except:
        poe = predict(
            poe[-1], 6, model, vocab_size, ctx, idx_to_char,
            char_to_idx)                
        print(poe[1:6]+'，')                
    
    try:
        pre = prefix[3]
        poe = predict(
            pre, 5, model, vocab_size, ctx, idx_to_char,
            char_to_idx)
        print(poe[0:5]+'。')
    except:
        poe = predict(
            poe[-1], 6, model, vocab_size, ctx, idx_to_char,
            char_to_idx)                
        print(poe[1:6]+'。')                             
    
    "third sentence"
    try:
        pre = prefix[4]
        poe = predict(
            pre, 5, model, vocab_size, ctx, idx_to_char,
            char_to_idx)
        print(poe[0:5]+'，')
    except:
        poe = predict(
            poe[-1], 6, model, vocab_size, ctx, idx_to_char,
            char_to_idx)                
        print(poe[1:6]+'，')                             
    
    try:
        pre = prefix[5]
        poe = predict(
            pre, 5, model, vocab_size, ctx, idx_to_char,
            char_to_idx)
        print(poe[0:5]+'。')
    except:
        poe = predict(
            poe[-1], 6, model, vocab_size, ctx, idx_to_char,
            char_to_idx)                
        print(poe[1:6]+'。')              

    "fourth sentence"
    try:
        pre = prefix[6]
        poe = predict(
            pre, 5, model, vocab_size, ctx, idx_to_char,
            char_to_idx)
        print(poe[0:5]+'，')
    except:
        poe = predict(
            poe[-1], 6, model, vocab_size, ctx, idx_to_char,
            char_to_idx)                
        print(poe[1:6]+'，')                              
    
    try:
        pre = prefix[7]
        poe = predict(
            pre, 5, model, vocab_size, ctx, idx_to_char,
            char_to_idx)
        print(poe[0:5]+'。')
    except:
        poe = predict(
            poe[-1], 6, model, vocab_size, ctx, idx_to_char,
            char_to_idx)                
        print(poe[1:6]+'。') 
    print('\n')            



