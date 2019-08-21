import os
import mxnet as mx
from modeling import load_data, train_and_predict, data_iter


"Create the directory to store the model parameters"
data_dir="./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data('./data/data.txt') # can put in your own directory

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

"User define"
model_ind, num_hiddens, num_layer, bidirectional = "LSTM", 360, 1, False
ctx, lr, num_epochs, data_iteror = mx.gpu(), 1e2, 1500, data_iter # you can change the ctx to mx.cpu()
num_steps, batch_size, clipping_theta = 40, 128, 1e-2
pred_period, prefix = 50, '書湖河美人山萬鄉' # you can define your own prefix

train_and_predict(model_ind, num_hiddens, num_layer, bidirectional, ctx, lr, num_epochs, vocab_size, 
                      data_iteror, num_steps, corpus_indices, batch_size, idx_to_char, 
                      char_to_idx, clipping_theta,
                      pred_period, prefix)
