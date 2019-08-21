import mxnet as mx
import pickle
from modeling import load_data, predict_new

with open('./data/char_to_idx.pkl', "rb") as fp:
      char_to_idx = pickle.load(fp) 
with open('./data/idx_to_char.pkl', "rb") as fp:
      idx_to_char = pickle.load(fp) 
with open('./data/vocab_size.pkl', "rb") as fp:
      vocab_size = pickle.load(fp) 

"User define"
model_directory = './data/params_LSTM_1249'
model_ind, num_hiddens, num_layer, bidirectional = "LSTM", 360, 1, False
prefix = '書湖河美人山萬鄉' # you can define your own prefix
ctx = mx.cpu()

predict_new(model_ind, model_directory, num_hiddens, num_layer, bidirectional, vocab_size, 
                idx_to_char, char_to_idx, prefix, ctx)
