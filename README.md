Tang-Poetry-Generator
============

In this project, a multi-layer Char-level (optional bidirectional) RNN, LSTM, GRU framework based on MXNET is built up. This project is inspired by the interactive deep learning course [Dive into Deep Learning](https://d2l.ai/). 
Users can modify the model setting to create their own Tang-Poetry generator. A trained one-layer LSTM model is also provided. Users can apply this trained model to create five-characters eight-lines poems("五言律诗"), a special form of tang-poetry.

Data
------------

The provided dataset came from [chinese-poetry project](https://github.com/chinese-poetry/chinese-poetry), a great database contains almost all the ancient poetries in Chinese. The provided trained LSTM model is based on 14k five-characters eight-lines poems from the database. Users can change the [data_cleaning.py](/data_cleaning.py) file to create your own training data.

Model training
------------

User can modify the [train_model.py](/train_model.py) and run the file to train your own model. The default optimizer is "SGD", users can also change the optimizer to "Adam" or other optimizers supported by MXNET in the [modeling.py](/modeling.py). More specific parameters details are provided in the file. Below is the setting parameters for the trained one-layer LSTM model.

```
model_ind, num_hiddens, num_layer, bidirectional = "LSTM", 360, 1, False
ctx, lr, num_epochs, data_iteror = mx.gpu(), 1e2, 1500, data_iter # you can change the ctx to mx.cpu()
num_steps, batch_size, clipping_theta = 40, 128, 1e-2
pred_period, prefix = 50, '書湖河美人山萬鄉' # you can define your own prefix
```

Model Predict
------------

User can make the trained model into a poetry generator by specifying the model parameters, the directory of the trained model and running the [predict.py](/predict.py) file.

```
One example poems generated by the model:

書信來相問，
湖上雲爲雨。
河漢日駸沒，
美人朝夕望。
人間無別離，
山中有幾處。
萬里長江水，
鄉關無夢歸。

```


