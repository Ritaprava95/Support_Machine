import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Activation, Concatenate
import matplotlib.pyplot as plt

def use_gpu():
    # Creates a graph.
    with tf.device('/gpu:0'):
      a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
      b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(c))
    print("Using The GPU")

def prepare_data():
    data = pd.read_csv("dataset.csv")
    X = data.loc[:,'sub']
    y = data.loc[:,'severity']
    train_X, train_y, test_X, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    
    t = Tokenizer()
    t.fit_on_texts(train_X)
    vocab_size = len(t.word_index)+1
    
    encoded_train_X = t.text_to_sequences(train_X)
    encoded_test_X = t.text_to_sequence(test_X)
    
    max_len = 20
    
    padded_train_X = pad_sequences(encoded_train_X, maxlen=max_len, padding='post')
    padded_test_X = pad_sequences(encoded_test_X, maxlen=max_len, padding='post') 
    
    embedding_index = dict()
    f = open('all_embeddings.txt')#custom embedding matrix 
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embedding_index[word] = coefs
    f.close()
    
    embedding_matrix = np.zeros([vocab_size, max_len])
    for word, i in t.word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return vocab_size, padded_train_X, train_y, padded_test_X, test_y, embedding_matrix
        
        
def model(vocab_size, padded_train_X, train_y, padded_test_X, test_y, embedding_matrix):
    use_gpu()
    sentences = Input((20,), dtype='int32')
    embeddings = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=20, trainable=False)
    
    X1 = LSTM(32, return_sequences=True)(embeddings)
    X1 = Dropout(0.2)(X1)
    X1 = LSTM(32, return_sequences=False)(X1)
    X1 = Dropout(0.2)(X1)
    
    X2 = LSTM(32, return_sequences=False)(embeddings)
    X2 = Dropout(0.2)
    
    X = Concatenate(axis=-1)([X1, X2])
    X = Dense(64)(X)
    X = Activation('relu')(X)
    X = Dense(1)()
    X = Activation('sigmoid')(X)
    
    model = Model(inputs = sentences, outputs = X)
    
    print(model.summary())
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    train = model.fit(padded_train_X, train_y, validation_split=0.2, epochs=50, verbose=1, batch_size=32)
    loss, accuracy = model.evaluate(padded_test_X,test_y, verbose=1)
    print('Model1 Accuracy: ' ,(accuracy*100))
    
    
    print(train.history.keys())
    plt.plot(train.history(['acc']))
    plt.plot(train.history(['val_acc']))
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

vocab_size, padded_train_X, train_y, padded_test_X, test_y, embedding_matrix = prepare_data()
model(vocab_size, padded_train_X, train_y, padded_test_X, test_y, embedding_matrix)
