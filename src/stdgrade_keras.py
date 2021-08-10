'''
keras module
'''
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def encode():
    '''
    encode data
    '''
    pd_data = pd.read_csv('student-mat.csv')
    data = pd_data.drop(['G1', 'G2', 'G3'], axis=1)
    data_cat = data[['school', 'sex', 'address', 'famsize', 'Pstatus',
                     'Mjob', 'Fjob', 'reason', 'guardian',
                     'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                     'higher', 'internet', 'romantic']]
    data_del = data.drop(['school', 'sex', 'address', 'famsize', 'Pstatus',
                          'Mjob', 'Fjob', 'reason', 'guardian',
                          'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                          'higher', 'internet', 'romantic'], axis=1)
    encoder = OneHotEncoder(sparse=False)
    data_cat = encoder.fit_transform(data_cat)
    data = np.concatenate((data_del, data_cat), axis=1)
    return data


def r2(y_true, y_pred):
    '''
    self-definded r2 function
    '''
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b / e
    return f


class KerasModule:
    '''
    initialize Keras Model
    '''
    def __init__(self):
        pd_data = pd.read_csv('student-mat.csv')
        self.pd_x = encode()
        temp = pd_data[["G1", "G2", "G3"]]
        self.pd_y = temp.mean(axis=1)
        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(self.pd_x, self.pd_y, test_size=0.2, random_state=532)
        self.pred_test_y = 0
        self.n_hidden_1 = 64
        self.n_hidden_2 = 64
        self.n_input = 56
        self.n_classes = 1
        self.training_epochs = 200
        self.batch_size = 10

    def normalize(self):
        '''
        normalize data
        '''
        sc = StandardScaler()
        self.train_x = sc.fit_transform(self.train_x)
        self.test_x = sc.fit_transform(self.test_x)
        print(self.train_x[0])
        print(self.test_x[0])

    def built_network(self):
        '''
        built model
        '''
        model = Sequential()
        model.add(Dense(self.n_hidden_1, activation='relu', input_dim=self.n_input))
        model.add(Dense(self.n_hidden_2, activation='relu'))
        model.add(Dense(self.n_classes))
        model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', r2])
        history = model.fit(self.train_x, self.train_y, batch_size=self.batch_size, epochs=self.training_epochs)
        self.pred_test_y = model.predict(self.test_x)
        print(self.pred_test_y)

    def plot(self):
        '''
        visualize
        '''
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(8, 4), dpi=80)
        plt.plot(range(len(self.test_y)), self.test_y, ls='-.', lw=2, c='r', label='true')
        plt.plot(range(len(self.pred_test_y)), self.pred_test_y, ls='-', lw=2, c='b', label='pred')
        plt.grid(alpha=0.4, linestyle=':')
        plt.legend()
        plt.xlabel('factor')
        plt.ylabel('grade')
        plt.show()
