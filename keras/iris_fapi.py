# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers

import keras.backend as K

def sqrt_regularizer(x):
    return K.sum(1.0e-3 * K.sqrt(K.abs(x)))
    # return K.sum(1.0e-3 * (K.abs(x) ** 0.01))


def main(args):
    iris = datasets.load_iris()
    
    x_vals = np.array([[x[2], x[3]] for x in iris.data])  # 後ろの２つの特徴量だけつかう
    # x_vals = np.array([[x[0], x[1]] for x in iris.data])  # 後ろの２つの特徴量だけつかう
    y_vals = np.array([1.0 if x == 0 else -1.0 for x in iris.target])  # 0のクラスかどうかの２値分類
    # y_vals = np.array([1.0 if x == 0 else -1.0 for x in iris.target])  # 0のクラスかどうかの２値分類
    y_vals = np.transpose([y_vals]) # shapeを整える

    inputs = Input(shape=(2, ))
    # x = Dense(4, activation='relu', kernel_regularizer=regularizers.l1(1.0e-3))(inputs)
    # x = Dense(4, activation='relu', kernel_regularizer=regularizers.l1(1.0e-3))(x)
    # predictions = Dense(1, kernel_regularizer=regularizers.l1(1.0e-3))(x)
    x = Dense(4, activation='relu', kernel_regularizer=sqrt_regularizer)(inputs)
    x = Dense(4, activation='relu', kernel_regularizer=sqrt_regularizer)(x)
    # predictions = Dense(1, kernel_regularizer=sqrt_regularizer)(x)
    predictions = Dense(1, activation='tanh', kernel_regularizer=sqrt_regularizer)(x)
    # x = Dense(4, activation='relu')(inputs)
    # # x = Dropout(0.5)(x)
    # x = Dense(4, activation='relu')(x)
    # # x = Dropout(0.5)(x)
    # predictions = Dense(1, activation='tanh')(x)
    
    model = Model(inputs=[inputs], outputs=[predictions])
    model.compile(
        # optimizer='rmsprop',
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy'])

    print(model.summary())

    model.fit(x_vals, y_vals, epochs=5, steps_per_epoch=1000)

    print(len(model.layers))
    for index, layer in enumerate(model.layers):
        print(index, layer.get_weights())
        if layer.get_weights():
            print(K.eval(1.0e3 * sqrt_regularizer(layer.get_weights()[0])))

    print("evaluate:", model.evaluate(x_vals, y_vals))

    x = [x[0] for index, x in enumerate(x_vals) if y_vals[index] < 0]
    y = [x[1] for index, x in enumerate(x_vals) if y_vals[index] < 0]
    plt.scatter(x, y, marker='o')
    x = [x[0] for index, x in enumerate(x_vals) if y_vals[index] > 0]
    y = [x[1] for index, x in enumerate(x_vals) if y_vals[index] > 0]
    plt.scatter(x, y, marker='x')

    minx = min(x[0] for x in x_vals) - 1
    maxx = max(x[0] for x in x_vals) + 1
    miny = min(x[1] for x in x_vals) - 1
    maxy = max(x[1] for x in x_vals) + 1

    al = []
    bl = []
    import random
    for i in range(1000):
        xv = random.random() * (maxx - minx) + minx
        yv = random.random() * (maxy - miny) + miny
        v = model.predict(np.array([[xv, yv]]))
        # print(xv, yv, v)
        if v < 0.0:
            al.append([xv, yv])
        else:
            bl.append([xv, yv])

    x = [x[0] for x in al]
    y = [x[1] for x in al]
    plt.scatter(x, y, marker='.')

    x = [x[0] for x in bl]
    y = [x[1] for x in bl]
    plt.scatter(x, y, marker='*')
    
        
    plt.show()


def __entry_point():
    import argparse
    parser = argparse.ArgumentParser(
        description=u'',  # プログラムの説明
    )
    parser.add_argument("args", nargs="*")
    main(parser.parse_args().args)


if __name__ == '__main__':
    __entry_point()
