# coding: utf-8
import numpy as np

from sklearn import datasets

from keras.layers import Input, Dense
from keras.models import Model


def main(args):
    iris = datasets.load_iris()
    
    x_vals = np.array([[x[2], x[3]] for x in iris.data])  # 後ろの２つの特徴量だけつかう
    y_vals = np.array([1.0 if x == 0 else 0.0 for x in iris.target])  # 0のクラスかどうかの２値分類
    y_vals = np.transpose([y_vals]) # shapeを整える

    inputs = Input(shape=(2, ))
    x = Dense(4, activation='relu')(inputs)
    x = Dense(4, activation='relu')(x)
    predictions = Dense(1)(x)
    
    model = Model(inputs=[inputs], outputs=[predictions])
    model.compile(
        optimizer='rmsprop',
        loss='mean_squared_error',
        metrics=['accuracy'])

    model.fit(x_vals, y_vals, epochs=10, steps_per_epoch=1000)
    



def __entry_point():
    import argparse
    parser = argparse.ArgumentParser(
        description=u'',  # プログラムの説明
    )
    parser.add_argument("args", nargs="*")
    main(parser.parse_args().args)


if __name__ == '__main__':
    __entry_point()
