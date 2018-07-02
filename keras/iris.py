# coding: utf-8
import numpy as np

from sklearn import datasets

from keras.models import Sequential
from keras.layers import Dense, Activation

def main(args):
    iris = datasets.load_iris()
    
    x_vals = np.array([[x[2], x[3]] for x in iris.data])  # 後ろの２つの特徴量だけつかう
    y_vals = np.array([1.0 if x == 0 else 0.0 for x in iris.target])  # 0のクラスかどうかの２値分類
    y_vals = np.transpose([y_vals]) # shapeを整える
    print(x_vals.shape)
    print(y_vals.shape)

    model = Sequential([
        # 最初のレイヤでinput_shapeを指定
        Dense(4, input_shape=(2,)),
        Activation('relu'),
        Dense(1),
#        Activation('softmax'),
    ])

    # 3つ指定する必要がある
    model.compile(
        # 最適化アルゴリズム
        optimizer='rmsprop',
        # 損失関数(自分で定義しても入れれる)
        loss='mean_squared_error',
        # 評価関数のリスト(自分で定義しても入れれる)
        metrics=['accuracy'])

    model.fit(x_vals, y_vals, epochs=100, batch_size=32)
    



def __entry_point():
    import argparse
    parser = argparse.ArgumentParser(
        description=u'',  # プログラムの説明
    )
    parser.add_argument("args", nargs="*")
    main(parser.parse_args().args)


if __name__ == '__main__':
    __entry_point()
