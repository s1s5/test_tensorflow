# coding: utf-8
import numpy as np
from keras.layers import Input, Dense, Dropout, LSTM
from keras.models import Model
import random

def main(args):
    seq_len = 10
    x_vals = []
    y_vals = []
    for i in range(1000):
        l = [0.0] * seq_len
        i0 = random.randint(0, seq_len - 1)
        i1 = random.randint(0, seq_len - 1)
        l[i0] = 1.0
        l[i1] = 1.0
        x_vals.append(np.array(l).reshape((seq_len, 1)))
        y_vals.append(np.array([1.0 if abs(i0 - i1) == 3 else 0.0]))
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    # print(x_vals)
    # print(y_vals)

    inputs = Input(shape=(seq_len, 1, ))
    x = LSTM(seq_len)(inputs)
    x = Dense(seq_len, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[inputs], outputs=[predictions])
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy'])

    print(model.summary())
    
    model.fit(x_vals, y_vals, epochs=2, steps_per_epoch=1000, validation_steps=1, validation_split=0.1)

    for i in x_vals:
        y = model.predict(np.array([i]))
        print(i, '=>', y)


def __entry_point():
    import argparse
    parser = argparse.ArgumentParser(
        description=u'',  # プログラムの説明
    )
    parser.add_argument("args", nargs="*")
    main(parser.parse_args().args)


if __name__ == '__main__':
    __entry_point()
