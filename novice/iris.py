# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf

def main(args):
    sess = tf.Session()

    iris = datasets.load_iris()

    # print(iris.feature_names)
    # print(iris.target_names)
    # print(iris.DESCR) # 
    # print(iris.data) # 入力データが入ってる
    # print(iris.target) # 教師データが入ってる

    x_vals = np.array([[x[2], x[3]] for x in iris.data])  # 後ろの２つの特徴量だけつかう
    y_vals = np.array([1.0 if x == 0 else 0.0 for x in iris.target])  # 0のクラスかどうかの２値分類
    y_vals = np.transpose([y_vals]) # shapeを整える
    print(x_vals.shape)
    print(y_vals.shape)

    x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    a = tf.Variable(tf.random_normal(shape=[2, 1]))
    b = tf.Variable(tf.random_normal(shape=[1]))

    output = tf.matmul(x_data, a) + b

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y_data)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)  # 最適化アルゴリズムの決定
    train_step = optimizer.minimize(loss)  # 最適化対象の計算ノード？を設定
    
    sess.run(tf.global_variables_initializer())  # 変数の初期化

    batch_size = 20

    for i in range(10000):
        # トレーニング用データをランダムに抽出
        rand_index = np.random.choice(x_vals.shape[0], batch_size)
        rand_x = x_vals[rand_index]
        rand_y = y_vals[rand_index]

        # print(rand_x.shape)
        # print(rand_y.shape)

        sess.run(train_step, feed_dict={x_data: rand_x, y_data: rand_y})

        if (i + 1) % 2000 == 0:
            print("step={}, a={}, b={}, loss={}".format(
                i + 1, sess.run(a), sess.run(b),
                sess.run(tf.reduce_mean(loss), feed_dict={x_data: x_vals, y_data: y_vals})))


def __entry_point():
    import argparse
    parser = argparse.ArgumentParser(
        description=u'',  # プログラムの説明
    )
    parser.add_argument("args", nargs="*")
    main(parser.parse_args().args)


if __name__ == '__main__':
    __entry_point()
