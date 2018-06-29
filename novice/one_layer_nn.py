# coding: utf-8
import numpy as np
from sklearn import datasets
import tensorflow as tf


def main(args):
    sess = tf.Session()

    iris = datasets.load_iris()

    x_vals = np.array([[x[0], x[1], x[2]] for x in iris.data])  # 後ろの２つの特徴量だけつかう
    y_vals = np.array([[x[3]] for x in iris.data])

    seed = 2
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # トレーニングデータの抽出
    train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

    x_vals_train = x_vals[train_indices]
    y_vals_train = y_vals[train_indices]

    x_vals_test = x_vals[test_indices]
    y_vals_test = y_vals[test_indices]

    # スケーリングして0-1の範囲に入力を収める
    def normalize_cols(m):
        col_max = m.max(axis=0)
        col_min = m.min(axis=0)
        return np.nan_to_num((m - col_min) / (col_max - col_min))
    x_vals_train = normalize_cols(x_vals_train)
    x_vals_test = normalize_cols(x_vals_test)

    batch_size = 50
    x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    hidden_layer_nodes = 5

    A1 = tf.Variable(tf.random_normal(shape=[3, hidden_layer_nodes]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
    A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))
    b2 = tf.Variable(tf.random_normal(shape=[1]))

    hidden_output = tf.nn.relu(tf.matmul(x_data, A1) + b1)
    final_output = tf.matmul(hidden_output, A2) + b2
    loss = tf.reduce_mean((y_data - final_output)**2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)  # 最適化アルゴリズムの決定
    train_step = optimizer.minimize(loss)  # 最適化対象の計算ノード？を設定
    
    sess.run(tf.global_variables_initializer())  # 変数の初期化
    
    loss_vec = []
    test_loss = []
    for i in range(500):
        sess.run(train_step, feed_dict={x_data: x_vals_train, y_data: y_vals_train})
        # print(sess.run(A2))
        # print(sess.run(b2))
        # print(sess.run(hidden_output, feed_dict={x_data: x_vals_test, y_data: y_vals_test}))
        # print(sess.run(final_output, feed_dict={x_data: x_vals_test, y_data: y_vals_test}))
        temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_data: y_vals_test})
        # break
        loss_vec.append(np.sqrt(temp_loss))
        if (i + 1) % 50 == 0:
            print("{} loss:{}".format(i + 1, temp_loss))
    


def __entry_point():
    import argparse
    parser = argparse.ArgumentParser(
        description=u'',  # プログラムの説明
    )
    parser.add_argument("args", nargs="*")
    main(parser.parse_args().args)


if __name__ == '__main__':
    __entry_point()
