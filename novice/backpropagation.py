# coding: utf-8
import numpy as np
import tensorflow as tf

def main(args):
    sess = tf.Session()

    x_vals = np.random.normal(1, 0.1, 100)  # 入力データの作成
    y_vals = np.repeat(10.0, 100)  # 入力データの作成（教師データ）
    x_data = tf.placeholder(shape=[1], dtype=tf.float32) # プレースホルダ
    y_data = tf.placeholder(shape=[1], dtype=tf.float32) # プレースホルダ

    A = tf.Variable(tf.random_normal(shape=[1]))  # 変数

    my_output = x_data * A  # モデル構造の定義計算グラフ作成

    loss = (my_output - y_data) ** 2  # 損失関数の作成

    sess.run(tf.global_variables_initializer())  # 変数の初期化
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)  # 最適化アルゴリズムの決定
    train_step = optimizer.minimize(loss)  # 最適化対象の計算ノード？を設定

    for i in range(100):
        # トレーニング用データをランダムに抽出
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]

        sess.run(train_step, feed_dict={x_data: rand_x, y_data: rand_y})

        if (i + 1) % 25 == 0:
            print("step : ", i + 1, ", A = ", sess.run(A),
                  ", loss=", sess.run(loss, feed_dict={x_data: rand_x, y_data: rand_y}))


def __entry_point():
    import argparse
    parser = argparse.ArgumentParser(
        description=u'',  # プログラムの説明
    )
    parser.add_argument("args", nargs="*")
    main(parser.parse_args().args)


if __name__ == '__main__':
    __entry_point()
