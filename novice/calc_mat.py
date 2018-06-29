# coding: utf-8
import numpy as np
import tensorflow as tf

def main(args):
    A = tf.truncated_normal([2, 3])
    B = tf.truncated_normal([3, 2])
    sess = tf.Session()
    print(sess.run(A))
    print(sess.run(A))
    print(sess.run(A ** 2))
    print(sess.run(tf.matmul(B, A)))

    A = np.random.rand(3, 3)
    X = tf.placeholder(tf.float32, [3, 3])
    B = tf.truncated_normal([3, 3])
    print(sess.run(tf.matmul(B, X), feed_dict={X: A}))


def __entry_point():
    import argparse
    parser = argparse.ArgumentParser(
        description=u'',  # プログラムの説明
    )
    parser.add_argument("args", nargs="*")
    main(parser.parse_args().args)


if __name__ == '__main__':
    __entry_point()
