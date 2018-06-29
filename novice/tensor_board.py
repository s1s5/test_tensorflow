# coding: utf-8
import tensorflow as tf

def main(args):
    sess = tf.InteractiveSession()

    # TensorBoard情報出力ディレクトリ
    log_dir = '/home/sawai/data/tmp/tensorflow/simple02'

    # 指定したディレクトリがあれば削除し、再作成
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    #add_scopeという名称でグルーピング
    with tf.name_scope('add_scope'):

        # 定数で1 + 2
        x = tf.constant(1, name='x')
        y = tf.constant(2, name='y')
        z = x + y

        # このコマンドでzをグラフ上に出力
        _ = tf.summary.scalar('z', z)

        # 上の結果に掛け算
        with tf.name_scope('multiply_scope'):
            zz = y * z

    # SummaryWriterでグラフを書く(これより後のコマンドはグラフに出力されない)
    summary_writer = tf.summary.FileWriter(log_dir , sess.graph)

    # 実行
    print(sess.run(z))

    # SummaryWriterクローズ
    summary_writer.close()



def __entry_point():
    import argparse
    parser = argparse.ArgumentParser(
        description=u'',  # プログラムの説明
    )
    parser.add_argument("args", nargs="*")
    main(parser.parse_args().args)


if __name__ == '__main__':
    __entry_point()
