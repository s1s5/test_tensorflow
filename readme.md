# workon tensorflow

## install tensorflow
`pip install tensorflow matplotlib sklearn scipy`


## アルゴリズムの一般的な流れ
1. データセットをインポート・生成
2. データを変換・正規化
3. トレーニングセット、テストセット、検証セットに分割
4. アルゴリズムのパラメタ（ハイパーパラメタ）を設定
5. 変数とプレースホルダを設定
6. モデル構造を定義（計算グラフ）
7. 損失関数を設定
8. モデルの初期化とトレーニング
9. モデルを評価
10. ハイパーパラメタおチューニング
11. デプロイと新しい成果指標の予測

``` python
import numpy as np
import tensorflow as tf

sess = tf.Session()

# 3. トレーニングセットの用意
x_vals = np.random.normal(1, 0.1, 100)  # 入力データの作成
y_vals = np.repeat(10.0, 100)  # 入力データの作成（教師データ）

# 5. 変数とプレースホルダを設定
x_data = tf.placeholder(shape=[1], dtype=tf.float32) # プレースホルダ
y_data = tf.placeholder(shape=[1], dtype=tf.float32) # プレースホルダ

A = tf.Variable(tf.random_normal(shape=[1]))  # 変数

# 6. モデル構造を定義（計算グラフ）
my_output = x_data * A  # モデル構造の定義計算グラフ作成

# 7. 損失関数を設定
loss = (my_output - y_data) ** 2  # 損失関数の作成

# 8. モデルの初期化とトレーニング
sess.run(tf.global_variables_initializer())  # 変数の初期化

# 4. アルゴリズムのパラメタ（ハイパーパラメタ）を設定
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
```

## テンソルとは？
TensorFlowが計算グラフを操作するために使用する第一のデータ構造である。以下はtensorの例
- zero_tsr = tf.zeros([row_dim, col_dim])
- filled_tsr = tf.fill([row_dim, col_dim], 42) == tf.constant(42, shape=[row_dim, col_dim])
- randomunif_tsr = tf.random_uniform([row_dim, col_dim], minval=0, maxval=1)
- shuffled_output = tf.random_shuffle(input_tensor)
- cropped_output = tf.random_crop(input_tensor, crop_size)

## プレースホルダと変数

> 変数
> アルゴリズムのパラメタ
> 損失関数を最小化するために、変数の値をいじくる

``` python
my_var = tf.Variable(tf.zeros([2, 3]))  # 初期値の設定が必要
sess = tf.Session()
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)  # 実際に初期化される
```

> プレースホルダ
> 特定の型と形状を持つデータを供給できるようにするオブジェクト
> 入力とか？
> sessionの"feed_dict"引数からデータを取得する

``` python
sess = tf.Session()
x = tf.placeholder(tf.float32, shape=(2, 2))
y = tf.identity(x)  # 恒等写像とりあえずなんか変換かけたい
sess.run(y, feed_dict={x: np.random.rand(2, 2)})
```

## 組み込み関数
abs, ceil, cos, exp, floor, log, maximum, minimum, negative, pow, round, rsqrt, sign, sin, sqrt, square
digamma, erf, erfc, igamma, igammac, lbeta, lgamma, squared_difference
tensorflow.nn : relu, relu6, sigmoid, tanh, softsign, softplus

## 損失関数
- L2
- L1
- Pseudo-Huber
- ヒンジ
- 交差エントロピー（シグモイド、重み付き、ソフトマックス等がある）

## 最適化アルゴリズム
- GradientDecent
- Momentum
- Adagrad
- Adadelta


# データセット
1. Iris
2. Low Birthweight 
3. Boston Housing
4. MNIST
5. SMS Spam Collection
6. Movie Review
7. CIFAR-10
8. Snakespeare
9. 英語/ドイツ語例文翻訳データ・セット

## scikit-learn

``` python
data = datasets.load_iris()

print(data.DESCR) # データ・セットに関する説明が入ってる

print(data.feature_names) # 入力データの特徴ベクトルごとの値の説明
print(data.target_names) # クラス名
print(data.data) # 入力データが入ってる
print(data.target) # 教師データが入ってる

```

# SVM系
`1/n Σmax(0, 1 - y_i (A x_i - b)) + α||A||^2`



# NN系
## 畳み込み
`tf.nn.conv2d(x_data, filter, strides, padding='SAME')`
出力は`(W - F + 2P) / S + 1`のサイズになる。Wは入力のサイズ、Fはフィルタのサイズ、Pはゼロパディング、Sはストライド


# Keras
## install Keras
- `pip install Keras`

## examples 
https://keras.io/ja/getting-started/sequential-model-guide/

# sequential model

## 損失関数（model.compileの時の'loss'引数）
https://github.com/keras-team/keras/blob/master/keras/losses.py

mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, 
mean_squared_logarithmic_error, squared_hinge, hinge, categorical_hinge, logcosh
categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy, kullback_leibler_divergence
poisson, cosine_proximity

``` python
def mean_squared_error(y_true, y_pred)
    return K.mean(K.square(y_pred - y_true), axis=-1)
```
こんな感じで定義されているっぽい

## fit

- verbose: 
- callbacks: keras.callbacks.Callback
- validation_split: 0から1までの浮動小数点数． 訓練データの中で検証データとして使う割合
- validation_data: 各エポックの損失関数や評価関数で用いられるタプル。訓練には使われない

Historyオブジェクトを返す

## evaluate
## predict
## get_layer

## save

    再構築可能なモデルの構造
    モデルの重み
    学習時の設定 (loss，optimizer)
    optimizerの状態．これにより，学習を終えた時点から正確に学習を再開できます
    keras.models.load_model(filepath)によりモデルを再インスタンス化できます．


# functional API
functional APIは，複数の出力があるモデルや有向非巡回グラフ，共有レイヤーを持ったモデルなどの複雑なモデルを定義するためのインターフェースです．
functional APIは複数の入出力を持ったモデルに最適です

## 中間層の出力

``` python
from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]
```
