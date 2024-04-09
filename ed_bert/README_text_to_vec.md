# スパンをBERTでベクトル化


## 適当にPythonの仮想環境構築
```
conda create -n geocode python=3.9
conda activate geocode
```

## 必要なライブラリをインストール
```
pip install -r requirements.txt
```

## ソースコードをダウンロード
```
git clone https://github.com/naist-nlp/geo_data.git
```

## ディレクトリ `geocoding` に移動
```
cd geocoding

ls
data    requirements.txt	src
```

## BERTなどのモデルを保存するディレクトリ `model` を作成
```
mkdir model

ls
data    model			requirements.txt	src
```

## BERTを実行するのに必要なファイルをダウンロード
- ディレクトリ `model/cl-tohoku` を作成
```
cd model
mkdir cl-tohoku
cd cl-tohoku
```
作成した `cl-tohoku` の下にBERTのモデルや設定ファイルをダウンロードしてくる

- `bert-base-japanese-whole-word-masking` のための設定
```
mkdir bert-base-japanese-whole-word-masking
cd bert-base-japanese-whole-word-masking
wget https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/vocab.txt
wget https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/config.json
wget https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/tokenizer_config.json
wget https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/pytorch_model.bin
```

- (Optional) `bert-base-japanese-v3` のための設定
```
mkdir bert-base-japanese-v3
cd  bert-base-japanese-v3
wget https://huggingface.co/cl-tohoku/bert-base-japanese-v3/resolve/main/vocab.txt
wget https://huggingface.co/cl-tohoku/bert-base-japanese-v3/resolve/main/config.json
wget https://huggingface.co/cl-tohoku/bert-base-japanese-v3/resolve/main/tokenizer_config.json
wget https://huggingface.co/cl-tohoku/bert-base-japanese-v3/resolve/main/pytorch_model.bin
```

## スパンをサブワード化してベクトル化

### 文脈なしで当該スパンのみをベクトル化
- 入力データの形式は1行1スパン（`data/sample_0201_no-context.txt`）
```
東京駅
東京都市
京都府
```

- 以下のコマンドを実行
```
python src/convert_text_to_vectors.py -i data/sample_0201_no-context.txt
```

- 3つのファイルが生成される
    - `data/sample_0201_no-context.subwords.bert-base-japanese-whole-word-masking.jsonl`：サブワード
    - `data/sample_0201_no-context.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5`：ベクトル
    - `data/sample_0201_no-context.subwords.bert-base-japanese-whole-word-masking.vecs.knn.txt`：試しに同一データ内でknnをやった結果

### 文脈を考慮して当該スパンをベクトル化
- 文脈を考慮する場合は入力データの形式は以下のようなもの（`data/sample_0201_context.txt`）
```
[スパン文字列]\t[先頭文字オフセット] [末尾文字オフセット]\t[文脈]
東京駅	4 7	これから東京駅から別府駅に向かいます。
東京都市	0 4	東京都市は架空の場所なのでご注意を！
京都府	8 11	このあと東京から京都府の駅に向かいます。
```

- 以下のコマンドを実行（引数に「-c」を付ける）
```
python src/convert_text_to_vectors.py -i data/sample_0201_context.txt -c
```

- 3つのファイルが生成される
    - `data/sample_0201_context.subwords.bert-base-japanese-whole-word-masking.jsonl`：サブワード
    - `data/sample_0201_context.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5`：ベクトル
    - `data/sample_0201_context.subwords.bert-base-japanese-whole-word-masking.vecs.knn.txt`：試しに同一データ内でknnをやった結果

- 生成された`*.knn.txt`を比べてみる
```
less data/sample_0201_no-context.subwords.bert-base-japanese-whole-word-masking.vecs.knn.txt

- Query: 東京駅
-- 1 東京駅 1.0000001192092896
-- 2 東京都市 0.8784476518630981
-- 3 京都府 0.7591466903686523

- Query: 東京都市
-- 1 東京都市 1.0
-- 2 東京駅 0.8784476518630981
-- 3 京都府 0.787593424320221

- Query: 京都府
-- 1 京都府 1.0
-- 2 東京都市 0.787593424320221
-- 3 東京駅 0.7591466903686523
```

```
less data/sample_0201_context.subwords.bert-base-japanese-whole-word-masking.vecs.knn.txt


- Query: 東京駅
-- 1 東京駅 0.9999998807907104
-- 2 東京都市 0.780155599117279
-- 3 京都府 0.7074990272521973

- Query: 東京都市
-- 1 東京都市 1.0
-- 2 東京駅 0.780155599117279
-- 3 京都府 0.7021964192390442

- Query: 京都府
-- 1 京都府 1.0
-- 2 東京駅 0.7074990272521973
-- 3 東京都市 0.7021964192390442
```

「京都府」をクエリとしたときの結果が違う。
文脈なしだと2位が「東京都市」（文字列的により似てるからだと思われる）。
一方、文脈ありだと2位が「東京駅」（文脈的により似てるからだと思われる）。
