# Geocoding


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
requirements.txt	src
```

## BERTなどのモデルを保存するディレクトリ `model` を作成
```
mkdir model

ls
model			requirements.txt	src
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


## データ準備
### OSMデータベース `20230620_all_extnames.txt` をダウンロード
- 使用するデータを保存するディレクトリ `data` を作成
```
mkdir data

ls
data			model			requirements.txt	src
```

- `20230620_all_extnames.txt` をダウンロード
```
wget xxx
```

### 旅行記データを作成
- `data_split` に移動して旅行記データ `train/dev/test1/test2.jsonl` を作成
```
pwd
/path/to/geocoding

cd ../data_split

python split.py --format json
```

- 作成した `test1.jsonl` と `test2.jsonl` を統合して `geocoding/data` にコピー
```
cat test1.json test2.jsonl > test.jsonl
cp test.jsonl ../geocoding/data
```

### 旅行記データからジオコーディングの対象となる文字列を作成
```
python src/ed_select_entity_names.py

ls data/
20230620_all_extnames.txt	test.jsonl			test.names.longest.txt
```
`test.names.longest.txt`が生成される

### OSMデータベースからジオコーディングの対象となる文字列を作成
- まずは `.jsonl` 形式に変換
```
python src/convert_extnames_txt_to_jsonl.py

ls data/
20230620_all_extnames.jsonl	test.jsonl
20230620_all_extnames.txt	test.names.longest.txt
```
`20230620_all_extnames.jsonl`が生成される

- 次に文字列を作成
```
python src/convert_extnames_jsonl_to_names.py

ls data/
20230620_all_extnames.jsonl	20230620_all_extnames.txt	test.names.longest.txt
20230620_all_extnames.names.txt	test.jsonl
```
`20230620_all_extnames.names.txt`が生成される

- 後ほど使うデータベース内のエントリIDを作成
```
python src/convert_extnames_jsonl_to_ids.py

ls data/
20230620_all_extnames.ids.txt
20230620_all_extnames.jsonl
20230620_all_extnames.txt
test.names.longest.txt
20230620_all_extnames.names.txt
test.jsonl
```
`20230620_all_extnames.ids.txt`が生成される

### 旅行記データの文字列をサブワード化してベクトル化
- CPUで行う場合は、`src/vectorizers.py`の13行目をコメントアウトする
```
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
```

- 以下のコマンドを実行
```
python src/convert_text_to_vectors.py -i data/test.names.longest.txt

ls data/
20230620_all_extnames.ids.txt
20230620_all_extnames.jsonl
20230620_all_extnames.names.txt
20230620_all_extnames.txt
test.jsonl
test.names.longest.subwords.bert-base-japanese-whole-word-masking.jsonl
test.names.longest.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5
test.names.longest.subwords.bert-base-japanese-whole-word-masking.vecs.knn.txt
test.names.longest.txt
```
- 3つのファイルが生成される
    - `test.names.longest.subwords.bert-base-japanese-whole-word-masking.jsonl`：サブワード
    - `test.names.longest.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5`：ベクトル
    - `test.names.longest.subwords.bert-base-japanese-whole-word-masking.vecs.knn.txt`：試しに同一データ内でknnをやった結果

### OSMデータベースの文字列をサブワード化してベクトル化
```
python src/convert_text_to_vectors.py -i data/20230620_all_extnames.names.txt --data_size 1000

ls data/
20230620_all_extnames.ids.txt
20230620_all_extnames.jsonl
20230620_all_extnames.names.subwords.bert-base-japanese-whole-word-masking.jsonl
20230620_all_extnames.names.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5
20230620_all_extnames.names.subwords.bert-base-japanese-whole-word-masking.vecs.knn.txt
20230620_all_extnames.names.txt
20230620_all_extnames.txt
test.jsonl
test.names.longest.subwords.bert-base-japanese-whole-word-masking.jsonl
test.names.longest.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5
test.names.longest.subwords.bert-base-japanese-whole-word-masking.vecs.knn.txt
test.names.longest.txt
```
- 旅行記データと同様に3つのファイルが生成される
    - `20230620_all_extnames.names.subwords.bert-base-japanese-whole-word-masking.jsonl`
    - `20230620_all_extnames.names.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5`
    - `20230620_all_extnames.names.subwords.bert-base-japanese-whole-word-masking.vecs.knn.txt`

- 注意：`--data_size`でベクトル化する事例数を指定できる。ローカルPCで小規模に試したい場合はこの引数で小さい数字を指定する。全事例を使いたい場合はこの引数は使わない


## ジオコーディングする
```
python src/ed_disambiguate_entities.py --input_path data/test.jsonl --input_vec data/test.names.longest.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5 --entry_id data/20230620_all_extnames.ids.txt --entry_vec data/20230620_all_extnames.names.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5 --entry_size 1000

ls data/
20230620_all_extnames.ids.txt
20230620_all_extnames.jsonl
20230620_all_extnames.names.subwords.bert-base-japanese-whole-word-masking.jsonl
20230620_all_extnames.names.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5
20230620_all_extnames.names.subwords.bert-base-japanese-whole-word-masking.vecs.knn.txt
20230620_all_extnames.names.txt
20230620_all_extnames.txt
test.ed_results.json
test.jsonl
test.names.longest.subwords.bert-base-japanese-whole-word-masking.jsonl
test.names.longest.subwords.bert-base-japanese-whole-word-masking.vecs.hdf5
test.names.longest.subwords.bert-base-japanese-whole-word-masking.vecs.knn.txt
test.names.longest.txt
```
`test.ed_results.json`が生成される。このファイルにジオコーディングの検索結果が保存されている