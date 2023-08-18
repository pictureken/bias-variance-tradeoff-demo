# 偏りと分散のトレードオフ

## 実行方法
### 多項式回帰
```bash
$ python model_wise.py
```

### 3層のニューラルネットワーク
```bash
$ model_wise_nn.py
```

### 出力ファイル
outputsファイルに実行結果が格納されます．ディレクトリ構造は以下のようになります．model_wise.pyはlinear-reg，model_wise_nn.pyはnnに出力されます．
```
.
└── outputs
    ├── linear-reg
        └──出力結果
    └── nn
        └──出力結果
```