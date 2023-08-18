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
outputsファイルに実行結果が格納されます．ディレクトリ構造は以下のようになります．model_wise.pyはlinear-reg，model_wise_nn.pyはnnに出力されます．また，Fixed Designの場合はTrueフォルダ，Random Designの場合はFalseフォルダに実行結果が格納されます．
```
.
└── outputs
    ├── linear-reg
    │    ├──True
    │    │   └──実行結果
    │    └──False
    │        └──実行結果
    └── nn
         ├──True
         │   └──実行結果
         └──False
             └──実行結果
```