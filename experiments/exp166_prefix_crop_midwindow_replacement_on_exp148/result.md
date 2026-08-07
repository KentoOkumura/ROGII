# exp166 結果

## 結論

案2の `tail500` / `tail1000` mid-window replacement-only は exp148 を改善しなかったため、不採用。提出しない。

最良は `prefix_crop_tail500_replacement` の `lgb0` で CV 8.566426970340796。exp148 `lgb_mean` CV 8.50128118189582 から +0.065145788444976 悪化した。`tail1000` の最良は `lgb2` CV 8.574216682757848 で、こちらも +0.072935500862028 悪化した。

## 実行内容

- Runtime: CPU `cpu_deterministic_threads8`
- 構成: 2段階。prefix crop feature cache を作成し、split train notebook `lgb0` / `lgb1` / `lgb2` が cache を読む。
- feature cache: `kentookumura/exp166-prefix-crop-midwindow-exp148-features` v1
- train: `kentookumura/exp166-prefix-crop-midwindow-exp148-train-lgb0/lgb1/lgb2` v2
- active variants:
  - `prefix_crop_tail500_replacement`
  - `prefix_crop_tail1000_replacement`
- control 再学習: なし

## 結果

| variant | lgb0 | lgb1 | lgb2 | best |
| --- | ---: | ---: | ---: | ---: |
| `prefix_crop_tail500_replacement` | 8.566426970 | 8.595296705 | 8.638434981 | 8.566426970 |
| `prefix_crop_tail1000_replacement` | 8.615045273 | 8.589982015 | 8.574216683 | 8.574216683 |

比較:

- exp148 `lgb_mean`: 8.50128118189582
- exp161 last50 add-only best single config: 8.56472499591314
- exp166 best: 8.566426970340796

exp166 best は exp148 に対して +0.065145788444976 悪く、exp161 last50 add-only best single に対しても +0.001701974427656 悪い。

## 失敗と修正

split train v1 は 3本とも `prefix_crop_join_start` 直後に `DeadKernelError`。LightGBM fold 開始前だったため、96列の prefix crop cache を全 variant 分まとめて full frame に concat したメモリピークが原因と判断した。

v2 では cache schema だけを先に読み、variant ごとに必要な 48列だけを `usecols` で読み込む構造に変更した。`tail500` を join して学習、解放してから `tail1000` を join するため、同じ lgb0/lgb1/lgb2 の3分割のまま完走した。

## 考察

tail500/tail1000 に広げても、full-prefix の急降下ノイズを取り除く効果は exp148 の既存特徴を置換する損失を上回らなかった。`slp_all`、`pfx_rmse`、calibration、SC/NCC、multiobs 系をまとめて落としたことで、急降下ノイズだけでなく exp148 が使っていた有効な長距離 prefix 情報も失った可能性が高い。

案1の last50 replacement-only は、局所性をさらに強める isolated test としては残る。ただし exp161 last50 add-only と今回の tail500/tail1000 replacement-only がどちらも exp148 を改善していないため、優先度は高くない。実行する場合は同じ 2段階 cache/train 構成で、置換対象を今回より狭めるか、単一系統の置換 ablation にする方がよい。
