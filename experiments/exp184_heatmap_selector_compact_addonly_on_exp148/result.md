# exp184_heatmap_selector_compact_addonly_on_exp148 結果

## 状態

`completed_train_side_rejected_no_submit`。

Kaggle CPU split train の `train_lgb0` / `train_lgb1` / `train_lgb2` version 1 はすべて完了。OOF が exp148 anchor より悪化したため、inference port と submit は行わない。

## 評価設計

- variant: `heatmap_selector_compact_addonly`
- parent/control retraining: なし。exp148 historical `lgb_mean` 8.50128118189582 / Public LB 7.960 を比較基準にする。
- runtime: CPU (`cpu_deterministic_threads8`, Kaggle `enable_gpu=false`)。
- train split: `train_lgb0` / `train_lgb1` / `train_lgb2`。各 notebook は `selected_lgb_config_indices=[0|1|2]` で 5 boosters だけを学習する。
- 追加 feature: exp184 selected path と exp182 heatmap validation prediction から作る compact confidence block。
- 禁止: exp184 selected TVT の direct replacement、blend、postprocess、hard gate、submit。

## 採否ゲート

global OOF が小幅改善しても、sparse heatmap distance bucket、worst-well、near `000_050`、`1000_plus`、exp115 hidden-like subgroup、raw-test/current-test feature generation parity が弱い場合は diagnostic として閉じる。提出候補化する場合は同じ実験内で inference port と schema parity を追加する。

split train の各 notebook 内 `lgb_mean` は選択済み 1 config と同値。3 config 横断の `lgb_mean` は、Kaggle output 3本の OOF prediction を取得し、chunked streaming で別途結合して計算した。

## 結果

| model | RMSE TVT | RMSE target |
| --- | ---: | ---: |
| lgb0 | 8.710685277 | 8.710685059 |
| lgb1 | 8.639432353 | 8.639432520 |
| lgb2 | 8.611075285 | 8.611075086 |
| cross-split lgb_mean | 8.604130846 | 8.604130684 |

- rows: 3,783,989
- wells: 773
- features: 322
- hmp184 compact features: 28
- feature join coverage: pass
- cross-split output: `artifacts/exp184_heatmap_selector_compact_addonly_on_exp148_split_lgb_mean_summary.json`

比較:

- exp148 GPU historical `lgb_mean`: 8.501281182。exp184 は +0.102849664 悪化。
- exp148 CPU runtime `lgb_mean`: 8.528698114。exp184 は +0.075432732 悪化。
- exp188 add-only `lgb_mean`: 8.539573790。exp184 はこれも下回る。

## Caveat

local helper smoke では exp148 OOF 差分込みで 31 features を生成したが、Kaggle split train の input source に exp148 train output を入れていなかったため、optional exp148 OOF delta features は unavailable となり、実行は 28 hmp184 features で完了した。

厳密な 31-feature rerun を行う場合は split train kernel sources に exp148 train output を追加する必要がある。ただし今回の 28-feature CV は exp148 anchor から大きく悪化しているため、現時点では再実行・inference・submit は採用しない。

## 判定

train-side negative。heatmap selector compact add-only は exp148 ML anchor の改善候補として採用しない。
