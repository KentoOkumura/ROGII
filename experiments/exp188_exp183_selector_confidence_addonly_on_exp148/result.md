# exp188_exp183_selector_confidence_addonly_on_exp148 結果

## 状態

Kaggle train v3 完了。train-side OOF は exp148 baseline より悪化したため、採用しない。inference port / submit には進めない。

## 設定

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- Route: `ml_model`
- variant: `exp183_selector_confidence_addonly`
- baseline: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- 追加特徴: exp183 OOF best-Viterbi selected path 由来の selected candidate/family、candidate 差分、path jump、segment stability など 32 features
- 学習: GPU、3 LightGBM configs x 5 folds = 15 boosters
- control 再学習: なし

## 結果

- Kaggle kernel: `kentookumura/exp188-exp183-selconf-exp148-train` version 3
- status: `COMPLETE`
- output: `experiments/exp188_exp183_selector_confidence_addonly_on_exp148/kaggle/output/train_v3/`
- rows / wells: 3,783,989 / 773
- features: 326
- exp183 selector features: 32
- feature join coverage: pass、dropped rows 0、dropped wells 0
- elapsed: 14,426.84 sec

| model | pooled RMSE |
| --- | ---: |
| lgb0 | 8.620017124 |
| lgb1 | 8.568069226 |
| lgb2 | 8.576058237 |
| lgb_mean | 8.539573790 |

exp148 `lgb_mean` 8.501281182 から、exp188 `lgb_mean` は +0.038292608 悪化した。

## 解釈

exp183 の selector selected-path / segment stability signal は、exp148 の learned-likelihood LightGBM anchor には add-only で効かなかった。`exp183_segment_len` や `exp183_distance_to_segment_boundary` は feature importance 上位に入ったが、global OOF は exp148 を下回った。

したがって、この add-only 設計は完了/不採用とする。一方、replacement-only は exp148 の既存 `ll_*` block との置換比較として別仮説なので、`exp183_selector_confidence_replacement_only_on_exp148` は backlog に戻して扱う。
