# 設計

## アプローチ

既存 OOF prediction を row-level で読み、prediction 内の `target_tvt` / `pred_tvt` から残差と TVT step を計算する。XY 座標は `artifacts/typewell_position_groups/native_overlap_1_well_position_typewell_summary.csv` の `x_mean` / `y_mean` を使う。1 well は 1 つの共通 typewell group に属する前提で、同 summary の `exact_typewell_group` を grouping key として使う。

診断は次の 5 面で行う。

1. well 別の RMSE / MAE / bias / offset 指標。
2. typewell group 別の RMSE / bias / offset well 数 / 残差形状類似度。
3. XY 近傍 k=8 wells の bias 類似度、same-sign rate、残差形状類似度。
4. true TVT の急変 step を global p99.5 で定義し、true step に対する predicted step の追従率を集計。
5. well 全体 offset を `abs(mean residual) >= 10 ft` かつ `abs(mean residual) / RMSE >= 0.70` で抽出し、typewell / XY への偏りを見る。

## 実験範囲

- 対象実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: なし。診断集計だけを追加する。
- 固定する変数: exp148 train v1 OOF prediction、raw train truth、native_overlap_1 typewell group。

## 再現性設計

- seed policy: 乱数なし。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp148 の保存済み OOF prediction を読むだけ。
- 並列処理と乱数の関係: 並列処理なし。
- CPU/GPU runtime と deterministic flags: 集計は CPU pandas/numpy の deterministic 処理。
- train cache / test feature regeneration の SHA 記録方針: 入力 prediction gzip は decompressed SHA、typewell summary と raw metadata は file SHA を記録。
- model manifest / prediction / submission SHA 記録方針: 新規 model / submission はなし。生成した diagnostic CSV / JSON の SHA を summary に記録する。
- Kaggle package bootstrap 確認方針: Kaggle push しない限り対象外。Kaggle 化する場合は train notebook だけを診断 notebook として準備する。

## リスク

- リークリスク: train OOF と train true TVT を使う診断なので、特徴量や提出候補として直接使わない。後続 feature 化する場合は fold-safe に作り直す。
- CV/LB 不一致リスク: OOF 診断は hidden LB を保証しない。typewell / XY 傾向は hidden-like stress で再確認が必要。
- ランタイム/メモリリスク: exp148 prediction は 4 models x 3,783,989 rows なので chunk 読みで `lgb_mean` だけを抽出する。
- 再現性リスク: 入力ファイルの取り違えが主リスク。source path、行数、well 数、SHA を summary に残す。
