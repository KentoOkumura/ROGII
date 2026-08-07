# exp099_pf_multi_observation_likelihood_probe

## 状態

- ルート: pf_beam
- 状態: completed_train_side_audit
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-21
- 親実験: `exp093_pf_candidate_coverage_then_ranker_audit`

## 仮説

exp093 では PF/Beam/likelihood-PF 候補集合に oracle headroom はあるが、target-free rank score が弱かった。既存候補を増やす前に、各候補 TVT が示す prefix TVT 位置の GR と、評価 row 周辺の複数 GR 観測点を比較する target-free likelihood で候補順位を改善できるかを監査する。

## 変更点

- exp072 deterministic full replay train cache を固定入力として読む。
- `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb` を multi-observation likelihood で再採点する。
- 新規診断候補として `multiobs_top1`、softmax weighted candidates、`likpf_multiobs_blend` を作る。
- exp072 と同じ wide feature cache 形式で、multiobs score / MAE / NCC などを下流 ranker 用特徴量として保存する。
- true TVT は candidate RMSE / oracle / rank metrics の scoring にだけ使う。
- PF/Beam 再実行、supervised ranker、inference port、提出は行わない。

## 検証方針

- Fold: train-side pseudo-tail audit の既存 cache に従う
- Group: well
- Stratification: distance / tail rank / eval length / PF seed std / likPF delta / multiobs score
- Leakage Check: multi-observation likelihood は raw horizontal GR、row index、finite prefix TVT_input、既存候補 TVT のみで計算する

## 実行入口

- 学習 notebook: `exp099_pf_multi_observation_likelihood_probe_train.ipynb`
- 推論 notebook: `exp099_pf_multi_observation_likelihood_probe_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp099_pf_multi_observation_likelihood_probe EXTRA_ARGS="--notebook train --run-on-push --strict"`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。
- cache 生成物:
  - `exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz`
  - `exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv`

## 結果

Kaggle train v1 完了。baseline primary oracle は RMSE 7.434030 / within10 0.906525、baseline+multiobs oracle は RMSE 6.897510 / within10 0.922941 で、oracle headroom は増えた。

一方で target-free rank score top1 は RMSE 89.994392 / within10 0.523815 と大きく悪化した。multi-observation likelihood は直接 scorer としては不採用だが、candidate ranker / learned observation likelihood の feature 材料として保持する。

## 所見

### 良かった点

- exp093 の scorer 失敗を、候補追加ではなく target-free likelihood scorer として切り分けた。
- true TVT を scoring のみに閉じ込め、PF/Beam 再実行や inference port を範囲外にした。

### 悪かった点

- v1 output は診断 CSV 中心で、exp072 と同じ wide train feature cache 形式ではなかった。
- そのため、train notebook を更新して cache 生成物を追加した。cache 追加版は Kaggle train v2 で完了し、3,783,989 rows / 40 features の exp072-style wide train cache を保存済み。

### リスク / 注意

- GR の局所一致だけで候補を選ぶと、同じ GR motif の別 TVT 位置へ引っ張られる可能性がある。
- 改善しても直接提出候補ではなく、ranker feature / scorer material として扱う。

## 次

1. `pf_candidate_ranker_or_nway_classifier` で v2 の wide cache を読み、multiobs score / MAE / NCC を特徴として追加する。
2. 候補選択が改善するか確認する。
