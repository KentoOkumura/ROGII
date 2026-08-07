# exp145_learned_likelihood_rawtest_feature_generator_parity

## 状態

- ルート: ml_model
- 状態: completed_generator_parity_pass_no_submit
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-27
- 親実験: exp144_learned_likelihood_hidden_stress_and_rawtest_parity

## 仮説

exp127 learned likelihood add-only features は hidden-like stress でも改善したが、raw-test/full-train feature regeneration が未実装だった。exp111 の保存済み learned likelihood model を target-free transform として使えば、exp112 と同じ `ml_features` schema を full train と raw test に再生成できる。

## 変更点

- exp111 feature schema と保存済み LightGBM booster を読み、candidate-long likelihood を raw train/test feature frame へ適用する generator を追加した。
- exp112 `ml_features` と同じ列を出す schema parity audit を追加した。
- full train は exp099 wide cache を chunk 読みする。
- raw test は同梱した exp072 replay code から PF/Beam/likelihood-PF features を再生成する。
- この実験では submission を作らない。

## 検証方針

- Fold: なし。新規学習なし。
- Group: coverage は well 単位で記録する。
- Stratification: なし。
- Leakage Check: true TVT、oracle candidate、absolute error、within-threshold label、true-error rank を generator 入力にしない。

## 実行入口

- 学習 notebook: `exp145_learned_likelihood_rawtest_feature_generator_parity_train.ipynb`
- 推論 notebook: `exp145_learned_likelihood_rawtest_feature_generator_parity_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp145_learned_likelihood_rawtest_feature_generator_parity`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| full-train rows | 3,783,989 |
| raw-test rows | 14,151 |
| schema parity | pass |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- schema parity と raw-test regeneration の blocker を実装対象として分離した。
- exp112 schema 互換の feature cache を full train / raw test に同じ関数で出せる。
- train v2 で full-train 773 wells、inference v3 で raw-test 3 wells の schema parity が通った。

### 悪かった点

- exp111 は学習時 imputation medians を保存していないため、generator は batch median imputation を使う。この制約は summary に記録する。

### リスク / 注意

- schema/coverage parity が通っても submit 判断にはしない。
- raw-test replay は PF/Beam/likelihood-PF を再実行するため Kaggle runtime で確認する。

## 次

1. exp145 cache を使って exp092 full-row learned likelihood add-only 実験を作る。
2. worst-well、exp115 hidden-like stress、raw-test flow が崩れないか確認する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
