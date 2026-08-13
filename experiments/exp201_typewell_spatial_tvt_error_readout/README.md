# exp201_typewell_spatial_tvt_error_readout

## 状態

- ルート: ml_model
- 状態: running
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-05
- 親実験: exp148_learned_likelihood_fulltrain_addonly_on_exp092

## 仮説

exp148 OOF の誤差には、共通 typewell group、XY 近傍、true TVT の急変形状、well 全体の offset によって説明できる局所的な偏りが残っている可能性がある。

## 変更点

- 新規学習なし。
- exp148 train v1 の `lgb_mean` OOF prediction を読み、raw train の MD / X / Y / Z / TVT と join する。
- `native_overlap_1` の 54 共通 typewell group と、XY 近傍 k=8 による誤差傾向を集計する。

## 検証方針

- Fold: exp148 保存済み OOF prediction を使用。
- Group: well 単位 OOF。
- Stratification: なし。診断面として typewell group / XY 近傍 / TVT 急変 / offset を後集計する。
- Leakage Check: 後続 feature 化や gate 化には直接使わない。診断結果だけを保存する。

## 実行入口

- 学習 notebook: `exp201_typewell_spatial_tvt_error_readout_train.ipynb`
- 推論 notebook: `exp201_typewell_spatial_tvt_error_readout_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp201_typewell_spatial_tvt_error_readout`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- TODO

### 悪かった点

- TODO

### リスク / 注意

- TODO

## 次

- TODO

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
