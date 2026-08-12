# 要件

## 依頼

train-sideでGaussian exact HMMより平均RMSE、4/5 folds、全固定scopeを改善した一方、
by-well tail guardで不採用となったexp374の固定Student-t exact HMMを、
単体物理モデルとしてPublic LBで記述評価する。
設計、steering、実験scaffold、バックログだけを確定し、実装・実行・提出は行わない。

## 制約

- Route: `pf_beam`
- 1物理モデルにつき1実験、将来の実行対象となるevaluation/inference Notebookは1本だけとする。
- 候補はexp374の`df=4.0` Student-t exact HMMだけとする。
- exp209のabsolute-TVT grid、41 rate states、transition、prior、sigma、GR補完、
  Type Well処理、posterior mean出力を変更しない。
- Gaussian control、Huber sibling、exp342 residual-offset Student-tを同じNotebookで実行しない。
- df、scale、temperature、clip、sigma、grid、transition、blend weightを探索しない。
- blend、selector、gate、postprocess、ML学習、parent/control再実行を行わない。
- suffix TVT、error、fold、hidden-like role、Public LBをdecode入力にしない。
- sample submission由来の動的ID/well contractを使い、公開3-well固定assertを禁止する。
- CPU、internet off、no RNGとし、入力順・well順・grid/rate順を固定する。
- 実装、Kaggle push/run、competition submissionにはそれぞれ別のユーザー承認を必要とする。

## 受け入れ基準

- Student-t式、HMM状態空間、入力、出力、実行量、禁止事項、LB解釈規則が一意に固定されている。
- 実装後の候補は各test wellで1回だけdecodeされ、Gaussian/Huber control runは0である。
- `submission.csv`はsample submissionと同じID集合・行数、列`id,tvt`、重複0、欠損0、finite 100%である。
- posterior normalization、row identity、well status、fallback 0をhard gateにする。
- input/source/config SHA、prediction SHA、submission SHA、Kaggle kernel versionを記録する。
- gzip生成物はdecompressed content SHAを主証拠として記録する。
- Public LB確定後も自動昇格、df調整、blend作成、追加候補生成を行わない。
