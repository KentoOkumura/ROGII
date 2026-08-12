# 要件

## 依頼

HMM 状態変数案として fault/reset duration semi-Markov HMM の backlog、実験ディレクトリ、
steeringを作り、実装前の設計を確定する。

## 制約

- Route: `pf_beam`
- `docs/06_reproducibility.md` に従う。
- 実装、Notebook置換、Kaggle実行、推論、提出は行わない。
- triggerはraw GRと保存済みexp209 emissionだけでtarget-freeにする。
- atlas、rate predictor、oracle、global branchを使わない。

## 受け入れ基準

- trigger、jump、duration、commit、refractoryが一意に固定されている。
- trigger/branch/selectionをtruth join前にfreezeする。
- Stage 0/1 gate、実行量、fail policyが固定されている。
- backlog と summary に保留・設計確定・未実装として登録する。
