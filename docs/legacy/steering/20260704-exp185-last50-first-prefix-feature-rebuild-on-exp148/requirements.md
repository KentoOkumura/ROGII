# 要件

## 依頼

`last50_first_prefix_feature_rebuild_on_exp148` backlog を実装し、Kaggle GPU で学習実行する。

## 制約

- Route: `ml_model`
- 親実験は `exp148_learned_likelihood_fulltrain_addonly_on_exp092`。
- control / exp148 baseline は再学習しない。保存済み exp148 CV / Public LB を比較基準にする。
- known prefix source を先に last50 に切ってから prefix 由来特徴を再計算する。
- 学習・評価行は crop しない。
- `last_known_tvt`、anchor row、base prediction、PF/Beam 候補値、U-projection、learned probability/error model は既存 exp148/exp145 surface を読む。
- valid/test true TVT、oracle best、true-error rank を特徴量生成に使わない。
- GPU push 前に active variant 数、mode 数、LGB config 数、fold 数、合計 booster 数、control 再学習なしを `SESSION_NOTES.md` に記録する。
- 再現性は `docs/06_reproducibility.md` に従う。

## 受け入れ基準

- exp185 の `config.yaml`、notebook `.py` / `.ipynb`、README、SESSION_NOTES、result、metrics が exp185 用に整合している。
- feature cache notebook が last50-first rebuild features を生成し、schema/summary/manifest を保存する。
- split train notebooks が `lgb0` / `lgb1` / `lgb2` を Kaggle GPU metadata で実行できる。
- `py_compile`、`ruff --select F821`、Jupytext 変換/検証、`make validate-exp` が通る。
- Kaggle 実行後、CV、fold 別 score、生成物、SHA、判断を `SESSION_NOTES.md` と `result.md` に記録する。
