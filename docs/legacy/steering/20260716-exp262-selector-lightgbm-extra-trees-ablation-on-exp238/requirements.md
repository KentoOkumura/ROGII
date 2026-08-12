# 要件

## 依頼

`KAGGLE_DIRECTION.md` の未着手バックログ
`selector_lightgbm_extra_trees_ablation_on_exp238` を
`exp262_selector_lightgbm_extra_trees_ablation_on_exp238` として実装する。

## 制約

- Route: `ml_model`。
- exp238 selector train v4 の outer 5 × inner 4 well GroupKFold、11候補、184 context + candidate 3、学習行上限、seed、objective、Viterbi ruleを固定する。
- selector LightGBMへ `extra_trees=True` だけを追加する。objective、seed、sampling、候補値、context feature、candidate bankを同時変更しない。
- 保存済みexp238の20 selector models、nested OOF score、metricsをcontrolとして再利用し、controlを再学習しない。
- 初回probeは1 selector config × outer 5 × inner 4 = 20 CPU boostersのみとし、exp218 downstream 15 boostersを再学習しない。
- selector guard通過前のraw-test inference、competition submit、threshold grid、exp251との併合を禁止する。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic sampling、LightGBM、Kaggle bootstrap、SHA記録を設計に明記する。

## 受け入れ基準

- Jupytext percent形式を起点に、人間が入力契約、fold、単一parameter差分、学習、control比較、guard、生成物を追えるtrain notebookを作る。
- configでactive variant 1、selector config 1、outer 5、inner 4、合計20 boosters、control再学習0、downstream再学習0を明示する。
- exp238 historical scoreを同一row/fold/candidate契約で読み、candidate error MAE/rank、candidate logloss、fixed top1/Viterbi RMSE、score相関を比較する。
- global / near / 1000+ / exp115 hidden-like 2面 / fold / by-well / worst-wellをcontrol比で評価し、全guardを満たすまで後段を停止する。
- model SHA、feature schema SHA、OOF score decompressed SHA、historical input SHAを保存する。
- Jupytext test、`py_compile`、ruff F821、strict experiment validationが通る。
- deterministic anchorとは扱わず、rerun前であることを記録する。

## 次のアクション

Kaggle CPU trainを実行する場合は、20 boosters・historical control再学習0・downstream再学習0を
確認してユーザーの明示承認を得る。selector guard通過後のdownstream再学習も別途確認する。
