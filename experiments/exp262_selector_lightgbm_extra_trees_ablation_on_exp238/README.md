# exp262_selector_lightgbm_extra_trees_ablation_on_exp238

## 状態

- ルート: `ml_model`
- 状態: Kaggle train v1完了、selector guard不通過・不採用
- CV: fixed Viterbi 8.826521（historical exp238 8.492559、+0.333962）
- Public LB: 未実行
- 親実験: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`

## 仮説

exp238のnested candidate-error selectorへ `extra_trees=True` だけを追加すると、
候補誤差scoreの汎化またはhistorical selectorとの多様性が改善する可能性がある。

## 変更点

- exp238 train v4の11候補、184 context、candidate-long 3列、outer 5 × inner 4、
  bounded sampling、objective、seed、fixed Viterbiを固定する。
- selector LightGBMの `extra_trees=True` だけを変更する。
- 保存済みexp238 nested OOF score/model manifestをcontrolとして再利用する。
- 初回は20 CPU boostersのみで、controlとdownstream exp218 LightGBMは再学習しない。

## 検証方針

candidate error MAE、oracle-candidate logloss、rank accuracy、score相関、fixed top1、
fixed Viterbi、global / near / 1000+ / exp115 hidden-like / fold / worst-wellを
historical exp238と比較する。全guard通過前はinference・downstream再学習・submitへ進まない。

## 所見

20 CPU boostersを完走したが、historical exp238比でcandidate error MAE +0.215430、
fixed Viterbi global +0.333962、1000+ +0.379748、hidden-like +0.798344 / +0.777549、
最大well回帰 +12.835997、nonworse 1/5 foldsとなりguard不通過。nearだけは-0.040474改善した。
実装/fold/SHA契約は通っており、extra-trees仮説をnegative resultとして閉じる。

## 実行入口

- 学習: `exp262_selector_lightgbm_extra_trees_ablation_on_exp238_train.ipynb`
- 推論: `exp262_selector_lightgbm_extra_trees_ablation_on_exp238_inference.ipynb`
  （guard通過と別途承認まで停止）
- 初回実行先: Kaggle CPU Notebook

## 実行量

- active variant: 1
- selector config: 1
- outer folds: 5
- inner folds: 4
- 合計: 20 CPU boosters
- control再学習: 0
- downstream再学習: 0

## 実行結果

- kernel: `kentookumura/exp262-sel-extra-trees-exp238-train` version 1 / `id_no=127468598`
- status / runtime: `COMPLETE` / 18,760.043秒
- models: 20/20、model SHA一致
- fixed prediction decompressed SHA: `ac3cc0bfa8d132641453f7f404f829ef3d9e2abc63e7151270b8a89c2363f41a`
- deterministic anchor: false（同一kernel rerun未確認）

## 次

extra-treesのparameter rescue grid、downstream再学習、raw-test inference、submitは行わない。
既存の`exp264_exp263_candidate_confidence_dual_selector`をstandard LightGBM固定の次routeとして維持する。
