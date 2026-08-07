# 設計

## アプローチ

テンプレートから `exp027_public_replay_needless090_sel15_spread3` を作成し、inference notebook を archived public notebook で置換する。

source:

- ref: `needless090/lb8-781-rogii-sel15-spread3`
- title score: `LB8.781`
- path: `docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_latest/needless090__lb8-781-rogii-sel15-spread3/lb8-781-rogii-sel15-spread3.ipynb`
- metadata: CPU、internet off、external dataset/kernel/model sources なし

Kaggle package は既存の `scripts/prepare_kaggle_notebooks.py` を使う。bootstrap cell は repository support files を展開するために自動挿入されるが、public notebook logic は変更しない。

## 実験範囲

- 対象実験: `exp027_public_replay_needless090_sel15_spread3`
- 親実験: `public_notebook_catchup_after_self_improvements`
- 変更する変数: Kaggle replay target notebook
- 固定する変数: public notebook code、competition input、internet off、GPU off、external sources none

## リスク

- リークリスク: formation/geology branch、public-visible branch、static submission/blend の flag がある。replay 後に output と code path を確認する。
- CV/LB 不一致リスク: local CV はなく public title score 由来の候補なので、self CV と比較しない。
- ランタイム/メモリリスク: 128 seed PF + beam ensemble なので CPU runtime が長い可能性がある。Kaggle runtime を記録する。
