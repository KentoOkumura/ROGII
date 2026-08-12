# 設計

## アプローチ

`exp027_public_replay_needless090_sel15_spread3` の replay 実験構造をコピーし、inference notebook を archived public notebook `lb-8-860-rogii-sel15-256seeds.ipynb` で置換する。

source:

- ref: `needless090/lb-8-860-rogii-sel15-256seeds`
- title score: `LB 8.860`
- path: `docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_latest/needless090__lb-8-860-rogii-sel15-256seeds/lb-8-860-rogii-sel15-256seeds.ipynb`
- metadata: CPU、internet off、external dataset/kernel/model sources なし

Kaggle package は既存の `scripts/prepare_kaggle_notebooks.py` を使う。bootstrap cell は repository support files を展開するために自動挿入されるが、public notebook logic は変更しない。

## 実験範囲

- 対象実験: `exp028_public_replay_second_sel15_or_blend_audit`
- Route: `pf_beam`
- 親実験: `public_notebook_catchup_after_self_improvements`
- 変更する変数: Kaggle replay target notebook
- 固定する変数: public notebook code、competition input、internet off、GPU off、external sources none
- blend: config では disabled とし、exp028 replay output / LB が揃うまで生成しない

## リスク

- リークリスク: formation/geology branch、public-visible branch、static submission/blend の flag がある。replay 後に output と code path を確認する。
- CV/LB 不一致リスク: local CV はなく public title score 由来の候補なので、self CV と比較しない。
- ランタイム/メモリリスク: 256 seed PF + beam ensemble なので exp027 より CPU runtime が長い可能性がある。Kaggle runtime を記録する。
