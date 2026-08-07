# exp405_geometry_reinjected_interval_semimarkov_fusion

## 状態

- Route: `pf_beam`
- 状態: implementation-only完了・未実行
- CV / LB: なし
- 親: `exp293_physics_only_candidate_bank_headroom_contract`
- promotion control: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- 設計の正:
  `.steering/20260726-exp405-geometry-reinjected-interval-semimarkov-fusion/`

## 仮説

exp293の保存済み12物理pathにはH512 oracle RMSE `3.683763 ft`の
十分なsupportがある。H256 blockの局所GR形状尤度、H512以上のcandidate
duration、dockingと独立したexp226 geometry再注入を組み合わせれば、
exp399のwrong-mode lock-inを避けながらこのheadroomを回収できる。

## 固定方式

- candidate: exp293 deployable12を値・順序・SHAごと固定
- block: non-overlap H256、candidate最小duration 2 blocks
- observation: candidate TVT周囲`±55 ft`を5 ft刻みで周辺化
- morphology: raw / rolling-21 / rolling-101を`0.50 / 0.25 / 0.25`
- geometry floor: 新segmentの`exp226_k16` prior `>=0.10`
- solver: exact log-space semi-Markov forward-backward
- output: block posteriorを補間した12 pathの凸結合
- Viterbi、hard top1、row-wise switch、ML、path再生成なし

## 検証方針

primary OOF RMSE `<=6.90 ft`、exp263比5/5 folds改善、1000+と
hidden-like 2面改善、well-tail非悪化、real GRが2 negative controlsより
5/5 foldsで良いこと、geometry posterior非退化を全ANDで要求する。

PASS時だけ同じexp405でcurrent-test実装を解禁する。技術的に有効な
scientific FAIL時はexp405を救済せず閉じ、exp406 Stage 0を解禁する。

## 所見

候補supportはexp293で確認済みであり、本実験の反証対象はtarget-freeな
区間posteriorがそのheadroomを回収できるかである。exp297、exp399、exp370と
同じ証拠・遷移・triggerへ戻らないことが設計上の重要条件である。

## 実装状態

- 別名Jupytext source / compact self-contained train候補: 実装済み
- exact semi-Markov / morphology / negative controls / truth-late readout: 実装済み
- dedicated synthetic contract test: 実装済み
- 正規train / inference Notebookと`settings.py`: template placeholderのまま
- Kaggle package / fixed16 / full OOF: 未作成・未実行
- current-test / inference / submission: train-side PASS前のため無効

実行flagは閉じており、fixed16 preflight、full saved-OOF、正規Notebook採用は
それぞれ別承認を必要とする。

## 表記

用語は`KAGGLE_DIRECTION.md`と`docs/glossary.md`に合わせ、
実験名・設定名以外は日本語優先で記録する。
