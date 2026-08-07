# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- Stage 0で両variantがmechanism gate FAILのため、Stage 1、inference、
  submissionはfail-closed。

## 完了

- exp434が別実験に使用済みであり、exp435が未使用であることを確認した。
- `docs/06_reproducibility.md`を読み、乱数なしCPU HMMのSHA方針を固定した。
- 現行HMM、41-rate memoryless、dz-onlyの状態と因果比較を固定した。
- `memoryless_41rate`の41 rate gridとzero-centered stationary重みを固定した。
- `dz_only_r0`を同一kernelのdelta-at-zero特殊ケースとして固定した。
- Stage 0 / Stage 1の実行量、gate、禁止事項を固定した。
- fixed32をmechanism-only、full OOFをpromotion判断とする境界を固定した。
- exp263固定式のHMM成分置換をroute adoption readoutとして固定した。
- steeringをdesign-onlyとして作成した。
- design-only実験scaffoldを作成した。
- `KAGGLE_DIRECTION.md`へ2 treatmentを含む1件のP2 backlogとして登録した。
- `experiment_summary.md`へdesign-only実験を反映した。
- strict experiment validationとtemplate validationを通した。
- 2026-07-29のユーザー追加指示によりimplementation承認を得た。
- compact self-contained Jupytext train候補を実装した。
- exp209には親compactがないため、同じfixed32 exact-HMM系のexp424 compactと
  章立て・記載量を比較し、同じ9章の読める構成を確認した。
- `memoryless_41rate`と`dz_only_r0`を同じTVT-only position kernelで実装した。
- stationary rate weight、delta-at-zero parity、rate-state非保持、
  normalization、truth / role / fold / episode / cause late-readの
  contract testを作成した。
- fail-closed inference placeholderを実装した。
- compact Notebookを正規train / inference Notebookへ採用した。
- py_compile、Ruff F821、専用test 11件、Jupytext train / inference
  round-trip、strict experiment validationを通した。
- Stage 0実行量を2 treatment ×32 wells = 64 HMM well-runs、
  保存parent rerun 0、model / booster / PF / Beam / GPU 0と再確認した。
- 2026-07-29のユーザー指示により、固定済みStage 0のpackage / push / run承認を得た。
- canonical Kaggle private CPU version 1（id_no `129049294`）を完了した。
- technical gateは全PASS、memoryless / dz-onlyのmechanism gateは双方FAILと判定した。
- Stage 1 eligible variantを空にし、same-OOF救済なしでbranchを閉じた。
