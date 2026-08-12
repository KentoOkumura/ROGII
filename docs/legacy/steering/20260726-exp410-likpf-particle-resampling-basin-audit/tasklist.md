# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 原因taxonomy、primary threshold、truth分離、再現性、Kaggle実行方針を事前登録。
- exp410 experimentをscaffoldし、route / lineage / approval / costを記録。
- PF-specific 496 wells / 839 episodes / 819,288 rowsのfloat32-normalized
  assetとSHAを固定。
- exp243 exact replay kernelへread-only stage diagnosticsを実装し、合成系列で
  bit-level parityを確認。
- 4 balanced shard entrypoint、Jupytext notebooks、strict validationを準備。
- Kaggle preflight v3を4 wellsで完了し、全well persisted parity 0、mean
  88.638 sec/well、peak RSS 1.980GBを確認。
- full対象496 wellsをKaggle CPU 4 shardで完了し、strict merge・coverage・
  parity・SHA guardをPASS。
- counterfactual 1 sentinel ×12 variantsのKaggle CPU preflightを完了し、
  baseline parity 0、12 variants / 22 readouts、peak RSS 1.981GB、全guardをPASS。
- target-late sentinel選択（cause代表 + global SSE、最大12 wells）と、
  baseline / initialization / transition / GR / resampling / roughening /
  clampの12 variantsを
  full結果を読む前に固定。
- 固定12 sentinel wells ×12 variantsをKaggle CPU 4 shardで完了し、
  144 well-runs / 22 readoutsをstrict merge、baseline parity 0と全guardをPASS。
- 観察監査、HMM比較、counterfactualからPF offsetの主因をfinite particle supportと
  particle / seed basin平均、resampling genealogyを副次的な増幅・回復レバーと確定。
- `SESSION_NOTES.md`、`result.md`、`metrics.json`、`experiment_summary.md`、
  `KAGGLE_DIRECTION.md`へ最終結果とbacklog影響を反映。
- JSON、py_compile、Ruff F821、
  `make validate-exp EXP=exp410_likpf_particle_resampling_basin_audit`をPASS。
