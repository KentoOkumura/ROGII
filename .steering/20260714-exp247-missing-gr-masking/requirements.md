# 要件

## 依頼

`KAGGLE_DIRECTION.md` の `missing_gr_masking` を `exp247_missing_gr_masking` として実装する。
exp221 exact HMM で horizontal GR 欠損 row を補間済み観測として扱っている箇所だけを変更し、raw GR が欠損した評価 row の GR emission contribution を 0 にする。

## 制約

- Route: `ensemble`。exp148 の固定 OOF LightGBM unary と exact HMM grammar の両方が予測生成に本質的に寄与する。
- exp221 の grammar、TVT grid、transition、GR sigma、LGB unary `sigma=20/lambda=0.50`、補間 control を固定する。
- exp221 の保存済み control prediction を固定入力として読み、control HMM や exp148 LightGBM を再学習・再生成しない。
- 新規に生成する HMM variant は `mask_only` 1本だけとし、LightGBM config 0、fold 0、booster 0 とする。
- 欠損判定は raw horizontal well の `GR.isna()` のみで行う。true TVT、OOF error、oracle、hidden-like role は gate に使わない。
- 評価 row の raw GR が欠損なら GR emission 行列を state-neutral な 0 にする。LGB unary と transition はその row でも通常どおり使う。
- raw train/test の欠損 run 分布を先に集計する。visible test は分布監査だけに使い、score evidence と解釈しない。
- sigma、lambda、grid、transition、補間法、run-length gate を同時変更しない。
- 初回実行では raw-test prediction、submission、selector、run-length 別 gate を作らない。
- 再現性: `docs/06_reproducibility.md` に従い、input/control/output の SHA と gzip decompressed content SHA を記録する。

## 受け入れ基準

- raw GR 欠損 row では GR emission が全 state で厳密に 0、観測 row では exp221 control と一致する synthetic assertion がある。
- exp221 control と mask-only の ID coverage、row count、well count、finite-state coverage が一致する。
- overall、raw missing-run 長、missing run 終端後 128/256 rows、distance `1000+`、exp115 hidden-like、worst-well、control からの連続分岐長を記録する。
- mask-only の overall / subgroup / by-well metric と control delta、改善/悪化 well 数、最大悪化を保存する。
- `config.yaml`、Jupytext train/inference、通常 `.ipynb`、`SESSION_NOTES.md`、`result.md`、`metrics.json` を作成し、`task validate-exp` 相当と静的検証が通る。
- Kaggle train package の variant/config/fold/booster 数と parent/control 再学習なしを `SESSION_NOTES.md` に記録する。
- gzip 生成物を比較する場合は raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録する。
