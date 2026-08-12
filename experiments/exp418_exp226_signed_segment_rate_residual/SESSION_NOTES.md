# exp418_exp226_signed_segment_rate_residual セッションノート

## 目的

exp226根本原因監査で確定したsigned rate mismatchの累積に対し、exp333と同じ
target-free feature/fold/modelを使いながら、K16 rateを連続積分する一因子実験を
design-onlyで固定する。

## 現在の状態

- Route: `ensemble`
- 状態: Stage 0 version 1 technical FAIL / terminal fail closed
- oracle RMSE: 0.6469514161595739（deployable model CVではない）
- LB: なし
- 親: exp333
- base: exp226
- model / prediction / inference / submission: なし
- compact train候補: 実装済み
- 正規Notebook: compact候補を採用
- Stage 1 / inference / submission: 未承認・未実行

## 固定した設計

- K16 assignment、outer/inner fold、136-feature contract、LightGBM configはexp333を継承
- residual sign: `true_tvt - nested_exp226_prediction`
- rate unit: `ft/row`
- target: first-row補正0の16列cumulative basisに対するfloat64 least squares
- interval assignment: destination rowのsegment
- correction: `basis @ predicted_rates`
- first unknown row correction: 0
- segment offset/intercept、clip、shrink、taper、absolute re-anchor: なし

## 実行予定量

| 段階 | variant | config | folds | boosters | exp226 fit | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 0 | 0 | 0 | reporting 5 | 0 | 0 | 0 |
| Stage 1 | 1 | 1 | training 5 | 5 | 0 | 0 |

- control再学習: 0
- PF/HMM/Beam再生成: 0
- Stage 1はStage 0全gate PASSと別承認が必要
- inference / submissionはさらに別承認が必要

これは固定した実行量である。実装は完了したが、Kaggle実行承認ではない。

## 再現性メモ

- exp333 nested baseとfold manifestをSHA固定で再利用する。
- expected exp333 feature-freeze SHA:
  `b2c7bff40f9fc994bd60471c03d9085ba48137c30b358402bfbb1cadecc4a078`
- expected feature schema content SHA:
  `8a6ae01d792e6cf352f22c4519ec7934e5e39ee032a8ae676da9449722684a45`
- saved exp226 OOF decompressed SHA:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- CPU LightGBM、`random_state=0`、`deterministic=true`、`force_col_wise=true`
- nested artifact欠損/SHA不一致時はexp226再生成で救済せずfail closed
- current-test rerunがないためdeterministic submission anchorとは呼ばない

## コマンドログ

2026-07-27:

- `make new-steering EXP=exp418_exp226_signed_segment_rate_residual`
- `make new-exp EXP=exp418_exp226_signed_segment_rate_residual`
- steering、scaffold、config、backlog、experiment summaryをdesign-onlyで作成
- 実装、Notebook編集、Kaggle package、push/run、推論、提出は0

2026-07-27 実装:

- ユーザー依頼「exp418を実装してください」を、compact train候補、
  対応Jupytext Notebook、専用contract testの実装承認として扱った。
- `exp418_exp226_signed_segment_rate_residual_compact_selfcontained_train.py`
  を作成した。
- `exp418_exp226_signed_segment_rate_residual_compact_selfcontained_train.ipynb`
  をJupytextで生成した。既存正規train Notebookは上書きしていない。
- exp333 Stage 1の保存SHA manifestからnested prediction、fold manifest、
  feature schemaを解決し、file/decompressed/content SHAを検証する。
- exp333と同じtarget-free 136-feature surfaceを再構築し、row content SHA
  `9475721131bfd93a036d0a636d473a8cf6cc8d7d46eaf203b3879ccba6272a79`
  とfeature-freeze SHA
  `b2c7bff40f9fc994bd60471c03d9085ba48137c30b358402bfbb1cadecc4a078`
  を照合する。
- exp226 sourceをimportせず、`build_fields`、`fit_kappa`、nested regenerationを
  実装に含めない。Stage 1のexp226 fitは0。
- destination row segment interval、first-row correction 0のK16 cumulative basis、
  float64 `numpy.linalg.lstsq(..., rcond=None)`、matrix/逐次integration parityを
  実装した。
- Stage 0 oracle、Stage 1 5-fold LightGBM、fold/distance/hidden-like/boundary/
  by-well/rate-sign/rate-RMSEのAND gate、feature importance plot、model/OOF/SHA
  保存を実装した。
- Stage 1はStage 0 summaryのfile SHAをconfigへ固定し、technical/scientific
  `PASS_STAGE0`を確認しなければfail closedとした。
- 実装量: Stage 0は0 model / 0 booster。Stage 1は1 variant ×1 config ×5 folds =
  5 CPU boosters。control再学習0、PF/HMM/Beam 0。
- 親compactとの比較:
  - exp333: 2,297行、12章
  - exp418: 2,342行、12章
  - 保存入力、basis、late truth、oracle、feature、training、metrics/importance/SHA、
    guarded executionの全役割を維持した。
- 検証:
  - `.venv/bin/pytest -q experiments/exp418_exp226_signed_segment_rate_residual/tests/test_exp418_exp226_signed_segment_rate_residual.py`
    → `14 passed`
  - `py_compile` → PASS
  - Ruff `--select F821` → PASS
  - Jupytext変換 / `--test` → PASS
  - `__file__` → 0件
- repository全体の`make test`も実行したが、exp418 test実行前のcollectionで既存6件が
  errorとなった。exp297 / exp301 / exp333 / exp336 / exp349は各sourceがrepository
  rootの`config.yaml`を誤って読む既存問題、exp411は`numba.__spec__ is None`の既存
  collection問題である。exp418専用14件、strict experiment validation、
  template validationはすべてPASSしている。今回のscope外ファイルは修正しない。
- Kaggle package、push/run、推論、提出は0。

2026-07-28 Stage 0実行承認:

- ユーザー依頼「実行してください」を、直前に提示した次段階であるcompact候補の
  正規train Notebook採用、Kaggle package、Stage 0 push/runの承認として扱う。
- 実行対象:
  - readout: 1
  - active variant: 0
  - model config: 0
  - trained fold: 0
  - booster: 0
  - exp226 fit / control再学習: 0
  - PF/HMM/Beam再生成: 0
  - GPU: 0（Kaggle private CPU、internet off）
- 入力はSHA固定したexp333 Stage 1 nested prediction / fold manifest /
  feature schema / SHA manifestと、保存exp226 OOFだけとする。
- canonical kernel:
  `kentookumura/exp418-exp226-signed-segment-rate-train`
- title: `exp418 exp226 signed segment rate train`。id末尾slugとtitle由来slugは一致。
- Stage 1、inference、submissionは未承認で、自動移行しない。
- repository標準の`task prepare-kaggle-notebooks`を試したが、この実行環境には
  `task` executableがなく終了した。処理は開始されておらず、同じrepository標準
  package scriptを呼ぶ`make prepare-kaggle-notebooks`へ切り替える。
- `make prepare-kaggle-notebooks ... --strict`でtrain packageを生成した。
  metadataはprivate / CPU / internet off / run-on-pushで、competition sourceと
  exp072・exp333・exp226の3 kernel sourceを確認した。bootstrap manifestには
  config、exp228 source、hidden-like fold assignmentを含み、package configとの
  byte一致も確認した。
- `make push-kaggle-train EXP=exp418_exp226_signed_segment_rate_residual`
  → Kaggle kernel version 1をpushした。
- push直後の状態: `KernelWorkerStatus.RUNNING`。
- version 1は`KernelWorkerStatus.COMPLETE`。Kaggle metadata id_noは
  `128832515`、Stage 0 summaryは約146秒で出力された。
- Stage 0結果:
  - rows / wells / segments: 3,783,989 / 773 / 12,368
  - model / booster / exp226 fit: 0 / 0 / 0
  - exp226 RMSE: 9.42710959658222
  - rate oracle RMSE: 0.6469514161595739
  - gain: 8.780158180422646 ft、fold gain 5/5 PASS
  - technical checks 8/9 PASS
  - matrix / sequential integration差:
    6.295408638834488e-12 ft > 固定上限1.0e-12 ft
  - `technical_fail` / `FAIL_CLOSE_BRANCH`
- outputを`/tmp/exp418-stage0-v1`へ取得し、summaryとSHA manifestを照合した。
  - summary file SHA:
    `07c719e0f174b1712650563620f6331504dbd1333969c8777f41ce46419dc412`
  - rate-target content SHA:
    `5c936b03e86e7250afdfef551e796e0beead22d50715d025b84acb9b13a9e2ff`
  - rate-target file SHA:
    `a705f37ba3529f92fcd7d5da441550d0e0742a462bb6d1131fab562cb0b3695e`
- 科学閾値2件は個別には成立したが、technical prerequisiteがFAILしたため
  `scientific_pass=false`である。Stage 1 approval keyを有効化せず停止した。
- `6.30e-12 ft`はfloat64の演算順による極小差と考えられるが、結果確認後の
  `1e-12` gate緩和はsame-OOF rescueになるため行わない。exp418は閉じる。

## 次のアクション

Stage 1、inference、submissionへ進まず停止する。必要なら別承認後、truth-freeな
cross-runtime ULP/scale-aware parityと単一canonical integrationを先に固定する
独立numerical-contract auditを別実験として設計する。
