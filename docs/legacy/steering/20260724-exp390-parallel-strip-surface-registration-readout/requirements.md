# 要件

## 依頼

XY 上で近接する horizontal well がほぼ平行である構造を、単なる well-level
近傍距離ではなく、共通の along-track / cross-track 座標へ登録して利用する実験を設計する。
`KAGGLE_DIRECTION.md` のバックログ、`docs/legacy/steering/`、実験ディレクトリを作成し、
仮説、単一変更、fold-safe な評価契約、停止条件、再現性契約を固定する。

今回は design-only とし、実装、正規 Notebook 採用、Kaggle package、push、run、
inference、submission は行わない。

## 制約

- 対象実験: `exp390_parallel_strip_surface_registration_readout`
- Route: `pf_beam`
  - fitted PF / Beam を実行するためではなく、outer-train の物理的な surface donor から
    決定論的 path candidate を作る route として扱う。
- 親・control: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
  の保存済み outer-5-fold OOF。control は再生成しない。
- 科学的には `exp383_all_tvt_stratigraphic_vector_drift_field` と独立した
  no-formation parallel-strip 仮説とする。ただし重複計算を避けるため、
  実装優先順位は exp383 の Stage 0 / Stage 1 結果を確認してから再判定する。
- target well は freeze 前に `MD/X/Y/Z/TVT_input` だけを使用できる。
  target の suffix `TVT`、生の6 Formation列、`GR` は使用禁止。
- donor surface は outer-train well のみから `S=TVT+Z` として作る。
  outer-valid well は donor、fit、pair family、閾値推定のすべてから除外する。
- visible 3 sample wells を pair threshold、bandwidth、gate、候補選択に使用しない。
- nearest-well TVT、OOF residual、GR waveform をそのままコピーしない。
- 変更する科学変数は、XY を query-centric な parallel-strip 座標
  `(s,n)` へ登録し、同じ `s` の donor `S` を `n` 方向へ補間することだけとする。
- 再現性は `docs/06_reproducibility.md` に従い、fold、well、pair、node、
  donor、fit、prediction の安定順序と logical content SHA を記録する。
- RNG、GPU、LightGBM、HMM、PF、Beam、seed bagging は使用しない。
- 同じ OOF を見た後の angle、distance、overlap、donor count、bandwidth、
  smoothing、Huber、prefix calibration、fallback、blend weight の救済 grid を禁止する。

## 受け入れ基準

- `docs/legacy/steering/20260724-exp390-parallel-strip-surface-registration-readout/` に
  requirements / design / tasklist があり、未記入placeholderなしで設計が固定されている。
- `experiments/exp390_parallel_strip_surface_registration_readout/` に
  route、lineage、検証段階、固定値、実行禁止フラグ、再現性契約が記録されている。
- `KAGGLE_DIRECTION.md` の未着手バックログに、exp383 との順序、
  1 candidate / 5 reporting folds / fitted model・HMM・PF・Beam・booster各0、
  control 再生成0が記録されている。
- `experiment_summary.md` に design-only 実験として登録されている。
- Stage 0 は target-free geometry/support/resource gate、Stage 1 は known-prefix
  rolling-origin、Stage 2 は truth-late suffix direct score と promotion safety に分離されている。
- direct candidate は two-sided support を満たす行だけ strip surface を使用し、
  それ以外は保存済み exp226 control へ exact fallback する単一variantである。
- scientific-support gate と inference-promotion gate が分離され、後者を通らない限り
  current-test generation / inference / submission に進めない。
- 実装前状態では正規 train / inference Notebook は template scaffold のままで、
  package / push / run の承認フラグがすべて false である。
- deterministic anchor として扱う場合は、Kaggle kernel version、入力SHA、
  pair/node/fit/prediction logical SHA、decompressed prediction SHA、
  成功rerun一致が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく
  decompressed content SHA を主証拠として記録している。
