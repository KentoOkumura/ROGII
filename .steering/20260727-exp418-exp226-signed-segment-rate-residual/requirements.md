# 要件

## 依頼

exp226 の persistent offset 根本原因監査で特定した「局所的な signed rate
mismatch の累積」に直接介入する `signed segment-rate residual` 案について、
backlog、実験ディレクトリ、steering を作成し、実装前の設計を確定する。

今回は design-only とし、実装、Notebook 編集、Kaggle package、Kaggle 実行、
推論、提出は行わない。

## 仮説

exp226 は最後の既知 TVT を一度だけ絶対 anchor とし、unknown suffix では
spatial donor 由来の相対増分を累積する。target well と donor field の小さな
signed rate mismatch が K16 区間をまたいで積分されることが persistent offset の
根本機構である。

exp333 の segment-constant offset target は exp226 を改善したが、K16 境界へ
level correction を直接加えるため near 0--250 と worst well を悪化させた。
同じ target-free feature、fold、LightGBM 設定を保ったまま、各 K16 区間の
signed residual rate を学習し、先頭補正 0 から連続積分すれば、境界 step を作らず
累積 drift の発生段階へ介入できる。

## 制約

- Route は `ensemble` とする。exp226 の deterministic geometry/KNN prediction と
  LightGBM residual-rate model の両方が最終予測へ直接寄与する。
- 一因子比較元は
  `exp333_exp226_k16_segment_residual_offset_target`、base prediction の親は
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction` とする。
- K16 assignment、保存済み exp226 reporting fold、exp333 strict nested inner fold、
  target-free 136-feature schema、LightGBM 1 config を変更しない。
- target は各 well の残差 `TVT - nested_exp226` を、先頭補正 0 の K16 cumulative-rate
  basis へ float64 最小二乗した 16 個の signed rate (`ft/row`) に固定する。
- target solve は intercept、ridge、Huber、clip、weight を持たない
  `numpy.linalg.lstsq(..., rcond=None)` とする。
- correction は first unknown row を 0 とし、各隣接 row intervalへ destination row の
  segment rate を加算して suffix 全体で累積する。
- segment constant offset、absolute re-anchor、clip、shrink、taper、interpolation、
  boundary smoothing、well gate を併用しない。
- feature は exp333 Stage 1 の 136 列を同じ順序・集約で再現する。新しい donor-risk、
  K12/K24、target/error/oracle、well ID、selector score feature を追加しない。
- strict nested exp226 prediction は exp333 Stage 1 の SHA-frozen target-free生成物を
  再利用する。exp418 内で donor field / kappa / exp226 control を再 fit しない。
- Stage 0 は保存生成物を使う 0-model / 0-booster oracle headroom 監査とする。
- Stage 1 は 1 variant × 1 LightGBM config × 5 outer folds = 5 CPU boosters。
  GPU、control再学習、PF/HMM/Beam再生成は 0 とする。
- Stage 0 実装、Stage 0 Kaggle実行、Stage 1 実装、Stage 1 Kaggle実行、
  current-test inference、submission はそれぞれ別の明示承認を必要とする。
- `docs/06_reproducibility.md` に従い、input、fold、feature、rate basis、target、
  model、OOF prediction の SHA を記録する。

## 受け入れ基準

- backlog、steering 3文書、実験scaffold、config、README、SESSION_NOTES、
  result、metrics、experiment summary が design-only 状態で整合する。
- target basis、符号、単位、first-row anchor、K16 boundary interval assignment、
  feature allowlist、fold、model、実行量、停止条件が一意である。
- config 上で実装、Notebook採用、Kaggle実行、推論、提出がすべて無効である。
- Stage 0 technical gate は `3,783,989 rows / 773 wells / 12,368 segments`、
  全 well で rate basis rank 16、first-row correction 0、finite coverage 1.0 を要求する。
- Stage 0 scientific gate は oracle continuous-rate correction が exp226
  `9.4271095966`を`1.00 ft`以上改善し、5/5 foldsで`0.50 ft`以上改善することを要求する。
- Stage 1 は exp228 `8.944085501`を`0.05 ft`以上改善し、exp226比4/5 folds改善、
  near/1000+/hidden-like/segment境界/by-well p95非悪化、worst-well
  `<=+0.25 ft`をすべて満たす場合だけ科学的PASSとする。
- Stage 1 が PASS しても自動的に推論や提出へ進めない。
- gzip生成物はraw gzip SHAとdecompressed content SHAを分け、後者を主証拠とする。

## 次

Stage 0実装はユーザーの別承認待ちとする。承認前はsource、test、Notebook、
Kaggle packageを作成しない。

## 2026-07-27 実装承認追記

ユーザーの「exp418を実装してください」を、Stage 0 / Stage 1 の compact train
候補と専用 contract test の実装承認として扱う。Kaggle package、正規Notebook
採用、Stage 0 run、Stage 1 run、推論、提出は承認範囲に含めない。

追加の受け入れ基準:

- Jupytext percent形式のcompact self-contained train候補と対応する`.ipynb`がある。
- exp333保存nested predictionをSHA manifest経由で検証し、exp226 fit/regenerationを
  呼ばない。
- Stage 0は0 booster、Stage 1は1 variant ×1 config ×5 folds = 5 CPU boosters。
- Stage 1はSHA固定されたStage 0 PASS summaryがなければfail closedとする。
- K16 basis、destination-row interval、first-row correction 0、float64 lstsq、
  matrix/逐次integration parity、late truth、feature allowlist、scope/tail AND gateに
  専用testがある。
- 正規train Notebookは既存placeholderを上書きせず、候補だけを生成する。
