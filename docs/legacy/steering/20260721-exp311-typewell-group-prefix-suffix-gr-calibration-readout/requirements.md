# 要件

## 仮説

同一Type Well群のbias/noise/reliability統計は、well IDやouter-valid suffix truthを使わずにpeer wellから転送できる。

## 依頼

Train の TVT 正解値を outer-train 内だけで使い、同一 Type Well 群に共通する horizontal GR↔Type Well GR の校正量・ノイズ量が、held-out well の未知 suffix へ転送できるかを監査する。今回は compact self-contained notebook、fold-safe readout、negative control、SHA、固定 gate まで実装し、Kaggle 実行はしない。

## 制約

- Route: `pf_beam`。予測器ではなく GR observation model の基礎監査である。
- 主群は `native_overlap_1`、`exact_typewell_content_sha` は感度分析だけに使う。
- outer-valid well の TVT・suffix error は統計量を凍結するまで参照しない。
- 行数の長い well に支配されないよう well 等重みとする。
- GR 補正値の直接投入、HMM/PF/Beam、ML 学習、inference、submission は禁止する。
- decoderを作らないため、suffix reconstructionの単位はTVTのftではなくhorizontal GR API unitとする。

## 受け入れ基準

- 5-fold same-group holdout、leave-one-group-out、spatial/typewell-purged の3面を定義する。
- slope/interceptだけでなく bias、MAD scale、fit RMSE、supportを出力する。
- group shuffle と horizontal GR circular shift の negative control を固定する。
- `config.yaml` のpromotion gate、実行量、leakage境界が文書と一致する。
- compact train/inference Jupytext sourceと変換済みnotebookが静的検証を通る。

## 変更点と次

設計-only scaffoldからtrain-side zero-booster readoutへ変更する。Kaggle実行、inference、submissionは今回の範囲外とする。
