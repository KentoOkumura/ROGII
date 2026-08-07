# exp423_same_typewell_gr_dtw_truth_warp_transfer_readout

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU audit 完了・gate FAIL・branch 閉鎖
- Stage: 0-model train-side OOF readout
- CV: `14.103812714`（primary `analog_top5_median`）
- Public / Private LB: 対象外
- 作成日: 2026-07-28
- 親実験: `exp109_typewell_neighbor_prior_features`
- steering:
  `.steering/20260728-exp423-same-typewell-gr-dtw-truth-warp-transfer-readout/`

## 仮説

同じ typewell group の中では、未知 suffix の GR 波形が似た donor well の真の
`MD -> TVT` 増分も query well に移送できる。GR constrained-DTW で得た対応を用い、
donor の正解 TVT path を query の最終既知 TVT に再アンカーすれば、group 平均 prior
より個体差を保った candidate を作れる。

## 設計の要点

- hard clustering は作らず、`exp065` の same-typewell group を donor pool guard に使う。
- `exp099` / `exp109` の 5-fold pseudo-tail inventory を再利用する。
- donor は同 group の outer-train well のみ。query/outer-valid well は donor にしない。
- 256 点へ正規化した suffix GR を fixed constrained-DTW で比較し、上位 5 donor を得る。
- donor の真の TVT 増分を query anchor へ転写する。
- primary は上位最大 5 path の row-wise median。top-1 は選択性診断に使う。
- per-well oracle-best top-5 は transferability headroom 専用で、deploy しない。
- query truth は candidate/control/artifact freeze 後にだけ評価用 join する。
- technical/scientific gate は steering と `config.yaml` に事前固定する。

## 検証方針

- Fold: `exp099` / `exp109` と同じ 5-fold pseudo-tail OOF
- Group: query well 単位。donor は outer-train かつ same-typewell group のみ
- Primary: `analog_top5_median` と `exp109_best_fixed` の score-row RMSE 差
- Diagnostics: top-5 per-well oracle、top-1対stable random、DTW cost-error相関
- Leakage check: donor/query交差0、query truthのfreeze前read 0、row identity完全一致
- Guard: overall、4/5 folds、1000+、hidden-like、by-well p95/worst の固定AND

## 今回は行わないこと

- LightGBM、PF/HMM/Beam、selector 学習
- test inference、submission
- 既存 baseline/control の再学習

compact self-contained train notebook を正規 train notebook として採用した。
正規 inference notebook は exp423 の対象外なので scaffold placeholder のまま保持する。
編集元は次の Jupytext source である。

- `exp423_same_typewell_gr_dtw_truth_warp_transfer_readout_compact_selfcontained_train.py`
- `exp423_same_typewell_gr_dtw_truth_warp_transfer_readout_compact_selfcontained_train.ipynb`

実装には fixed GR preprocessing、run-length 制約付き banded DTW、donor truth-warp、
outer-fold 分離、query truth late join、target-free freeze、decompressed/logical SHA、
oracle/random/control readout、technical/scientific gate を含む。

## 所見

Kaggle CPU version 2 / 3 を完走した。primary RMSE は `14.103812714` で、
`exp109_best_fixed=11.143366769` より `2.960445945 ft` 悪化した。top-5
per-well oracle も `12.285086482` で exp109 より `1.141719713 ft` 悪く、
transferability headroom 自体がなかった。top-1 は stable random donor より
`1.233003803 ft` 良かったものの、固定 gate の一部診断だけなので昇格根拠にしない。

support は `286 / 773 wells`、`1,394,464 / 3,783,989 rows` に限られ、
technical coverage gate も FAIL した。logical content SHA は独立 rerun で一致し、
query truth pre-freeze read 0、donor/query intersection 0、input SHA は PASS した。

## 実行入口

Kaggle kernel
`kentookumura/exp423-gr-dtw-truth-warp-readout-train` の version 2 を初回有効 run、
version 3 を独立 rerun として完了した。inference/submission は exp423 の対象外で、
未承認・未実行である。

## 結果

固定 technical/scientific gate は FAIL。oracle 不合格の分岐規則に従い、
same-typewell donor truth-warp transfer 仮説を parameter rescue なしで閉じる。
詳細は [result.md](result.md) に記録した。
