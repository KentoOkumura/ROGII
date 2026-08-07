# 要件

## 依頼

波形が似た別 well を analog donor とし、その donor の正解 TVT 区間に含まれる
GR マッチング内容（深度方向のシフト、伸長、収縮、局所 warp）を query well に移す
0-model readout を設計する。

このターンでは設計を確定し、backlog、steering、実験 scaffold を作る。実装、学習、
推論、Kaggle 実行、提出は行わない。

## 仮説

同じ typewell group に属する well のうち、未知 suffix の GR 波形が似ている donor は、
query well の真の `MD -> TVT` 増分も似ている。GR だけで選んだ donor の真の TVT
増分を query の最終既知 TVT に再アンカーすれば、group 平均 prior より個体差を
保持した TVT candidate を作れる。

## 実験境界

- 対象: `exp423_same_typewell_gr_dtw_truth_warp_transfer_readout`
- Route: `pf_beam`
- 親実験: `exp109_typewell_neighbor_prior_features`
- 参照実験: `exp065`（same-typewell group）、`exp099`（5-fold pseudo-tail）、
  `exp119`（nearest-well leakage guard）、`exp230`（late truth join）、
  `exp282`（GR alignment readout）
- Stage 0 の train-side OOF readout のみを対象とする。
- LightGBM、PF/HMM/Beam、selector、test inference、submission は対象外とする。
- query の未知 suffix GR は観測可能、query の未知 suffix TVT は観測不能とする。
- donor の正解 TVT は outer-train well に限り利用可能とする。

## 固定要件

- `exp099` / `exp109` と同じ 5-fold pseudo-tail inventory と outer fold を再利用する。
- donor pool は query と同じ `native_overlap=1` typewell group かつ outer-train well
  のみに限定する。
- donor 検索は query GR と donor GR のみで行い、donor TVT は検索順位に使わない。
- query/outer-valid の suffix TVT は candidate、donor 順位、support、fallback、
  control が freeze されるまで読まない。
- donor/query well 集合の交差は 0 とし、各 fold で機械検証する。
- top-K は 5 に固定し、ハイパーパラメータ探索や結果を見た rescue grid は行わない。
- primary candidate は top-5 donor path の row-wise median に固定する。
- top-1、per-well oracle-best top-5、安定 random donor は診断用とし、primary の
  差し替えには使わない。
- tie-break は `(DTW cost asc, donor well_id asc)` に固定する。

## 受け入れ基準

- steering 3 文書、実験 scaffold、`config.yaml`、`README.md`、
  `SESSION_NOTES.md`、`result.md`、`metrics.json`、backlog、実験一覧が整合する。
- fixed preprocessing、DTW 制約、warp 転写式、fallback、対照群、late join、
  成功/停止条件が文書化され、実装者の追加判断を必要としない。
- `experiment.route=pf_beam`、親実験、5-fold、seed、0-model、CPU-only が設定に残る。
- 実装用 notebook は scaffold placeholder のままとし、コードは追加しない。
- deterministic artifact として昇格させる場合の input/schema/content/prediction SHA
  と再実行一致方針が定義される。
- gzip 生成物を比較する場合は raw `.csv.gz` SHA ではなく decompressed content SHA
  を主証拠として記録する。

## 今回の非目標

- 現在の test well に donor transfer を適用すること。
- analog path を既存 PF/Beam candidate set に追加すること。
- GR similarity を学習すること、または真値で donor selector を学習すること。
- hard clustering 自体を目的にすること。
- baseline/control を再学習すること。

## 2026-07-28 実装追補

上記の「このターンでは設計のみ」は初回設計依頼時の境界である。後続のユーザー依頼
`exp423を実装してください` により、固定済み Stage 0 の compact self-contained
Jupytext source/notebook、専用 test、target-free freeze、metric/gate readout の実装を
追加承認した。正規 notebook 採用、Kaggle package/run、test inference、submission、
baseline/control 再学習は引き続き非目標とする。

## 2026-07-28 実行追補

後続のユーザー依頼 `実行してください` により、正規 train notebook 採用、
Kaggle package、CPU audit の push / 実行を追加承認した。正規 inference notebook、
test inference、submission、baseline/control 再学習は承認範囲外のままとする。

Kaggle version 2 を初回有効 run、version 3 を独立 rerun として完了した。
logical content SHA は一致したが、support coverage、oracle、primary、hard-bucket、
tail gate が FAIL した。固定分岐規則に従い、same-typewell donor truth-warp
transfer 仮説を parameter rescue なしで閉じる。
