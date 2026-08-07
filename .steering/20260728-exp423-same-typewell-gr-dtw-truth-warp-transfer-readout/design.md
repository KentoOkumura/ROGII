# 設計

## 1. 結論

hard cluster を新たに学習するのではなく、既存の same-typewell group を donor pool の
安全な境界として使い、その内側で GR constrained-DTW による soft kNN を行う。
donor の正解 TVT から得た局所 warp を query の最終既知 TVT へ再アンカーし、
個々の analog path と top-5 median path を作る。

Stage 0 は「似た GR の donor に移送可能な TVT warp が存在するか」と「GR 距離だけで
良い donor を選べるか」を分離して読む 0-model OOF 診断である。

`exp119` は query の局所 GR window を donor の既知 prefix window に対応付け、
`TVT_input` 由来の slope/path を弱い `alpha=0.1` 補正として使ったが negative だった。
exp423 はその parameter rescue ではない。query と donor の未知 suffix 全体を対応付け、
outer-train donor の suffix 正解 TVT warp 自体に移送可能性があるかを先に oracle/control
付きで問う。exp109 の group 平均 prior が positive だったこととの間を原因分離する。

## 2. 実験範囲

- 対象実験: `exp423_same_typewell_gr_dtw_truth_warp_transfer_readout`
- Route: `pf_beam`
- 親実験: `exp109_typewell_neighbor_prior_features`
- 比較対象:
  - `exp109` same-typewell group-average prior / selected correction
  - `exp099` likelihood-PF OOF
  - stable random same-group donor
- 変更する変数: group 内 donor を平均ではなく GR 類似度で選び、donor 固有の真の
  TVT 増分を query へ転写する。
- 固定する変数: pseudo-tail 行、5-fold、同一 group 定義、query anchor、評価指標、
  baseline prediction、bucket 定義。
- 実行対象: outer-valid train wells の OOF readout のみ。
- 実行対象外: ML、PF/HMM/Beam、test inference、Kaggle submission。

## 3. 入力と fold contract

### 3.1 行 inventory

- query 行と fold は `exp099` / `exp109` の pseudo-tail OOF inventory を再利用する。
- row identity は少なくとも `(fold, well_id, row_idx)` で一意にする。
- 各 query well の anchor は pseudo-tail 直前の最後の既知 TVT とする。
- suffix progress `s` は最初の未知行を 0、最後の未知行を 1 とする。
- query suffix の GR は入力として使用できる。query suffix の真の TVT は freeze 後の
  評価 join 以外で使用してはならない。

### 3.2 donor pool

fold `f` の query に対する donor は、次をすべて満たす well とする。

1. `fold != f` の outer-train well。
2. query と同じ `exp065` の `native_overlap=1` typewell group。
3. pseudo suffix の finite GR coverage が 70% 以上。
4. robust scale が `1e-6` より大きい。

eligible donor が 2 本未満なら unsupported とする。2–4 本なら全 donor を、5 本以上なら
上位 5 本を使う。unsupported 行の deployable output は固定 baseline をそのまま使うが、
transfer 本体の supported-only 指標へ混ぜない。

## 4. 固定 GR similarity

### 4.1 preprocessing

query と donor の suffix ごとに独立して次を行う。

1. MD 昇順に並べ、重複 row identity を拒否する。
2. finite GR に幅 5 row の centered rolling median を適用する。
3. suffix progress `[0, 1]` 上の 256 点へ線形補間する。外挿はしない。
4. median と `1.4826 * MAD` で robust z-normalize する。
5. 元の finite support mask を 256 点にも伝播し、共通 support が 70% 未満の pair は
   eligible から外す。

この正規化は絶対 GR level ではなく波形形状を比較するためのものとし、forward
orientation のみを許可する。

### 4.2 constrained DTW

- sequence length: 256
- point cost: 共通 finite 点の squared robust-z GR difference
- path: monotone forward
- Sakoe-Chiba band: 32 points（12.5%）
- step: `(1,1)`, `(1,0)`, `(0,1)`
- consecutive horizontal/vertical step 上限: 4
- pair cost: valid path point の mean point cost
- ranking: `(normalized DTW cost asc, donor well_id asc)`
- K: 5

検索順位には query/donor の GR、group、fold、support metadata だけを使う。donor の
TVT、candidate RMSE、query truth は使わない。

## 5. donor truth-warp transfer

DTW path を単調補間し、query progress から donor progress への写像
`psi_q_to_d(s)` を得る。donor の真の TVT curve を donor progress 上で線形補間し、
query anchor `A_q` へ次のように再アンカーする。

```text
TVT_hat_q_from_d(s)
  = A_q
  + TVT_true_d(psi_q_to_d(s))
  - TVT_true_d(psi_q_to_d(0))
```

この差分に donor のシフト、伸長、収縮、局所 warp が含まれる。出力は query の元の
row inventory へ線形補間し、anchor 連続性、finite、row count、一意性を検証する。

## 6. candidate と control

### 6.1 deployable candidate

- `analog_top1`: GR-DTW 1 位 donor の path。selectability 診断用。
- `analog_top5_median`: 利用可能な上位最大 5 donor path の row-wise median。
  **この実験の唯一の primary candidate**。

run 後に top-1 と median の良い方を選ぶことは禁止する。

### 6.2 post-freeze diagnostics

- `analog_top5_oracle_well`: query truth を join した後、上位 5 donor のうち query
  well 全体の RMSE が最小の 1 donor を選ぶ。行単位 oracle は作らない。
- `stable_random_same_group`: eligible donor を `SHA256(query_well_id)` で安定に
  1 本選ぶ negative control。
- `exp109_best_fixed`: `exp109` の固定 best
  `native_overlap_0p999_likpf_mean_corr_a0p2_c40`
  （保存 OOF RMSE `11.143359521`）。
- `exp099_likpf_mean`: 既存 likelihood-PF baseline。

oracle は transferability の上限、top-1/random/DTW 相関は selectability を測る。
oracle を candidate、feature、fallback、提出へ使用しない。

## 7. late truth join と artifact contract

query truth join 前に、次を freeze する。

- fold と row inventory
- eligible/support/fallback 判定
- donor ranking、DTW cost、DTW path
- donor 由来の各 analog path
- `analog_top1`、`analog_top5_median`、stable random control
- config snapshot、input manifest、schema SHA、decompressed content SHA

許可される truth access を分離して監査する。

- donor outer-train truth: analog path materialization に限り freeze 前に使用可。
- query/outer-valid truth: freeze 前の read count は必ず 0。
- freeze 後の query truth: metric、oracle、readout にのみ使用可。

各 fold で `query_well_set ∩ donor_well_set == ∅` を assert し、違反時は即時停止する。

## 8. 評価指標

主指標は score rows の RMSE とし、overall、fold、by-well、`1000+`、既存 hidden-like
spatial/typewell-purged bucket を同じ inventory で報告する。

追加 readout:

- supported well/row coverage と fallback 件数
- group size、donor count、DTW cost、warp length、step/run-length
- donor rank 別 path RMSE
- DTW cost と donor path RMSE の Spearman 相関
- top-1 と stable random の差
- primary、oracle、各 baseline の fold / bucket / by-well delta
- by-well delta の p50/p90/p95/worst

## 9. 事前固定した成功条件

### 9.1 technical gate（すべて必須）

- donor/query well intersection: 全 fold で 0
- query truth read before freeze: 0
- row identity の欠落・重複・順序違反: 0
- supported well fraction: 90% 以上
- supported score-row fraction: 90% 以上
- supported path の finite coverage: 100%
- 同一 input/config の再実行で schema/content SHA が一致

### 9.2 scientific gate（すべて必須）

1. **transferability**:
   `analog_top5_oracle_well` が `exp109_best_fixed` より overall RMSE を 0.30 ft 以上改善し、
   5 fold 中 4 fold 以上で non-worse。
2. **selectability**:
   DTW cost と donor path RMSE の pooled Spearman が 0.15 以上で、5 fold 中 4 fold
   以上で正。
3. **negative control**:
   `analog_top1` が stable random donor より overall RMSE を 0.10 ft 以上改善し、
   5 fold 中 4 fold 以上で non-worse。
4. **deployable primary**:
   `analog_top5_median` が `exp109_best_fixed` より overall RMSE を 0.10 ft 以上改善し、
   5 fold 中 4 fold 以上で non-worse。
5. **hard-bucket safety**:
   primary の `1000+` と hidden-like bucket はともに non-worse。
6. **tail safety**:
   primary の by-well delta p95 は `<= 0.00 ft`、worst は `<= +0.25 ft`。

1–6 の AND を PASS とする。閾値は実行後に変更しない。

## 10. 分岐規則

- technical gate 不合格: invalid。原因を直す場合も同じ設計の実装修正だけを許し、
  similarity/threshold の探索は別実験にする。
- oracle 不合格: donor truth-warp transfer 仮説を閉じ、PF/Beam へ統合しない。
- oracle 合格かつ selectability/primary 不合格:
  `headroom_only_selection_failed`。この実験を昇格せず、selector/ranker は別仮説・
  別実験としてユーザー確認後に設計する。
- 全 gate 合格: 別実験で raw-test parity と既存 candidate set への固定統合を設計する。
  exp423 自体へ inference や PF/Beam を後付けしない。

## 11. 再現性設計

- seed policy: matching 自体は乱数なし。negative control だけ stable SHA256 mapping。
- stochastic 処理: なし。
- PF/Beam / likelihood-PF / seed bagging: 新規実行なし。既存 prediction は read-only。
- 並列処理: 初回実装は single process。将来並列化しても結果順序と tie-break を固定する。
- runtime: CPU-only、AMP/GPU なし。
- artifact: raw input SHA、row inventory SHA、config SHA、feature schema SHA、
  decompressed feature/prediction content SHA を記録する。
- model manifest/model SHA: model を作らないため `not_applicable`。
- submission SHA: submission を作らないため `not_applicable`。
- Kaggle package bootstrap: Stage 0 実行前に requirements と実行環境を manifest 化する。
  現時点では package/install/run を行わない。
- deterministic anchor 昇格: 同一 input/config の独立再実行で logical content SHA が
  一致した場合に限る。

## 12. リスク

- リークリスク: donor pool に outer-valid well が混入する、query truth を donor
  selection に使う、oracle を deployable path と混同する。
- CV/LB 不一致: train pseudo-tail と test の自然な missing suffix、typewell group
  coverage、available donor pool が異なる。Stage 0 PASS だけで test inference へ進めない。
- ドメインリスク: GR 類似性が TVT warp 類似性を意味しない、または同 group 内にも
  多峰性がある。
- DTW リスク: 波形の反復により不適切な位相を対応付ける。band と run-length 上限、
  random control、warp diagnostics で読む。
- runtime/メモリ: well pair 数と DTW matrix。same-group、256 点、band 制約、
  fold 単位 streaming で抑える。
- 再現性: equal cost tie、行順、gzip metadata。stable sort と decompressed content
  SHA で固定する。

## 13. 実装・実行履歴

設計確定時点では未実装だったが、2026-07-28 の後続ユーザー依頼により、別名
compact self-contained Jupytext source/notebook、DTW、target-free freeze、
metric/gate readout、専用 test まで実装した。その後の実行依頼により正規 train
notebook を採用し、Kaggle CPU version 2 / 3 を完了した。logical content SHA は
独立 rerun で一致したが、support coverage、oracle、primary、scope、tail gate は
FAIL した。第10節の分岐規則どおりtruth-warp transferをno-rescueで閉じ、
inference、submissionは行わない。
