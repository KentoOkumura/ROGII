# タスクリスト

## TODO

- なし。

## 追加作業に明示承認が必要

- exp423 の再開または追加 Kaggle run。
- 別実験での GR quality feature 化。
- PF/Beam candidate 統合、inference、submission。

## 進行中

- なし

## ブロック中

- なし。

## 完了

- [x] steering を scaffold より先に作成した。
- [x] `exp423_same_typewell_gr_dtw_truth_warp_transfer_readout` scaffold を作成した。
- [x] Route を `pf_beam`、親を `exp109_typewell_neighbor_prior_features` に固定した。
- [x] 0-model Stage 0 と非目標を固定した。
- [x] fold、donor pool、GR preprocessing、DTW、warp 転写式を固定した。
- [x] query truth late join と donor/query 分離 audit を固定した。
- [x] primary、oracle、negative control、baseline を固定した。
- [x] technical/scientific gate と fail-closed 分岐を固定した。
- [x] `docs/06_reproducibility.md` に沿う SHA / deterministic 方針を記録した。
- [x] backlog と実験一覧へ設計-only状態を記録した。
- [x] ユーザーの明示依頼後に Jupytext compact self-contained source を作成した。
- [x] fixed GR preprocessing、constrained DTW、truth-warp transfer を実装した。
- [x] fold separation、late truth join、row identity、support/fallback audit を実装した。
- [x] target-free freeze と raw/decompressed/logical SHA 保存を実装した。
- [x] fixed candidate/control と overall/fold/bucket/by-well readout を実装した。
- [x] 別名 compact self-contained notebook へ変換した。
- [x] 専用 unit test、構文、F821、Jupytext round-trip、experiment validation を確認した。
- [x] 実行承認を記録し、compact self-contained 版を正規 train notebook に採用した。
- [x] CPU / internet off の canonical Kaggle package を作成した。
- [x] Kaggle version 2 の初回有効 run を完了した。
- [x] 初回 logical content SHA を固定し、version 3 の独立 rerun 一致を確認した。
- [x] technical/scientific gate を固定閾値の AND で評価した。
- [x] oracle 不合格の分岐規則に従い、truth-warp transfer branch を閉じた。
- [x] `result.md`、`metrics.json`、`experiment_summary.md`、
  `KAGGLE_DIRECTION.md` を更新した。
