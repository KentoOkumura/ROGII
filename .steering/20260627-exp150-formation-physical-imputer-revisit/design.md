# 設計

## アプローチ

exp138 の ANCC-only surface audit を拡張し、6 formation contact を同時に推定する。推定 surface から Sunny 風の physical TVT 候補を作るが、global contact reference は well ごとの既知 `TVT_input` prefix で offset calibration するための中間量に留める。

候補は次の 3 種類に限定する。

- `contact_median`: 6 contacts の physical TVT 候補の中央値。
- `contact_prefix_weighted`: prefix calibration MAE が小さい contact を重くした平均。
- `contact_best_prefix`: prefix calibration MAE が最小の contact。

## 実験範囲

- 対象実験: `exp150_formation_physical_imputer_revisit`
- Route: `ml_model`
- 親実験: `exp138_ancc_surface_predictability_audit`
- 変更する変数: formation surface を ANCC 単独から 6 contacts へ拡張し、physical TVT 候補と confidence 指標を追加する。
- 固定する変数: GroupKFold by well、CPU-only、LightGBM/PF/Beam なし、提出なし。

## 再現性設計

- seed policy: fixed global seed 42 + fold offset。
- stochastic 処理の有無: `row_knn_xy` の row subsampling のみ。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。
- 並列処理と乱数の関係: parallel RNG なし。subsampling 後の KNN / local plane prediction は deterministic。
- CPU/GPU runtime と deterministic flags: CPU-only。GPU は使わない。
- train cache / test feature regeneration の SHA 記録方針: CSV content SHA を `metrics.json` に記録する。
- model manifest / prediction / submission SHA 記録方針: persistent model と submission はない。OOF feature CSV / metrics CSV SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` を通し、Kaggle train output を正とする。

## リスク

- リークリスク: valid fold formation / true TVT を candidate generation に使うと leakage。実装では scoring-only に分離する。
- CV/LB 不一致リスク: 今回は提出しないため LB とは比較しない。後続 feature 化する場合は exp092/073 の既存 OOF と hidden-like stress で別途確認する。
- ランタイム/メモリリスク: full OOF rows が大きい。wide row-level CSV を 1 つに抑え、per-formation long prediction は保存しない。
- 再現性リスク: row subsampling 以外に乱数を使わない。seed と output SHA を記録する。
