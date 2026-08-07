# 設計

## 方針

`heel_calibrated_shift_scan_pfbeam_audit` とは異なり、GR likelihood は変えない。既に生成された PF/Beam TVT candidate の well 固有 offset を、observed `TVT_input` prefix から推定して posthoc に補正できるかを確認する。

## 実装

1. exp072 fixed candidate cache から tail rows と `pf_ancc` / `beam_mean` / `likpf_mean` / `sc_ens` / `hyb` を読む。
2. raw train horizontal の known prefix 末尾 `prefix_holdout_rows` を一時的に NaN にして、既存 public replay helper で PF/Beam candidate を再生成する。
3. holdout prefix 上で `candidate_tvt - TVT_input` の median / Huber / recent median / IQR / slope を candidate 別、well 別に計算する。
4. tail candidate へ `candidate - alpha * capped(offset)` を distance fade-in 付きで適用する。IQR が大きい、prefix rows が不足する well は補正しない。
5. RMSE / MAE / within10 / bucket / group / by-well / max regression を保存する。

## 再現性

- Prefix replay は `public_notebook_replay_audit.py` の stable per-well seed を使う。
- 新規 supervised model はない。
- `metrics.json` と summary JSON に input cache SHA、prefix offset SHA、OOF gzip decompressed SHA を記録する。
