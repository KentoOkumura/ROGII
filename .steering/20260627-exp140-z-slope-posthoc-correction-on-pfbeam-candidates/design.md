# 設計

## アプローチ

exp083 の plot で見えた Z-driven な TVT 変化を、PF/Beam 再生成ではなく既存候補への低頻度 posthoc 補正として反証する。

`target_slope = -dZ/dMD`、`base_slope = d(candidate)/dMD` とし、`slope_gap = target_slope - base_slope` を well 内で平滑化する。評価 row の MD step で累積した correction を作り、near prefix guard と target-free gate を掛けて `pred = base + alpha * gate * clipped_correction` を比較する。

## 実験範囲

- 対象実験: `exp140_z_slope_posthoc_correction_on_pfbeam_candidates`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 比較親: `exp083`, `exp100`, `exp104`, `exp106`, `exp126`
- 変更する変数:
  - base candidate: `likpf_mean`, `pf_ancc`, `beam_mean`
  - alpha: `0.10`, `0.20`, `0.35`, `0.50`
  - correction clip: `10`, `20`
  - `abs(dZ/dMD)` threshold
  - candidate disagreement threshold
  - auxiliary mode: `none`, `pfz_agree`, `pfz_pull`
- 固定する変数:
  - exp072 feature cache
  - PF/Beam/likelihood-PF candidate generation
  - train well pseudo-tail scoring surface
  - no inference / no submission

## 再現性設計

- seed policy: 新規乱数なし。
- stochastic 処理の有無: exp140 内は deterministic posthoc grid のみ。上流 exp072 PF/Beam cache は既存生成物として扱う。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。既存 `pf_ancc`, `beam_mean`, `likpf_mean`, `pf_z` を読む。
- 並列処理と乱数の関係: 並列処理なし。
- CPU/GPU runtime と deterministic flags: CPU notebook、GPU 無効。
- train cache / test feature regeneration の SHA 記録方針: exp072 input cache SHA、decompressed SHA、schema SHAを summary JSON に記録する。
- model manifest / prediction / submission SHA 記録方針: model / submission なし。OOF gzip は raw SHA と decompressed SHA を記録する。
- Kaggle package bootstrap 確認方針: `make prepare-kaggle-notebooks --strict` 実行後、metadata と bootstrap support file を validate する。

## リスク

- リークリスク: true TVT を gate に使うと漏洩する。実装では true TVT は metrics のみに限定する。
- CV/LB 不一致リスク: train-side posthoc で良くても hidden raw-test parity が未確認。改善しても直接 submit しない。
- ランタイム/メモリリスク: 3.8M rows に数十 variant を作るためメモリ使用が増える。OOF 保存は上位 variant に限定する。
- 再現性リスク: exp072 cache が Kaggle input に mount されない場合は実行不能。kernel source に exp072 output を明示する。
