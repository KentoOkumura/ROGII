# 設計

## アプローチ

`exp189_denoised_gr_pfbeam_generation_audit` の scoped PF/Beam audit を親にし、GR smoothing 軸を affine calibration 軸へ置き換える。scoring surface は exp072 の `TVT_input_missing_equivalent_exp063_rows` を使い、target wells は true error を使わず eval row 数と md_since coverage で最大 64 wells を選ぶ。

affine observation は、eval zone より前の known prefix だけで `horizontal_GR ~= a * typewell_GR(TVT_input) + b` を robust fit する。PF/Beam の観測costでは calibrated horizontal GR `(GR - b) / a` と raw typewell GR を比較する。fit guard に失敗した well/variant は raw observation へ fallback し、fallback reason を残す。

P0-A backlog の効果分解に合わせ、transition は classic と weak prefix structural の2種を用意する。prefix structural は `TVT_input + Z` を prefix surface として MD に対して robust fit し、eval row の `Z` から expected TVT を作る。これは hard window ではなく soft cost とし、P0-B本体の探索ではなく P0-A 比較用の固定弱priorとして扱う。

## 実験範囲

- 対象実験: `exp211_affine_calibrated_gr_observation_pfbeam`
- Route: `pf_beam`
- 親実験: `exp189_denoised_gr_pfbeam_generation_audit`
- 参照実験: `exp072_exp063_full_replay_feature_cache`、`exp170_heel_calibrated_shift_scan_pfbeam_audit`
- 変更する変数: observation kind (`raw` / `affine_calibrated`) と transition kind (`classic` / `prefix_structural`)
- 固定する変数: target well selection、PF particles/seeds/noise、likelihood temperature、Beam width/move radius/cost、score rows、reference exp072 candidates
- 実行予定: active variant 4、LightGBM config 0、fold 0、booster 0、control/親実験の再学習なし

## 再現性設計

- seed policy: `stable_sha256_per_query_well_seed_index_shared_across_observation_variants`
- stochastic 処理の有無: PF particle propagation / resampling あり。Beam は deterministic。
- PF/Beam / likelihood-PF / seed bagging の有無: PF 240 particles x 8 seeds / variant、Beam top1 / variant。LightGBM seed bagging はなし。
- 並列処理と乱数の関係: 初回 audit は sequential。各 PF seed は experiment name、well、seed index から stable SHA256 で生成し、global RNG に依存しない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU false、internet false。
- train cache / test feature regeneration の SHA 記録方針: exp072 input cache SHA と row candidate gzip raw/decompressed SHA を `metrics.json` に記録する。test regeneration はこの実験では行わない。
- model manifest / prediction / submission SHA 記録方針: model、prediction、submission は生成しない。row candidates と summary の SHA のみ記録する。
- Kaggle package bootstrap 確認方針: push 前に `prepare-kaggle-notebooks --strict` を使い、metadata と bootstrap に `config.yaml`、helper、train/inference `.py`、`settings.py` が入ることを確認する。

## リスク

- リークリスク: calibration fit と structural fit が known prefix のみであることを guard する。evaluation target は scoring と oracle diagnostic 以外に使わない。
- CV/LB 不一致リスク: train pseudo-tail diagnostic であり、raw test behavior は未確認。positive でも submit へ直行しない。
- ランタイム/メモリリスク: exp189 の 3 variants から 4 variants へ増えるため runtime はおおむね 1.3倍。target wells は最大64に制限する。
- 再現性リスク: PF stochastic なので deterministic submission anchor ではない。per-well stable seeds と decompressed SHA を記録する。
