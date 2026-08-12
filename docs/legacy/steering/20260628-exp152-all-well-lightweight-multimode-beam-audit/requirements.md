# 要件

## 依頼

`all_well_lightweight_multimode_beam_audit` を実装する。

## 背景

`exp143_multimode_pfbeam_local_correlation_audit` は 6 well / 12,000 rows の scoped audit で、best multimode `multimode_pf_zacc_s010_a020_noise050_best_lik_seed` が従来 `exp072_beam_mean` を RMSE 70.297647 -> 60.763085 に改善した。一方、full 773 well run は heavy diagnostic のため timeout した。

## 制約

- Route: `pf_beam`
- 親実験: `exp143_multimode_pfbeam_local_correlation_audit`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- ML 再学習、推論 port、提出はしない。
- exp072 cache は上書きしない。
- 全 train well を対象にするが、各 well は tail 500 rows に制限する。
- exp143 の local correlation、mode entropy row-level 詳細、candidate_long 全量保存は避ける。
- 再現性: `docs/06_reproducibility.md` に従い、PF/Beam stochastic seed、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp152_all_well_lightweight_multimode_beam_audit/` に config、実装、train/inference notebook、記録ファイルがある。
- `config.yaml` の `experiment.route` は `pf_beam`。
- Kaggle train package は CPU-only / internet disabled / exp072 cache source mounted で作成できる。
- train-side output は candidate metrics、by-well metrics、bucket metrics、minimal candidate_wide、summary JSON を保存する。
- summary JSON には exp072 cache SHA、入力 raw file SHA、生成物 SHA、gzip decompressed SHA が記録される。
- 主指標は `exp072_beam_mean` との差分で、`likpf_mean` / `pf_z` 超えは必須条件にしない。
