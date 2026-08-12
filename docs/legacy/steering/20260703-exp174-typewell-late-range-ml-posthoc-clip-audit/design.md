# 設計

## アプローチ

保存済み ML OOF prediction を固定し、typewell TVT range に対する位置だけを使った posthoc grid を評価する。

1. `exp148` の row OOF prediction を読み、`variant=learned_likelihood_confidence_addonly` / `model=lgb_mean` を primary source とする。
2. raw train/test の horizontal / typewell CSV から well ごとの `typewell_min/max/span` と last known `TVT_input` を計算する。
3. OOF prediction に `pred_pct = (pred_tvt - typewell_min) / typewell_span`、`target_pct`、`known_last_pct`、distance bucket を付与する。
4. `known_last_pct >= threshold` かつ `pred_pct < lower_bound_pct` の行だけ、`pred_tvt + alpha * (lower_bound_tvt - pred_tvt)` に shrink する。
5. baseline、fixed lower bound、`known_last_pct - margin` lower bound を比較し、global RMSE だけでなく changed rows / wells、near-row、longtail、front-half exception、worst-well regression を保存する。

## 実験範囲

- 対象実験: `exp174_typewell_late_range_ml_posthoc_clip_audit`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 参照実験: `exp092_u_projection_correction_disagreement_fullrun`、`exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- 変更する変数: `known_last_pct_min`、lower bound pct / margin、shrink alpha
- 固定する変数: ML OOF prediction、raw/typewell input、validation rows、true TVT scoring

## 再現性設計

- seed policy: no_new_rng_posthoc_grid
- stochastic 処理の有無: なし。既存 OOF prediction に deterministic な grid を適用するだけ。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。上流 exp148 / exp092 の生成物は既存 anchor として読む。
- 並列処理と乱数の関係: `num_workers=1`、global RNG 不使用。
- CPU/GPU runtime と deterministic flags: CPU notebook、GPU 不使用、LightGBM 学習なし。
- train cache / test feature regeneration の SHA 記録方針: source prediction gzip は raw SHA と decompressed SHA、small CSV/JSON は raw SHA を summary JSON に記録する。
- model manifest / prediction / submission SHA 記録方針: model / submission は生成しない。OOF posthoc prediction gzip の raw SHA と decompressed SHA を記録する。
- Kaggle package bootstrap 確認方針: push 前に `prepare-kaggle-notebooks --notebook train --strict` を使い、exp148 train output が kernel source に入ることを確認する。

## リスク

- リークリスク: lower bound grid の選択を true TVT で過剰最適化すると漏洩的になる。結果は no-training diagnostic とし、positive でも raw-test parity / exception well guard を通す。
- CV/LB 不一致リスク: test 3 wells は late range だが、train には前半例外があり target TVT は単調でない。global OOF 小改善だけで submit しない。
- ランタイム/メモリリスク: exp148 prediction は約 3.8M rows x model variants で大きいが、必要列だけを読み、primary model に filter してから grid を回す。
- 再現性リスク: 上流 exp148 OOF 生成物に依存するため deterministic submission anchor とは扱わない。
