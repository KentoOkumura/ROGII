# 設計

## アプローチ

exp077 で実装した fixed inference postprocess hook を再利用し、実験ディレクトリと Kaggle kernel を exp084 として分離する。実行対象は inference notebook のみで、exp073 の saved booster manifest と regenerated raw test features から `pred_delta_raw` を作り、`pf_confidence_residual_clip_q995` を適用して submission を生成する。

## 実験範囲

- 対象実験: `exp084_pf_confidence_residual_clip`
- Route: `ml_model`
- 親実験: `exp077_full_replay_postprocess_guard`
- 変更する変数: `inference.postprocess_policy=pf_confidence_residual_clip_q995`
- 固定する変数: exp073 saved boosters、selected mode/model、PF seeds/particles、`residual_clip_limit=66.5908203125`

## 再現性設計

- seed policy: stable sha256 per well
- stochastic 処理の有無: PF/Beam/likelihood-PF replay が stochastic seed bag を使うが、seed は stable に固定する
- PF/Beam / likelihood-PF / seed bagging の有無: あり、`pf_seeds=128`、`pf_particles=500`
- 並列処理と乱数の関係: per-well stable seed を使い、global RNG 依存にしない
- CPU/GPU runtime と deterministic flags: saved booster inference は exp073 manifest を使用し、feature generation は `use_gpu=auto`
- train cache / test feature regeneration の SHA 記録方針: raw gzip SHA と decompressed content SHA を記録する
- model manifest / prediction / submission SHA 記録方針: manifest SHA、prediction SHA、submission SHA を `SESSION_NOTES.md` と `metrics.json` に記録する
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` と generated package の JSON / py_compile を確認する

## リスク

- リークリスク: hidden target は使わない。clip limit は exp073 OOF target_delta 由来の固定値として扱う。
- CV/LB 不一致リスク: exp077 OOF では best policy ではないため、LB 改善は期待しすぎない。
- ランタイム/メモリリスク: exp077 inference と同等。raw test feature replay が主なコスト。
- 再現性リスク: Kaggle bootstrap と PF/Beam seed bag の SHA 記録で監査する。
