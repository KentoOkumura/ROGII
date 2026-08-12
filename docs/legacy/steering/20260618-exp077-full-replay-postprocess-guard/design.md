# 設計

## アプローチ

exp073 の OOF prediction を読み、`pred_delta` に対する固定後処理 policy を比較する。exp072 full replay train feature cache が利用できる場合は、`md_since`、`eval_len`、`tw_range`、`pf_ancc_std`、`beam_std_d`、`likpf_mean_d` などの target-free geometry / confidence 特徴を id join して guard 条件に使う。

LightGBM feature importance は、exp073 train source の saved booster manifest から fold/model ごとの gain / split importance を復元する。feature ごとに fold/model 平均と標準偏差を集計し、matplotlib の horizontal bar plot を生成して notebook 上に表示する。

## 実験範囲

- 対象実験: `exp077_full_replay_postprocess_guard`
- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- 変更する変数: exp073 OOF prediction に対する固定後処理 policy
- 固定する変数: exp073 full replay feature surface、exp073 saved boosters、exp072 deterministic train cache

## 再現性設計

- seed policy: 新規 stochastic 処理なし。親 exp073 / exp072 の stable SHA256 per-well seed policy を継承して参照する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: exp077 train audit では新規生成しない。inference port を行う場合のみ exp073 の deterministic regeneration path を使う。
- 並列処理と乱数の関係: 後処理は deterministic vector operation のみ。
- CPU/GPU runtime と deterministic flags: 後処理は CPU で deterministic。optional retrain path は exp073 と同じ LightGBM deterministic flags を保持する。
- train cache / test feature regeneration の SHA 記録方針: exp073 OOF prediction file SHA、exp072 feature cache SHA、exp073 model manifest SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: postprocess policy ごとの prediction SHA を保存する。inference port 時は submission SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` 後に notebook JSON と metadata を確認する。

## リスク

- リークリスク: same-OOF 上で policy を選ぶと過適合しやすい。採用判断は固定 policy の inference port と LB / hidden-safe 証拠が必要。
- CV/LB 不一致リスク: global blend や hard switch は避け、効果が小さい保守 policy に限定する。
- ランタイム/メモリリスク: OOF prediction と feature cache は大きい。feature join は必要列のみ読む。
- 再現性リスク: gzip raw SHA は揺れる可能性があるため、親実験同様に content SHA を優先する。
