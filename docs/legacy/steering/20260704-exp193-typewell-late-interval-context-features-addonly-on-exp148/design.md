# 設計

## アプローチ

exp148 の full-train learned-likelihood LightGBM flow を維持し、typewell 後半区間の well-context feature を add-only で追加する。exp176 で positive だった late-range signal から candidate-long / candidate-specific 部分を外し、`typewell_min/max/span`、late50/60/70 interval、`known_last_pct`、late interval 開始との差分、inside flag だけで小さく反証する。

## 実験範囲

- 対象実験: `exp193_typewell_late_interval_context_features_addonly_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: typewell late-interval context feature group の add-only 追加
- 固定する変数: exp148 base feature surface、exp145 learned likelihood features、GroupKFold by well、LightGBM config family、GPU mode
- 比較対象: exp148 historical、exp174 posthoc clip negative、exp176 selector positive、exp160/162/183 系 exp148 add-only候補

## 再現性設計

- seed policy: GroupKFold seed 42。LightGBM は exp148 系 GPU deterministic config を使う。
- stochastic 処理の有無: 新規 feature generation には乱数なし。学習は GPU LightGBM のみ stochastic component として扱う。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規 PF/Beam 生成なし。既存 exp072 / exp145 cache を読む。
- 並列処理と乱数の関係: typewell context join は deterministic。LightGBM は `gpu_use_dp=true`, `deterministic=true`, `force_col_wise=true`, `n_jobs/num_threads=8`。
- CPU/GPU runtime と deterministic flags: Kaggle GPU train、1 variant x 3 configs x 5 folds = 15 boosters。
- train cache / test feature regeneration の SHA 記録方針: exp072 / exp145 source SHA と raw typewell file set hash を summary / manifest に記録する。
- model manifest / prediction / submission SHA 記録方針: train 後に model SHA、prediction SHA、feature schema を記録する。初期実装では submission SHA なし。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` 後、metadata と bootstrap config が exp193 / typewell late interval / GPU train になっていることを確認する。

## リスク

- リークリスク: hidden-tail true TVT、oracle best、true-error rank、OOF absolute error を feature source に入れない。`known_last_pct` は observed `TVT_input` prefix の最後だけから作る。
- CV/LB 不一致リスク: exp160 は CV positive / LB negative だったため、CV 改善だけで submit しない。raw-test/current-test parity、near-row、worst-well、hidden-like stress を後続で確認する。
- ランタイム/メモリリスク: exp148 + 17 本程度の context feature なので exp148/exp160 と同程度。単一 GPU notebook で実行する。
- 再現性リスク: raw typewell file set と upstream cache に依存するため、summary / manifest に source SHA を残す。
