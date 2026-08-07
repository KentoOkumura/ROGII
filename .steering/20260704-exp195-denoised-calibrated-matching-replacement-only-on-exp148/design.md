# 設計

## アプローチ

exp190 add-only は `lgb_mean` では exp148 をわずかに下回ったが、`lgb1` 単体は exp148 同 config を改善した。DCM signal と exp145 learned likelihood confidence が重複・競合した可能性を切り分けるため、今回は追加ではなく full block replacement として評価する。

exp148 の base 196 features、`projection_correction`、`u_disagreement` は維持し、`learned_likelihood_confidence` の `ll_*` block をモデル特徴から外す。代わりに exp190 と同じ target-free DCM feature group を入れる。

## 実験範囲

- 対象実験: `exp195_denoised_calibrated_matching_replacement_only_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 参照実験: `exp167_fft_denoised_gr_matching_audit`, `exp170_heel_calibrated_shift_scan_pfbeam_audit`, `exp171_bimodal_posterior_pfbeam_candidate_audit`, `exp190_denoised_calibrated_matching_features_on_exp148`
- 変更する変数: active model feature group から `learned_likelihood_confidence` を外し、`denoised_calibrated_matching` を入れる
- 固定する変数: exp072 cache、base 196 features、U-projection/disagreement、LightGBM config family、GroupKFold by well、DCM generator config

## Feature 設計

- raw train の horizontal well / typewell GR を読む。
- known `TVT_input` prefix から固定 slope prior を作る。
- raw / rolling median / Savitzky-Golay GR に対して、固定 shift grid の matching cost を計算する。
- full-row coverage は 16 row 間隔の deterministic scan grid を well 内補間して作る。
- DCM 列は cost/gap/entropy/top1-top2 shift/posterior p/posterior entropy/posterior-minus-likpf/raw-vs-smoothed movement/candidate spread/distance interaction/prefix backtest quality に限定する。
- DCM の top1/top2/posterior TVT は model feature の差分・不確実性としてのみ使い、直接予測値として採用しない。

## 再現性設計

- seed policy: LightGBM GroupKFold seed fixed。GR shift-scan feature generation に乱数は使わない。
- stochastic 処理の有無: 新規 feature generation は deterministic。GPU LightGBM は stochastic component として記録する。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規 PF/Beam 再生成なし。exp072 fixed cache を読む。
- 並列処理と乱数の関係: feature generation は per-well deterministic。LightGBM は `deterministic=true`, `force_col_wise=true`, `n_jobs/num_threads=8`。
- CPU/GPU runtime: Kaggle GPU train。`gpu_use_dp=true`。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を主証拠にする。初期実装は train-side only。
- model manifest / prediction / submission SHA 記録方針: train 完了時に model manifest SHA、prediction SHA を記録する。submission SHA は初期対象外。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --strict` と `validate_experiment.py` で確認する。

## リスク

- リークリスク: prefix backtest で hidden-tail true TVT を使わないことを守る。feature source に oracle / abs error を混ぜない。
- CV/LB 不一致リスク: exp160/exp162 は train-side positive でも LB negative だった。global OOF が positive でも near-row / worst-well / hidden-like / current-test parity を確認するまで submit しない。
- ランタイム/メモリリスク: full-row direct scan は重いため 16 row stride + well 内補間を使う。LightGBM は単一 notebook で 15 boosters。
- 再現性リスク: GPU LightGBM は deterministic anchor ではないため、submission anchor 扱いしない。
