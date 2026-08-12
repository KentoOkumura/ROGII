# 設計

## アプローチ

exp148 の既存 feature surface を維持し、target-free GR shift-scan confidence feature を add-only で追加する。

exp167 の結果から、FFT denoise ではなく rolling median / Savitzky-Golay smoothing の surface sharpness に限定する。exp171 の posterior candidate は direct replacement として弱いため、posterior p / entropy / posterior-minus-candidate を ambiguity feature としてだけ使う。

## 実験範囲

- 対象実験: `exp190_denoised_calibrated_matching_features_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 参照実験: `exp167_fft_denoised_gr_matching_audit`, `exp170_heel_calibrated_shift_scan_pfbeam_audit`, `exp171_bimodal_posterior_pfbeam_candidate_audit`
- 変更する変数: `denoised_calibrated_matching` feature group の追加
- 固定する変数: exp072 cache、exp145 learned likelihood features、exp148 control baseline、LightGBM config family、GroupKFold by well

## Feature 設計

- raw train の horizontal well / typewell GR を読む。
- known `TVT_input` prefix から固定 slope prior を作る。
- raw / rolling median / Savitzky-Golay GR に対して、固定 shift grid の matching cost を計算する。
- full-row coverage は 16 row 間隔の deterministic scan grid を well 内補間して作る。
- 追加列は cost/gap/entropy/top1-top2 shift/posterior p/posterior entropy/posterior-minus-likpf/raw-vs-smoothed movement/candidate spread/distance interaction/prefix backtest quality に限定する。

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
- CV/LB 不一致リスク: GR matching surface は train-side diagnostic で弱い。global CV が小幅 positive でも near-row / worst-well / hidden-like / current-test parity を確認するまで submit しない。
- ランタイム/メモリリスク: full-row direct scan は重いため 16 row stride + well 内補間を使う。LightGBM は単一 notebook で 15 boosters。
- 再現性リスク: GPU LightGBM は deterministic anchor ではないため、submission anchor 扱いしない。
