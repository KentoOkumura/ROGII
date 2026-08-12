# 設計

## アプローチ

exp188 の実装を親に、exp183 OOF best-Viterbi selected path から作る selector confidence features を再利用する。差分は active variant の feature group 構成だけに絞り、exp148 の `learned_likelihood_confidence` group を外して `exp183_selector_confidence` に置換する。

学習は exp188 と同じ GPU LightGBM family 3 configs / GroupKFold 5 folds。比較は historical exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960 と exp188 add-only CV 8.539573790 に対して行う。

## 実験範囲

- 対象実験: `exp194_exp183_selector_confidence_replacement_only_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- selector 親: `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector`
- 変更する変数: active feature group から `learned_likelihood_confidence` を外し、`exp183_selector_confidence` を入れる。
- 固定する変数: exp072 full replay cache、exp092 U-projection / disagreement feature logic、exp183 selected variant、GroupKFold split、LightGBM config family、GPU deterministic flags。
- 初期範囲外: current-test exp183 selector feature generation、inference port、submission、direct TVT replacement、blend、postprocess、hard gate。

## 再現性設計

- seed policy: GroupKFold seed 42、LightGBM config seeds は exp063 family を踏襲する。
- stochastic 処理の有無: 新規 feature generation には乱数なし。upstream exp072 / exp145 / exp183 artifacts は固定入力として扱う。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成しない。exp072 / exp183 の保存済み結果を読む。
- 並列処理と乱数の関係: LightGBM は `deterministic=true`、`force_col_wise=true`、`n_jobs=8`、`num_threads=8`。
- CPU/GPU runtime と deterministic flags: Kaggle GPU、`gpu_use_dp=true` を使う。deterministic submission anchor とは扱わない。
- train cache / test feature regeneration の SHA 記録方針: train 実行後、gzip は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: train 実行後、model manifest SHA と OOF prediction SHA を `SESSION_NOTES.md` / `metrics.json` に記録する。初期実装では submission SHA は不要。
- Kaggle package bootstrap 確認方針: prepare 後、kernel sources、GPU/internet metadata、bootstrap 内 `config.yaml` の active variant と feature groups を確認する。

## リスク

- リークリスク: exp183 OOF selector path は upstream train-side selected artifact。`true_tvt`、`abs_error`、`oracle_candidate`、`oracle_label` は downstream feature から除外する。
- CV/LB 不一致リスク: train-side positive でも current-test selector feature parity が未実装なので、そのまま submit しない。
- ランタイム/メモリリスク: exp188 v1/v2 はピークメモリで落ちたため、replacement-only では active variant が要求しない `ll_*` columns を `full_frame` に attach しない。
- 再現性リスク: GPU LightGBM と upstream artifacts を使うため deterministic anchor にはしない。採用候補化する場合は Kaggle kernel version と SHA を記録する。
