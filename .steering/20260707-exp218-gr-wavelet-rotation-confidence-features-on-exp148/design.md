# 設計

## アプローチ

`exp190_denoised_calibrated_matching_features_on_exp148` の exp148 add-only LightGBM
training scaffold を流用し、GR signal feature block を wavelet/rotation confidence block
へ差し替える。

各 well の raw train horizontal GR と typewell GR、known `TVT_input` prefix だけから、
row-local な denoise quality、multi-scale wavelet residual energy、FFT rotation energy、
raw-vs-denoised candidate observation consistency、candidate disagreement interaction を生成する。
追加特徴は `grwr_` prefix に統一し、`gr_wavelet_rotation_confidence` feature group として
exp148 surface に add-only する。

## 実験範囲

- 対象実験: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: GR wavelet / FFT rotation / denoise confidence features の追加有無
- 固定する変数: exp072 replay cache、exp092 U-projection surface、exp145 learned-likelihood feature、GroupKFold by well、LightGBM config family、fold 数、control baseline は保存済み exp148 metrics

## 再現性設計

- seed policy: LightGBM config は既存 exp063 family の fixed seeds を使う。feature generation は deterministic で乱数を使わない。
- stochastic 処理の有無: wavelet / FFT / rolling / Savitzky-Golay / candidate observation-cost feature generation は deterministic。LightGBM GPU 学習だけが環境差で微小に揺れる可能性がある。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成はしない。exp072/exp145 の固定 artifact と既存 candidate columns を feature source として読む。
- 並列処理と乱数の関係: feature generation は sequential well loop。学習は LightGBM の fixed seed / deterministic flags に従う。
- CPU/GPU runtime と deterministic flags: primary は `gpu_repro_guard_dp_threads8`、`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、`n_jobs/num_threads=8`。
- train cache / test feature regeneration の SHA 記録方針: exp072/exp145 入力 SHA、生成 feature schema、OOF prediction SHA、model SHA は Kaggle train output の summary / manifest に保存する。
- model manifest / prediction / submission SHA 記録方針: train で model manifest と OOF prediction SHA を保存する。初期実装では submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --strict` 後、package の `py_compile` と validate を実行する。

## リスク

- リークリスク: GR features は raw horizontal/typewell GR と observed `TVT_input` prefix のみに限定する。hidden-tail true TVT、oracle label、OOF error は使わない。
- CV/LB 不一致リスク: public discussion 由来の GR denoise signal は train CV で小改善しても hidden-like / near-row / worst-well に弱い可能性がある。global RMSE だけで採用しない。
- ランタイム/メモリリスク: per-well wavelet/FFT と candidate observation-cost feature は exp190 より軽い想定だが、全行生成のため重くなる可能性がある。feature 列数を 20-80 列程度に抑える。
- 再現性リスク: LightGBM GPU の bitwise reproducibility は保証しない。採用候補になった場合は CPU deterministic control または rerun 差分確認を検討する。
