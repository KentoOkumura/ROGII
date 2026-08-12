# 設計

## アプローチ

exp221 は exp148 `lgb_mean` を HMM の Gaussian emission center として使い、sigma は全 row 共通 `20` だった。exp229 では exp148 feature surface 上で LightGBM quantile model を追加学習し、`q16/q50/q84` から `q_mid_tvt` と `sigma_tvt` を作る。HMM は `q_mid_tvt` を center、`sigma_tvt` を row-wise emission sigma として受け取る。

## 実験範囲

- 対象実験: `exp229_lgb_quantile_band_emission_hmm_on_exp148`
- Route: `ensemble`
- 親実験: `exp148`, `exp193`, `exp209`, `exp221`
- 変更する変数:
  - LightGBM objective を `regression` から `quantile` にする。
  - alpha は初回 `0.16/0.50/0.84`。
  - HMM emission sigma を fixed scalar から quantile band 由来の row-wise sigma にする。
- 固定する変数:
  - exp148 feature surface と GroupKFold by well。
  - HMM grid / transition / GR emission は exp221/exp209 系と同じ。
  - 初回の LightGBM config は exp063 family の `lgb1` 1本だけ。
  - 親/control の再学習はしない。

## Notebook 構成

1. `exp229_lgb_quantile_band_emission_hmm_on_exp148_train.ipynb`
   - Kaggle GPU。
   - exp148 surface を組み立て、q16/q50/q84 の OOF と saved boosters を保存する。
   - 15 boosters のコストガードを notebook 上に表示する。

2. `exp229_lgb_quantile_band_emission_hmm_on_exp148_train_aggregate.ipynb`
   - Kaggle CPU。
   - train output の quantile band を読み、crossing 補正済み `q_mid_tvt` / `sigma_tvt` を HMM に渡す。
   - overall、distance bucket、hidden-like、by-well、HMM std calibration、step-delta を保存する。

3. `inference.ipynb`
   - 初回実装では deferred。audit が exp221 fixed-sigma と現行 ML anchor に対して採用候補になった場合だけ同じ exp 内で追加する。

## 再現性設計

- seed policy: GroupKFold は deterministic、LightGBM config seeds は exp063 family を継承。任意の train-row subsampling は local `np.random.default_rng(42)`。
- stochastic 処理の有無: LightGBM GPU 学習は bitwise deterministic と見なさない。HMM には RNG なし。
- PF/Beam / likelihood-PF / seed bagging の有無: この実験では PF/Beam 再生成なし。保存済み exp072 cache を comparison baseline として読む。
- 並列処理と乱数の関係: HMM outer parallel は RNG なし。Numba / thread 浮動小数差は summary に記録する。
- CPU/GPU runtime と deterministic flags: train は T4 GPU、`gpu_use_dp=true`, `deterministic=true`, `force_col_wise=true`, `num_threads=8`。HMM audit は CPU。
- train cache / test feature regeneration の SHA 記録方針: quantile predictions gzip SHA、feature schema、model manifest、HMM cache summary を保存する。gzip content SHA は実行後に必要に応じて補う。
- model manifest / prediction / submission SHA 記録方針: train は model SHA と prediction SHA を記録。submission は inference 実装後のみ。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --notebook train` と `--notebook train_aggregate` 後に package-side py_compile / ruff F821 を通す。

## リスク

- リークリスク: quantile band 幅を true error で直接 fitting しない。HMM lambda/floor/cap は固定グリッドのみ。
- CV/LB 不一致リスク: exp221 は train-side 改善が Public LB に十分転移しなかった。inference は audit 後に判断する。
- ランタイム/メモリリスク: train は 15 boosters に抑える。HMM は lambda 3 variants で exp221 v2 timeout の反省から variant 数を限定する。
- 再現性リスク: LightGBM GPU は完全 bitwise anchor としない。採用候補なら saved booster inference と SHA 記録を追加する。
