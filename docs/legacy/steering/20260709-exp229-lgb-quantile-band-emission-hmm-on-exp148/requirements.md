# 要件

## 依頼

`lgb_quantile_band_emission_hmm_on_exp148` backlog を実験化する。exp221 の fixed-sigma LGB emission HMM を拡張し、LightGBM quantile band から行ごとの予測信頼度を推定して HMM emission uncertainty として使う。

## 制約

- Route: `ensemble`。LightGBM 予測と HMM sequence smoothing の両方が予測生成に本質的に寄与する。
- notebook は分割する。`train` は quantile LightGBM OOF と saved boosters、`train_aggregate` は quantile-band HMM train-side audit を担当する。
- 初回 train は 1 active variant x 1 LightGBM config x 3 quantiles x 5 folds = 15 boosters に限定する。
- exp148 / exp193 / exp221 の control 再学習はしない。保存済み OOF / metrics を比較基準にする。
- HMM lambda / sigma floor / sigma cap は config の固定グリッドだけを使い、true TVT error で row-wise tuning しない。
- 再現性: `docs/06_reproducibility.md` に従い、GPU 学習、HMM 生成、Kaggle bootstrap、SHA 記録を設計と結果に残す。

## 受け入れ基準

- `config.yaml` に route、親、active variant、quantile alpha、HMM grid、Kaggle input sources、booster 数ガードが明記されている。
- `exp229..._train.ipynb` が q16/q50/q84 の OOF band、model manifest、feature importance、summary を保存する。
- `exp229..._train_aggregate.ipynb` が quantile OOF band を読み、row-wise sigma 付き HMM cache と exp221 相当の comparison readout を保存する。
- crossing rate、band coverage、sigma floor/cap rate、overall/bucket/hidden-like/by-well/step-delta を記録できる。
- deterministic anchor として扱わず、採用判断は Kaggle train/audit 実行後の証拠に限定する。
