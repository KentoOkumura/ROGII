# 要件

## 依頼

バックログ `crossfitted_residual_scale_emission_hmm_on_exp218` を、`exp234_crossfitted_residual_scale_emission_hmm_on_exp218` として実装する。

保存済みの exp218 `lgb_mean` OOF を HMM の point center として固定し、同一 row の真値残差を使わない GroupKFold（well 単位）の cross-fitted residual scale だけを Gaussian emission の row-wise sigma に使う。HMM は事前の scale readout guard を通過した場合に限り、固定設定の single variant だけを train-side で比較する。

## 制約

- Route: `ensemble`。exp218 ML point prediction と exact HMM posterior の双方が予測生成に本質的に寄与する。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- exp218 の control / baseline LightGBM は再学習しない。残差 scale の cross-fit と HMM audit は CPU のみで実行する。
- q50 の再学習、lambda / floor / cap grid の拡張、raw-test inference、submission は train-side guard が通るまで禁止する。
- sigma は各 held-out well の residual を学習に含めない。HMM の target は評価・readout 以外に使わない。

## 受け入れ基準

- 保存済み exp218 `lgb_mean` OOF が ID 一意・全 train unknown-suffix row を網羅し、HMM center として変更されずに使われる。
- GroupKFold by well の各 held-out row の sigma は、その row / well の真値 residual を学習に含めない。
- HMM 前に residual-scale error lift、decile calibration、sigma floor / cap rate、入力 SHA を生成・記録する。
- guard を満たす場合だけ、固定 lambda / floor / cap の HMM single variant を train-side 比較する。guard 不通過時は HMM / inference / submit を実行しない。
- HMM 実行後は overall、distance bucket、hidden-like、worst-well、step delta と exp218 / exp221 / exp229 を比較して記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録する。
