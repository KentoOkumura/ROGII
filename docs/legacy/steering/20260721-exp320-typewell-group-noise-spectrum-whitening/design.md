# 設計

## アプローチ

exp311のidentity-shrunk calibration residualを欠損をまたがないfinite runへ分割し、outer-trainの同一群からYule-Walker AR(1) rhoを推定する。support `k=200`でglobal rhoへ縮約し、held-out候補のresidualを`e_t - rho e_{t-1}`へ変換したinnovation likelihoodで順位付けする。GR signal自体は変えない。

## 実験範囲

- 対象: `exp320_typewell_group_noise_spectrum_whitening`
- Route: `pf_beam`
- 親: `exp311_typewell_group_prefix_suffix_gr_calibration_readout`
- 変更: independent residual likelihoodからfixed AR(1) innovation likelihoodへ。
- 固定: lag=1、support、shrinkage、clip、fallback、candidate bank、gate。
- 計算量: primary 1 + diagnostics 2 + control 1、5 folds、model/booster/decoder 0。

## 再現性設計

- finite run segmentation、pair count、rho table、fallback、candidate scoreのschema/content SHAを保存する。
- stable well/fold/group順とdeterministic reductionsを使い、global RNGは使わない。
- Kaggle CPU/internet disabled、kernel version、bootstrap config一致を記録する。

## リスクと停止条件

- residual自己相関がwell固有で群共通でない場合、group shuffleとの差が消える。
- AR whiteningが局所geology信号も除く可能性があるためrankとhidden-like/worst guardを要求する。
- FAIL時はAR orderやrho clipを同一readoutで救済しない。
