# 要件

## 依頼

閉鎖済み`exp324_exp226_donor_covariance_segment_sig_r`をreopenせず、exp323や
exp307--309 chainに依存しない新番号の独立実験として設計を確定する。
今回は設計とscaffoldまでとし、実装・実行は行わない。

## 制約

- Route: `pf_beam`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 履歴参照: `exp324_exp226_donor_covariance_segment_sig_r`
- exp209のconstant rate mean、Gaussian観測、`sig_p`、grid、momentum、prior、
  posterior meanを固定する。
- 唯一の変更は`sig_r=0.002`をfold-safe exp226 donor covariance由来の
  K16 segment scheduleへ置換すること。
- exp355/exp323のtime-varying rate meanを使わず、独立に実行可能とする。
- Stage 0はdiagnostic 1、5 reporting folds、HMM/model/trained fold/booster各0。
- Stage 1はStage 0全gate PASSと別承認時だけ1 variant / 773 HMM well-runs。
- 親/control再実行0、GPU/internet off、inference/submission別判断。

## 受け入れ基準

- donor identity、weight、projected rate、effective support、scale、clip、
  fallback、segment scheduleをsuffix truth結合前にSHA固定する。
- minimum effective donors 10、support 50 log-shrink、clip`[0.001,0.004]`、
  fallback`0.002`を固定し、gridを行わない。
- Stage 0でtransition NLL 1%以上、4/5 folds、68/95% coverage非悪化、
  stress非悪化、fallback/clip各50%以下を要求する。
- Stage 1でexp209比0.05 ft以上、4/5 folds、tail/hidden-like非悪化を要求する。
- design-only、未実装、実行flag全offで全記録を整合させる。
