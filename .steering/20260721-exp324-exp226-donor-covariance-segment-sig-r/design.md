# 設計

> **閉鎖済み（2026-07-22）**: 本文は旧lineageの設計履歴であり、実装入口ではない。後継資格はexp338の`successor_policy`を正とする。

## アプローチ

exp226 local-linear k50 donorを同じouter-fold契約で取得し、対象方位へ射影した`Δ(TVT+Z)/ΔMD`のweighted MADとeffective sample sizeを計算する。親well別`sig_r`とのlog shrink後、K16 piecewise scheduleとして凍結する。

## 実験範囲

- 対象: `exp324_exp226_donor_covariance_segment_sig_r`
- Route: `pf_beam`
- 親: `exp323_time_varying_exp226_dip_rate_prior`
- 変更: `sig_r`をwell固定からK16 segment別へ変更。
- 固定: exp323 `mu_r,t`、観測、momentum、position、state grid、decoder。

## 再現性設計

RNGなし。outer fold/well/segment/donor順を固定し、donor source、weight、n_eff、MAD、shrink、clip、fallback、schedule、prediction SHAを保存する。CPU/internet off、0 booster。

## リスク

- epistemic donor分散と物理process noiseの混同: Stage 0 NLL/calibrationで先に反証する。
- 遠方donor: n_eff不足時は親sigmaへfallbackする。
- tail悪化: Stage 1にp95/worst/hidden-like hard guardを置く。
