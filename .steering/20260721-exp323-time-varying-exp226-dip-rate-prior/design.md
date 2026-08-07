# 設計

> **閉鎖済み（2026-07-22）**: 本文は旧lineageの設計履歴であり、実装入口ではない。後継資格はexp338の`successor_policy`を正とする。

## アプローチ

outer-fold train wellsだけでexp226 K16 geometry-only fieldを再構築し、`r_geo,t=Δ(TVT_geop+Z)/ΔMD`を得る。親HMMの初期rateを`r_parent`として、`μ_r,t=r_parent+(r_geo,t-r_geo,first)`を凍結する。Stage 0でrate-changeと累積経路を評価し、全gate PASS時だけ`r_t=μ_r,t+δr_t`の1 HMM variantを許可する。

## 実験範囲

- 対象: `exp323_time_varying_exp226_dip_rate_prior`
- Route: `pf_beam`
- 親: `exp309_well_adaptive_transition_noise`
- 変更: rate transitionの平均scheduleだけ。
- 固定: 観測モデル、`sig_r`、`sig_p`、41 rate states、span、momentum、position kernel、prior、posterior mean。

## 再現性設計

- RNGなし。outer fold、well、K16 segmentの順序を固定する。
- CPU / internet off。Stage 0は0 HMM、Stage 1最大773 HMM runs、0 booster。
- raw identity、outer-fold donor field、rate schedule、predictionのdecompressed content SHAを保存する。
- Kaggle package時はmetadataとbootstrap configを照合する。submissionは作らない。

## リスク

- donor TVT leakage: outer-fold source exclusionとtruth late joinで防ぐ。
- exp226 bias: 絶対levelを捨て、target-well初期rateへanchorする。
- rare tail: overallだけでなく1000+、hidden-like、p95、worst gateを必須にする。
- runtime: Stage 0で科学的に反証してから5時間級HMMを許可する。
