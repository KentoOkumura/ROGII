# 要件

## 依頼

- `exp338_exp209_well_adaptive_transition_noise`を新規実験として作成し、バックログ、steering、実験ディレクトリで設計を確定する。
- 科学的親を`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`へ直接固定し、exp209 exact-HMMのwell共通`sig_r=0.002`だけをknown-prefix由来のwell別`sig_r,w`へ置換する。
- 既存`exp309_well_adaptive_transition_noise`は式の参照元に限定し、exp307 finite-only `sigma_GR`、exp308 missing-GR confidence、旧parent prediction/SHAを持ち込まない。
- 今回はdesign-onlyとし、科学実装、Notebook編集、Jupytext変換、Kaggle package/push/run、inference、submissionを行わない。
- exp338が全promotion gateをPASSした場合だけ、旧exp323を再開せず、exp338を親にした「新exp323相当」を新しい実験番号で作成する。
- 新exp323相当が全promotion gateをPASSした場合だけ、旧exp324--327を再開せず、それぞれの仮説を新しい実験番号へ分けて分岐させる。
- 旧exp323--328は旧exp307/308/309 lineageの閉鎖履歴として維持し、reparent、実装、実行しない。

## 2026-07-22 実装承認による更新

- ユーザーの`exp338を実装してください`という別依頼により、design-only停止を解除し、科学実装と静的検証までを承認済みとする。
- compact self-contained train候補、fail-closed inference候補、専用contract testを実装する。
- 既存の正規Notebook placeholderは上書きせず、compact self-contained候補を別名で作成する。
- Kaggle package/push/run、inference、submission、新exp323相当の作成は引き続き未承認とする。

## 仮説

exp209の観測モデルは固定したまま、known prefixの`U=TVT_input+Z`から観測されるrate innovationのrobust scaleをwell別rate diffusionへ使えば、曲率の小さいwellでは不要なrate wanderingを抑え、曲率の大きいwellでは固定`sig_r=0.002`より追従性を上げられる。

```text
q_i = delta(TVT_input + Z)_i / delta(MD)_i
e_i = (q_i - q_(i-1)) / sqrt(0.5 * (delta_MD_i + delta_MD_(i-1)))
s_raw = 1.4826 * MAD(e)
alpha = n / (n + 100)
log(sig_r,w) = alpha * log(max(s_raw, 1e-6)) + (1-alpha) * log(0.002)
sig_r,w = clip(sig_r,w, 0.001, 0.004)
```

有効innovationが20未満またはscaleがnonfiniteの場合は`sig_r,w=0.002`へno-op fallbackする。

## 制約

- Route: `pf_beam`。
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`。
- 変更する変数はwell別rate diffusion `sig_r,w`だけ。
- exp209のknown-prefix zero-fill population-std `sigma_GR`、evaluation GR補間、Type Well GR前処理、Gaussian emission、`step=0.35`、41 rate states、`rate_span=0.10`、`sig_p=0.02`、position floor `0.1225`、momentum `0.998`、start/rate prior、posterior meanを固定する。
- GR affine、finite-only sigma、missing-GR weight、row/segment別`sig_r,t`、time-varying prior mean、momentum、`sig_p`、state grid、blend weightを同時に変更しない。
- `sig_r`式、minimum innovation、pseudocount、floor、clipを結果後に変更しない。
- unknown-suffix TVT/error/oracle、formation、hidden-like label、fold scoreはtransition auditとcandidate predictionのcontent SHAを凍結するまで読まない。
- saved exp209 exact-HMMをcontrolとして使い、control HMM、LikPF、PF、Beam、modelを再実行しない。
- 再現性は`docs/06_reproducibility.md`に従い、raw/input/scientific contract、transition audit、prediction、metricsのdecompressed content SHAを記録する。

## 受け入れ基準

- input well identity、saved exp209 HMM cache、saved LikPF cache、fold、hidden-like assignmentの契約とSHAが一致する。
- transition auditは773 wellsを一意に覆い、finite `sig_r,w`率100%、fallback率50%未満、clip率50%未満を満たす。
- 保存済みexp209 raw HMM RMSE `11.9382872349`から`0.05 ft`以上改善し、4/5 folds以上改善する。
- 1000+、hidden-like spatial、hidden-like typewell-purged、by-well p95を非悪化とし、worst-well regressionを`+0.25 ft`以下にする。
- saved LikPFとの固定50:50 blendをexp209基準`10.2696961466`から悪化させない。
- baseline再評価差はfloat/input精度を考慮して絶対`1e-5 ft`以内とする。
- 1 gateでもFAILした場合は、`sig_r` grid、clip/pseudocount/threshold変更、`sig_p`/momentum/grid変更、blend救済、inference、submissionへ進まず閉じる。
- 本実験はdeterministic submission anchorではない。model/submission SHAは非該当で、Kaggle kernel version、transition audit、prediction/content SHAは実行時に記録する。

## 後続分岐契約

1. exp338 PASS時のみ、新番号で`time_varying_exp226_dip_rate_prior_on_exp338`（新exp323相当）を設計する。exp338の観測モデル、well別`sig_r,w`、grid、prior、posteriorを固定し、exp226由来のtime-varying dip-rate prior meanだけを変更する。
2. 新exp323相当 PASS時のみ、次を互いに独立した新番号へ分岐する。
   - 新exp324相当: exp226 donor covarianceによるsegment別`sig_r,t`。
   - 新exp325相当: exp226 window likelihoodの疎なHMM観測因子。
   - 新exp326相当: residual-rate time-varying momentum。
   - 新exp327相当: quantization-aware time-varying position sigma。
3. 各分岐は別steering、別config、別実装承認、別Kaggle実行承認を必要とし、自動作成・自動実行しない。
4. 旧exp328相当はこのpromotion chainに含めない。再検証入口はexp209を直接親にした独立兄弟`exp345_exp209_time_varying_gr_affine_calibration_hmm`として作成済みであり、旧exp328をreparentしない。exp338とexp345は相互のPASS条件にしない。

## 次のアクション

Kaggle CPU version 3は773/773 HMMを完了したが、direct `+2.124061 ft`、0/5 folds、clip fraction `1.0`で受け入れ基準をFAILした。要件どおり救済変更、inference、submission、後続実験を行わずterminal closeする。

## 2026-07-23 最終判定

- promotion gate: FAIL
- decision: `adaptive_sig_r_failed_close_without_rescue`
- 全773 wellsの最終`sig_r`が上限`0.004`となり、well-adaptiveという中心要件を満たさなかった。
- 新exp323相当以降の後続分岐条件は不成立。
