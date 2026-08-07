# 設計

## アプローチ

exp209 exact HMMの遷移率に7つの離散modeを追加する。modeは `base_prefix_rate` と6 formation rate。初期確率はbase 0.5、各formation 1/12。K16境界では同一mode維持0.95、他の各modeへ0.05/6、区間内はmode切替なしとする。emission、state grid、prefix条件、評価処理はexp209から変えない。

## 実験範囲

- 対象実験: `exp379_formation_transition_mixture_exact_hmm`
- Route: `pf_beam`
- 親実験: `exp209_emission_dynamics_direct_hmm`
- 物理候補: exp378
- 変更する変数: transition rate modeとmode transition。
- 固定する変数: exp209 emission/state grid/prefix、fold、評価scope。
- Stage 0: 事前固定16坑井でparity・posterior・runtime・RSS監査。
- Stage 1: 1 variant、773坑井のfull CV。Stage 0後に再承認。

## 停止条件

exp377/378不合格、base parity不一致、posterior退化、runtime/RSS超過のいずれかで停止する。ハイパーパラメータ探索や救済variantは追加しない。

## 再現性設計

- seed policy: exact HMMで乱数なし。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: exact HMMのみ。評価用固定likelihood-PF blendは既存artifactを使う。
- 並列処理と乱数の関係: 乱数なし。well順・mode順を固定する。
- CPU/GPU runtime と deterministic flags: CPUのみ。
- train cache / test feature regeneration の SHA 記録方針: exp378候補、fold manifest、mode-rate tensorのcontent SHAを保存する。
- model manifest / prediction / submission SHA 記録方針: transition matrix、HMM manifest、well prediction SHAを保存する。submissionは対象外。
- Kaggle package bootstrap 確認方針: Stage 0実装時にoffline importを確認する。

## リスク

- リークリスク: exp378のrole不一致artifactを使わない。
- CV/LB不一致リスク: mode数増加でCV固有の面へposteriorが寄る可能性があるためfold・scope・well tail gateを固定する。
- ランタイム/メモリリスク: state×7 modeで計算量が増えるため16坑井からfull projectionを行う。
- 再現性リスク: 浮動小数集約順をmode/well順で固定し、posterior normalizationを監査する。
