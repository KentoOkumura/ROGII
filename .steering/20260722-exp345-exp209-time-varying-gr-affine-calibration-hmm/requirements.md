# 要件

## 依頼

旧exp328の時間変化GR affine `a_t,b_t`仮説を、閉鎖済みlineageの再開やreparentではなく、exp209を直接親にした独立の新規実験として記録する。2026-07-22の初回依頼ではdesign-onlyとし、同日の別依頼「exp345を実装してください」を科学実装の承認として、compact self-contained train候補とcontract testsまで作成する。Kaggle実行は承認範囲に含めない。

## 位置づけ

- 対象: `exp345_exp209_time_varying_gr_affine_calibration_hmm`
- Route: `pf_beam`
- 科学的親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 設計元: 閉鎖済み`exp328_time_varying_gr_affine_calibration_hmm`
- sibling: `exp338_exp209_well_adaptive_transition_noise`
- exp338のPASS/FAILと相互依存させない。
- exp338から新exp323相当、新exp324--327相当へ進むchainには含めない。
- 旧exp328はterminal closedのまま再開・reparentしない。

## 科学的制約

- 変更はexp209のGR observation centerに適用するcurrent-well causal `a_t,b_t` scheduleだけとする。
- exp209 posterior mean/stdをbase pathとして凍結し、affine scheduleを1回だけfilterして凍結し、variant exact HMMを1回だけ再実行する。
- visible prefixだけで初期affine stateをfitする。
- suffix affine updateは有限raw GR、frozen base mean上のType Well GR、base TVT stdだけを使う。raw GR欠損rowはupdateをskipする。
- exp209 zero-fill std `sigma_GR`、missing weight 1、GR補間、Gaussian emission、41 rate states、`sig_r=0.002`、`sig_p=0.02`、position floor、momentum、prior、posterior meanを固定する。
- exp307 finite-only/MAD scale、exp308 missing-GR confidence、exp338 well別`sig_r`、same-Type-Well group priorを使わない。
- joint TVT/rate/affine state、bidirectional affine smoother、複数回filter、iterative joint fit、parameter grid、transition変更、prediction blendを禁止する。
- unknown-suffix TVT、error、formation、oracle branch、hidden-like roleはscheduleとprediction freeze後の採点にだけ使う。

## 段階実行制約

- 実装前に別承認を必要とする。
- runtime microbenchmarkはstable SHA順32 wells、親/variant合計64 HMM runsとする。
- Stage 0はruntime外挿`<=8.5 h`をPASSし、別の実行承認を得た場合だけ、last-640 prefix maskで親/variant合計1,546 HMM runsを行う。
- Stage 1はStage 0全gate PASSと別承認後だけ、保存済みexp209 base pathを使う新variant 773 HMM runsを行う。full suffix parent controlは再実行しない。
- LightGBM config、学習fold、booster、PF、Beamはすべて0。CPU、internet offとする。
- 科学実装は有効。既存正規Notebookは上書きせず、compact self-contained候補を別名で作る。
- Kaggle package/push/run、正規Notebook採用、inference、submissionは現時点で無効とする。

## 受け入れ基準

- backlog、実験ディレクトリ、steeringに同じ親、単一変更、固定項目、禁止項目、段階実行量、gate、独立関係が記録されている。
- Stage 0は親比`>=0.05 ft`、4/5 folds、GR NLL改善、boundary jump p95 `<=3 sigma`、hidden-like非悪化、worst `<=+0.25 ft`、fallback `<=50%`、runtime `<=8.5 h`を全て要求する。
- Stage 1はexp209 raw HMM比`>=0.05 ft`、4/5 folds、1000+、hidden-like spatial/typewell-purged、by-well p95非悪化、worst `<=+0.25 ft`を全て要求する。
- exp209 HMM decompressed SHA `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`を依存契約にする。
- RNGなしの順序固定、kernel/input/schema/content/prediction SHA方針を記録する。gzipはdecompressed content SHAを主証拠にする。
- compact self-contained Jupytext train候補、段階実行flag、truth late-join、SHA出力、contract testsが実装される。

## 次

実装は完了した。別途Kaggle実行承認が得られるまで、package生成、push、runへ進まない。
