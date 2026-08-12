# 要件

## 依頼

exp293のscientific support PASSを受け、固定deployable12 candidate bankを変更せず、known-prefixで校正した
Type Well/horizontal GR evidenceがH256/H512 blockでtruth-good candidateへ確率質量を置けるか監査する。

## 制約

- Route: `pf_beam`。ML predictor/selector、学習済みTVT model、ML blendを使わない。
- 親は`exp293_physics_only_candidate_bank_headroom_contract` version 2 support PASS。
- candidate順・値はexp293 deployable12に固定し、追加、削除、再生成、weight変更を行わない。
- known-prefix calibrationは最大512 rows、finite pair最小64、Huber IRLS 2回、slope clip `[0.25,4.0]`。
- residual scaleは`1.4826*MAD`を`[10,60]`へclipする。
- registration gridは`[-20,20] ft`を2 ft刻みの21 states。Type Wellは`candidate_tvt + delta`で参照し、
  deltaをTVT predictionへ加えない。
- scoreはStudent-t raw residual (`df=4`)、NCC、chain-rule derivative residualをstate集合内median/MADで
  標準化し、各`1/3`で固定する。
- reliable priorは0.9、registration priorは`exp(-abs(delta)/10)`、candidate priorは12本一様。
- candidate-independent outlier likelihoodは正規化log-likelihood 0に固定し、unreliable質量は
  `exp226_w500_50_50`だけへ置く。
- real/shuffleとも同じwell、block、candidate、registration、GR finite maskを使う。shuffleはstable SHA seed。
- target-free joint/candidate/registration posteriorをSHA freeze後にだけsuffix truthを読む。
- hard top1、candidate TVT平均、posterior mean TVT、selected/corrected row prediction、inference、submissionを作らない。
- Stage 2 PASS/FAIL条件はexp293 `downstream_branch_contract.md`から変更しない。
- 再現性は`docs/06_reproducibility.md`に従う。

## 受け入れ基準

- steering/config/docsにcandidate、registration、calibration、reliability、shuffle、truth freeze、PASS条件が固定されている。
- Jupytext compact self-contained trainとfail-closed inference、contract testsが実装されている。
- H128/H256/H512の全blockについてjoint reliable posterior、candidate posterior、registration posterior、entropy、
  mode gap、reliabilityを保存する設計である。
- H256 expected candidate SSE recovery `>=0.35`、5/5 folds正、realがshuffleをpooledかつ5/5 foldsで上回る、
  H512 recovery低下`<=0.05`、1000+/hidden-like 2面anchor非悪化をPASS条件とする。
- feature/score/posterior freeze前のtruth accessが0である。
- model/config/trained fold/booster/HMM-PF再生成はすべて0。
- `make validate-exp EXP=exp297_prefix_calibrated_latent_registration_gr_evidence`が通る。
- canonical notebook採用、Kaggle package/push/run、inference、submissionは未実施のままにする。

## 次

compact実装と静的検証後、canonical採用と1回のKaggle private CPU auditは別承認を待つ。
