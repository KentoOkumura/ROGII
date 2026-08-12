# 設計

## 目的と判断対象

exp293ではdeployable12 bankのH512 oracle RMSEが`3.683763`、6.5に必要なSSE headroom回収率が
`0.471825`でsupport PASSした。exp297はcandidateを増やさず、Type Well/horizontal GR観測からblockごとの
candidate確率`p_b(c)`をtruthなしで構築し、その確率質量がoracle headroomを回収できるかだけを判定する。

## 入力契約

- exp293 version 2のcandidate bank float32 matrix、bank manifest、block assignmentをfile/logical SHA固定で読む。
- rows/wells/candidatesは`3,783,989 / 773 / 12`、outer foldはexp293の0..4をそのまま使う。
- raw horizontalからtarget-freeに読む列は`MD/GR/TVT_input`だけ。Type Wellは`TVT/GR`だけ。
- hidden-like assignmentはexp115の固定SHAを使う。
- true `TVT`はtarget-free生成物を閉じて再hashした後のreadout loaderだけが読む。

## Calibration

- known prefixのfinite `TVT_input`をType Wellへno-extrapolationで写像し、最後の最大512 pairを使う。
- minimum pair 64、Type Well GR std minimum 5、Huber delta 1.345、IRLS 2回。
- slopeは`[0.25,4.0]`へclip、interceptはfinal Huber weightで再計算する。
- prefix RMSEが60超、非finite、分散不足はwell全体をunreliableとする。
- raw residual sigmaは`1.4826*MAD`を`[10,60]`、derivative sigmaはprefix residual差分MADを`[1,30]`へclip。

## Block evidence

- exp293のnon-overlap H128/H256/H512 groupをそのまま使い、末尾short blockも保持する。
- registration stateは`delta=-20,-18,...,20 ft`。参照GRは`g_typewell(candidate_tvt + delta)`。
- block/stateのfinite pairは`max(64, ceil(0.5*block_rows))`、derivative pairはその1少ない値を必要とする。
- Type Well local forward-GR std minimum 5、median absolute forward derivative minimum 0.25をexp292から継承する。
- raw componentはStudent-t df4平均log-likelihood、NCCはfinite pair Pearson、derivativeは
  `-mean(abs(observed_dGR-chain_rule_dGR))/derivative_sigma`。
- 各block/control/componentを12×21 states内median/MAD z-scoreし`[-5,5]`へclip、等重みで加算する。
- invalid stateはreliable posterior 0。valid stateはcandidate一様、registration Laplace prior scale 10、
  reliable prior 0.9とする。
- outlier/unreliableはcandidate/stateに依存しないnormalized log-likelihood 0、prior 0.1とし、
  final candidate posteriorではsafe candidateだけへ加算する。

## Negative control

- wellごとに`SHA256(experiment,seed,well)`からstable rotationを作る。
- suffix GRのfinite値列だけをcircular rotateして元のfinite位置へ戻し、NaN mask、block、candidate、horizonを保つ。
- すべてのhorizonで同じrotationを使う。

## 保存するtarget-free evidence

- block indexとcalibration eligibility。
- reliable joint posterior dense float32 NPY `[block_control,12,21]`。
- candidate posterior Parquet、registration posterior Parquet。
- block summary: reliable/unreliable posterior、candidate/registration entropy、candidate mode gap、eligible states。
- selected TVT prediction、candidate加重TVT、registration補正TVTは保存しない。

## Truth readout

- block candidate SSE `SSE_b(c)`から`E[SSE_b]=sum_c p_b(c)SSE_b(c)`を計算する。
- recoveryは`(SSE_anchor-SSE_expected)/(SSE_anchor-SSE_oracle)`。
- pooled/foldでreal/shuffle、H128/H256/H512を出す。primaryはH256。
- 1000+、hidden-like spatial/typewell-purgedはH256 real expected SSEでanchor非悪化を判定する。

## PASS/FAIL

- H256 pooled recovery `>=0.35`。
- H256 recovery 5/5 foldsで正。
- H256 real expected SSEがshuffleよりpooledかつ5/5 foldsで小さい。
- H512 recoveryがH256から0.05を超えて低下しない。
- 1000+、hidden-like spatial/typewell-purgedでanchor非悪化。
- freeze前truth access 0。

PASS時だけStage 3へ進む。FAIL時は停止し、Stage 4へ自動分岐しない。

## 実験範囲

- 対象: `exp297_prefix_calibrated_latent_registration_gr_evidence`
- Route: `pf_beam`
- 親: `exp293_physics_only_candidate_bank_headroom_contract`
- 変更: GR観測evidenceとposterior readoutだけ。
- 固定: candidate bank、fold、block、registration grid、weights、priors、thresholds、PASS条件。
- model/config/trained fold/booster/HMM-PF run: `0/0/0/0/0`。

## 再現性設計

- seed policy: realはRNGなし、shuffleだけSHA256 per-well local RNG。
- stochastic処理: stable circular shuffle negative controlのみ。
- PF/Beam再生成: なし。exp293保存candidateを読むだけ。
- parallelism: single process、global RNGなし。
- runtime: Kaggle private CPU、GPU/AMP/internet off予定。
- input/evidence: raw/file/decompressed/logical/schema SHAを記録する。
- model/prediction/submission SHA: 対象なし。posterior/evidence SHAを記録する。
- bootstrap: loose/package/bootstrap config/source bytesをpush前に照合する。
- deterministic anchor: submission anchorではなくfixed-input diagnostic。

## リスク

- leakage: posterior freeze前にtrue TVTを読むとreadout全体が無効。loaderとcounterを分離する。
- compute: 3.78M×12×21 stateをwell単位vectorizeし、block sufficient statisticsで集約する。
- posterior過信: unreliable safe massを常時保持し、GR不一致時にcandidate移動を強制しない。
- oracleとの差: exp293 oracleは上限であり、0.35 recoveryは厳しい。FAIL時のweight/grid救済は禁止。
