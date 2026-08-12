# 要件

## 依頼

exp408で主因と確定したforward rate-prior hysteresisに対し、predictive→filtered
rate innovationを使ってrate transitionだけを一時的にde-stickする第一案を、
独立したバックログ、実験ディレクトリ、steeringとして設計確定する。実装、
Notebook編集、Kaggle package / push / run、inference、submissionはまだ行わない。

## 制約

- Routeは`pf_beam`。
- 科学的親は`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`、
  原因証拠は`exp408_hmm_message_rate_basin_audit`とする。
- 変更はrate transitionのstay massをinnovation方向へ一時移動する処理だけ。
- position grid / kernel、41 rate states / span、GR preprocessing / emission、
  `sig_r=0.002`、`sig_p=0.02`、`mom=0.998`、prior、backward smoothing、
  posterior-mean readoutは固定する。
- mode ID固定、全well一律`sig_r`拡大、GR weight、position exact-mean、
  Viterbi / MAP置換、blendを併用しない。
- triggerはHMM内部のtarget-free predictive / filtered rate momentだけで計算し、
  truth、error、episode、fold、hidden-like roleを入力にしない。
- Stage 0とStage 1の閾値、持続長、stay mass移動率を同一OOFで探索しない。
- 親controlのfull HMMは再実行せず、保存済みexp209 prediction / metricsを使う。
- 新規model、LightGBM config、trained fold、booster、PF、Beam、GPUは0。
- 再現性は`docs/06_reproducibility.md`に従い、入力、trigger schedule、
  prediction、metricsのdecompressed content SHAを記録する。

## 受け入れ基準

- CUSUM式、固定定数、transition変更式、edge処理、refractoryを一意に定義する。
- Stage 0の32-well選択規則、実行量、technical / mechanism gateを固定する。
- Stage 1の773-well実行量、promotion gate、fail-closed条件を固定する。
- truth-late境界とtrain / hidden-testで同一のtrigger生成契約を明記する。
- config、README、SESSION_NOTES、result、metricsはdesign-only状態を明記する。
- backlogではP2の手堅い第一案として、P3のexp412より先行させる。
- 実装・実行承認は今回の依頼に含めない。
