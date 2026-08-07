# 要件

## 依頼

exp440の「曖昧ならpredictive priorを維持する」を、そのまま救済せず、
「前後の文脈は同じTVTを支持しているのに、現在のraw GR 1点だけが
局所的な外れ値として別方向へ更新させる場合」に限定して再設計する。

この条件を満たす行だけcurrent GR emissionを出力readoutから除外し、
current emission適用前のpredictive messageを維持した
leave-one-current-observation-out posterior meanへ置換する。
親HMMのforward/backward状態と後続行の予測は変更しない。

今回は`KAGGLE_DIRECTION.md`のbacklog、steering 3文書、
`experiments/exp482_isolated_gr_shock_prior_hold/`のdesign-only scaffoldを
作成して科学設計を確定する。candidateコード、専用test、Jupytext source、
正規Notebook実装、Kaggle package、push、run、inference、submissionは行わない。

## 仮説

過去側predictive messageと現在観測を除いたfuture messageが同じTVT近傍で一致し、
raw GRの現在点だけが前後の局所水準から孤立している場合、current emissionを
使った親smoothed predictionより、current emissionを除いたrow-local posteriorを
使う方が正しい確率が高い。

## 制約

- Route: `pf_beam`
- 親は`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`とする。
- `exp440_ambiguity_gated_predictive_prior_hmm`のambiguity schedule、threshold、
  active row、candidate prediction、truth-late結果をtrigger設計へ使わない。
- raw GR shock、past/future message agreement、current-emission conflictの
  3条件をANDで満たす行だけを対象にする。
- raw GR shockはcurrent rowを除く`±5`行のraw finite GRから計算する。
  imputed GR、true TVT、error、fold、hidden-like role、well ID ruleを使わない。
- trigger rowだけをrow-local leave-one-out meanへ置換し、親HMMのfiltered state、
  backward message、後続予測を変更しない。
- 連続・近接shockは孤立観測とみなさず、`±2`行内に別shockがあるclusterは
  全行非activeとする。
- Stage 0はraw-only census後にtarget-freeに作るfixed64
  （shock-support 32 wells + zero-shock matched control 32 wells）だけを使う。
- Stage 0/1ともscientific candidateは1本。threshold、window、scale floor、
  output weight、soft blend、emission family、well/row selectorを探索しない。
- 保存済みexp209 predictionをcontrolとし、control predictionを再生成しない。
  内部message生成に必要なunchanged exp209 HMM replay数は別に記録する。
- Stage 1 full OOFはStage 0の全gate PASSと別のユーザー承認を必須とする。
- inferenceとsubmissionはStage 1 promotion PASS後も自動で行わず、別承認を必須とする。
- 再現性は`docs/06_reproducibility.md`に従い、truth-late freeze、入力、
  raw-shock census、manifest、message、trigger、prediction、metricsのSHAを記録する。

## 受け入れ基準

- `.steering/20260730-exp482-isolated-gr-shock-prior-hold/`の
  requirements/design/tasklistが単一のtrigger、単一candidate、段階gate、
  no-rescue、承認境界を一意に記述している。
- `experiments/exp482_isolated_gr_shock_prior_hold/`に`config.yaml`、README、
  SESSION_NOTES、result、metricsとテンプレートNotebook scaffoldが存在する。
- 実験状態が`implemented_not_run`、Routeが`pf_beam`、
  implementationのみ承認済み、canonical Notebook採用/run/inference/submissionが
  未承認で一致している。
- raw-shock、past/future agreement、current-emission conflict、
  row-local leave-one-out置換の数式と固定値がconfig/designで一致する。
- Stage 0のraw census、fixed64 manifest生成、64 message replays、
  Stage 1の773 message replays、model/booster/PF/Beam/GPU各0が記録されている。
- trigger、prediction、manifestをtrue TVT、fold、role、error結合前にfreezeする。
- Stage 0/1のtechnical/scientific gateとFAIL時のterminal actionが固定されている。
- `KAGGLE_DIRECTION.md`へ既存候補と比較した低・P3の実装済み・未実行項目として
  記録されている。
- `experiment_summary.md`へ`implemented_not_run`として記録されている。
- deterministic anchorとは扱わず、gzip生成物はdecompressed content SHAを
  主証拠にする。

## 2026-07-30 実装承認追記

ユーザーの「exp482を実装してください」により、上記design-only境界のうち
compact self-contained train候補、fail-closed inference候補、専用test、
静的検証までを承認済みに変更する。正規Notebook上書き、Kaggle package、
push/run、Stage 1、inference、submissionは引き続き未承認とする。
