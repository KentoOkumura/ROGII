# 要件

## 依頼

exp408で第二経路として確定したfuture betaとforward filterのrate disagreementを使い、
baseline passでfreezeしたtrigger scheduleに従って第二passのrate transitionだけを
方向付きde-stickする高リスク第二案を、独立したバックログ、実験ディレクトリ、
steeringとして設計確定する。実装、Notebook編集、Kaggle package / push / run、
inference、submissionはまだ行わない。

## 制約

- Routeは`pf_beam`。
- 科学的親は`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`、
  原因証拠は`exp408_hmm_message_rate_basin_audit`とする。
- first passは変更なしexp209でfiltered / smoothed rate momentを生成する。
- second passで変更するのは、freeze済みbeta-filter trigger rowのrate transitionだけ。
- position、GR emission、beta weight、history mass、state support、readoutは変更しない。
- triggerはbaseline HMMのtarget-free filtered / smoothed rate momentだけで生成し、
  truth、error、episode、fold、hidden-like roleを入力にしない。
- baseline internal messageが全773 wellsで未保存のため、Stage 1は
  `773 baseline + 773 treatment = 1,546 HMM well-runs`を必要とする。
- 親control再実行を含むため、Stage 0 / Stage 1の各Kaggle実行前に明示承認を得る。
- model、LightGBM config、trained fold、booster、PF、Beam、GPUは0。
- exp411より低いP3とし、exp411の結果またはユーザーoverrideなしに実装・実行しない。
- 同一OOFでbeta threshold、window、persistence、gammaを探索しない。
- 再現性は`docs/06_reproducibility.md`に従い、baseline parity、trigger schedule、
  treatment prediction、metricsのdecompressed content SHAを記録する。

## 受け入れ基準

- beta-filter disagreement、active-row判定、方向決定、transition変更を一意に定義する。
- backward / forward原因を含むStage 0 fixed32 sampleとgateを固定する。
- two-passのtruth-late境界、実行量、runtime / RSS上限を固定する。
- Stage 1のtargeted backward-cause gateと全体promotion gateを固定する。
- exp411に対する実装・実行順序とfail-closed条件を明記する。
- config、README、SESSION_NOTES、result、metricsはdesign-only状態を明記する。
- 実装・実行承認は今回の依頼に含めない。

## 2026-07-28 追加依頼

ユーザーの「exp412を実装してください」を実装承認として追加する。

- compact self-contained Jupytext train候補を実装する。
- fail-closed inference候補と専用contract testを実装する。
- fixed32 manifestを生成しSHAを固定する。
- 正規Notebookの既存placeholderは、別の採用判断なしに上書きしない。
- parent controlを含むStage 0 64 HMM well-runsのKaggle package / push / runは
  この追加依頼にも含めず、別の明示承認を必要とする。
