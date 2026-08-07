# 要件

## 仮説

exp226 K16 geometryの相対rate変化には、exp209 constant rate-prior meanを改善する
fold-safeな未知suffix signalがあるかを二段階で検証する。

## 依頼

閉鎖済み`exp323_time_varying_exp226_dip_rate_prior`をreopenせず、失敗した
exp307--309 chainへの依存を削除した新番号の独立実験として設計を確定する。
初回はbacklog、steering、実験scaffoldだけを作成し、科学実装や実行は行わない。

## 追加依頼（2026-07-23）

ユーザーの「exp355を実装してください」により、Stage 0の科学実装、
compact self-contained train候補、fail-closed inference候補、契約テストの作成を
承認する。正規Notebookへの採用、Kaggle package/push/run、Stage 1の773 HMM runs、
inference、submissionはこの承認に含めない。

## 実行依頼（2026-07-23）

ユーザーの「実行してください」により、compact train候補の正規Notebook採用、
Kaggle CPU package/push/run、Stage 0完了監視を承認する。Stage 1の773 HMM runs、
inference、submissionは引き続き承認対象外とする。

## Stage 1 override依頼（2026-07-23）

ユーザーの「平均で改善しているのなら次に進んでください」により、Stage 0の
worst-well guard FAILを明示的にoverrideし、平均改善を根拠としてStage 1へ進む。
承認範囲はtrain-sideの候補1件、773 wellのexact-HMM、Kaggle CPU package/push/run、
完了監視までとする。parent/control再実行、parameter rescue、raw-test inference、
submissionは含めない。

## 制約

- Route: `pf_beam`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 履歴参照: `exp323_time_varying_exp226_dip_rate_prior`
- exp209のGaussian観測、`sig_r=0.002`、`sig_p=0.02`、grid、momentum、
  prior、posterior meanを固定する。
- 唯一の候補変更は、constant rate-prior meanをfold-safe exp226 K16
  geometry rate-change scheduleへ置換すること。
- exp226からはpre-GR / pre-U geometry-only `tvt_geop`の変化だけを使い、
  `tvt_pred`、GR correction、absolute TVT unary、blendを禁止する。
- Stage 0はdiagnostic 1、5 reporting folds、HMM/model/trained fold/booster各0。
- Stage 1は原則Stage 0全gate PASS時のみとするが、今回はユーザーの明示overrideにより
  1 variant / 773 HMM well-runsを実行する。
- 親/control再実行は0。inference、submissionは別判断。
- 再現性は`docs/06_reproducibility.md`に従い、fold/well/segment順と
  schedule/content SHAを固定する。

## 受け入れ基準

- 初回scaffoldは`config.yaml`、README、SESSION_NOTES、result、metrics、
  placeholder Notebookがdesign-only / 未実装 / 実行無効で整合する。
- 追加実装後はcompact self-contained train/inference候補、config、文書、契約テストが
  Stage 0実装済み / 正規Notebook未採用 / 実行無効で整合する。
- Stage 0の入力、式、fallback、gate、truth late-join、実行量が結果前に固定される。
- Stage 0でsegment rate-change RMSE 5%以上、cumulative path RMSE 0.05 ft以上、
  4/5 folds、1000+・hidden-like 2面非悪化、worst-well +0.25 ft以下を要求する。
- Stage 1でexp209比0.05 ft以上、4/5 folds、stress/tail非悪化を要求する。
  overrideは実行許可だけであり、Stage 1のpromotion gateは緩和しない。
- deterministic anchorとは扱わず、Kaggle package/push/runを無効にする。
- 実行依頼までは`execution.kaggle_push_approved=false`、`run_stage_0=false`を維持し、
  実行依頼後だけ両方をtrueにする。
