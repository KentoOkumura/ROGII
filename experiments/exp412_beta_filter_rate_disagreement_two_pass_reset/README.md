# exp412_beta_filter_rate_disagreement_two_pass_reset

## 状態

- Route: `pf_beam`
- 状態: Stage 0 Version 3完了・`stage0_fail_closed`
- 優先度: P3・高リスク
- CV / Public LB / Private LB: 未実行
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 原因証拠: `exp408_hmm_message_rate_basin_audit`
- 先行実験: `exp411_predictive_filtered_rate_innovation_destick`

## 仮説

future betaとforward filterのrate meanが標準化差2以上で持続的に不一致となる区間では、
betaのabsolute position basinが誤っていても、local rate方向はtrue-rate correctionを
示す場合がある。baseline passでその区間をfreezeし、second passのrate transitionだけを
beta方向へde-stickすれば、rate回復を早めてdatum offset形成を抑えられる。

## 親と単一変更

first passは完全なexp209 control。直近16 rowsのうち8 rows以上で
`|smoothed-filtered| / max(filtered_std, 0.005) >= 2`、かつ75%以上が同符号ならactive。
second passではactive rowへ入るtransitionだけ、stay massの10%をその方向の隣接rate
stateへ移す。beta weight、position、emission、support、readoutは変えない。

## 検証方針

- Stage 0: backward cause 8 + forward cause 8 + matched control 16のfixed32。
- Stage 0実行量: baseline 32 + treatment 32 = 64 HMM well-runs。
- Stage 1: baseline 773 + treatment 773 = 1,546 HMM well-runs。
- parent internal message再生成が不可避なため、各Kaggle実行は明示承認必須。
- Stage 0はbeta方向一致、backward SSE、forward / control安全性、runtimeを全AND評価。
- Stage 1はexp209比`>=0.05 ft`、4/5 folds、exclusive backward / forward、
  1000+、hidden-like、GR missing、well-tail、fixed blendを全AND評価する。

## 先行条件

exp411 Stage 0 Version 5はfuture-rate方向一致`0.225397`、passing folds`0 / 5`、
control active-row fraction`0.136119`、persistent-control active-well差`0.0`で
`stage0_fail_closed`となった。causal trigger / future evidence不足という先行条件は
成立した。2026-07-28のユーザー指示で実装し、別の実行指示でStage 0を完了した。

## 禁止事項

beta weight / history mass、GR / position / support、全体`sig_r`、trigger grid、
treatmentからのtrigger再計算、Viterbi / MAP、blend、inference、submissionは禁止。

## 実装状態

- compact self-contained train / fail-closed inference候補をJupytext起点で実装。
- exp209 baseline parity、beta-filter schedule、frozen-schedule second pass、
  truth/cause late join、Stage 0 gateをNotebook上に展開。
- fixed32 manifest:
  backward 8 + forward 8 + control 16、SHA256
  `1edb1e1481af84af4e8178fb6e0743fa40315eab0b7441eeff9232b571f93c30`
- 専用test 14件、dedicated + notebook contract 18件、py_compile、Ruff F821、
  Jupytext round-trip、
  strict experiment validationはPASS。
- 2026-07-28の実行指示によりcompact train / fail-closed inferenceを正規Notebookへ
  採用し、Kaggle private CPU Version 3で64 / 64 HMM well-runsを完了。
- technical 12 / 13、mechanism 5 / 6 PASS。方向一致`0.776347`、4 / 5 folds、
  forward / control安全性はPASSしたが、backward cause SSE reduction
  `-0.069575`とfull runtime projection`51,753.199秒`がFAIL。

## 所見

beta-filter disagreementはexp411より方向性と選択性が高かったが、固定10% transition
de-stickはbackward causeを安定して修復しなかった。改善wellと悪化wellが混在し、
主目的のpooled SSEが`6.96%`悪化したため、same-OOFでtriggerやtransfer量を救済しない。

## 次

exp412はnegative resultとして閉じる。Stage 1、inference、submissionは実行しない。
