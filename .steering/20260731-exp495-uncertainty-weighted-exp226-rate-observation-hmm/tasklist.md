# タスクリスト

## 現在の状態

`stage_0b_completed_fail_closed`

## 完了

- [x] 最新番号exp494の次としてexp495を採番した。
- [x] steeringを実験scaffoldより先に作成した。
- [x] `pf_beam` route、親exp209、rate evidence exp355、mechanism evidence
  exp408、negative evidence exp411 / exp491を固定した。
- [x] exp226 final差分のhard置換ではなく、geometry U-rate Gaussian観測を
  exp209 rate transitionへ掛ける一要因を固定した。
- [x] known-prefix tail128 MADによるwell-level uncertainty式とparent fallbackを固定した。
- [x] Stage 0A 0-HMM reliability、Stage 0B fixed32、Stage 1 full OOFの
  三段階gateと別承認lockを固定した。
- [x] leakage、coordinate、tail、runtime、再現性、CV/LBリスクを記録した。
- [x] 実験scaffoldとバックログ項目を作成した。
- [x] Stage 0A compact self-contained Jupytext train候補を実装した。
- [x] exp226 geometry / exp209 input SHA、strict allowlist、prefix/suffix freeze guardを実装した。
- [x] `mu_226` / `sigma_226` / fallback / reliability gateの18契約テストを実装した。
- [x] compact候補を別名`.ipynb`へ変換し、py_compile / ruff / pytest / Jupytext testを通した。
- [x] ユーザーの実行指示を受け、compact候補を正規train Notebookへ採用した。
- [x] canonical id/title、private CPU、internet offでKaggle train packageを作成した。
- [x] Kaggle private CPU version 1でStage 0Aを完走し、technical gate全件PASSを確認した。
- [x] mechanism gate 2件FAILを固定閾値で判定し、HMM実装前にbranchを閉じた。
- [x] kernel version / id_no / runtime / input・uncertainty・schedule SHAを記録した。
- [x] ユーザーがStage 0A FAIL停止点を明示overrideし、固定済みStage 0Bの実装・実行を承認した。
- [x] Stage 0B fixed32のGaussian rate observation HMMを実装した。
- [x] 1 variant × 32 candidate HMM well-runsをKaggle private CPU version 4で完走した。
- [x] technical 1件 / mechanism 7件FAIL、artifact SHA、runtimeを記録した。

## ブロック中

- Stage 1: Stage 0B gate FAILかつ別承認なしにより実行禁止。
- inference / submission: 本設計の対象外。

## 停止条件

- Stage 0A fail-closedはユーザーの明示overrideによりStage 0Bだけ解除済み。FAIL判定自体は変更しない。
- Stage 0Bでmechanism gateが1件でもFAILならStage 1へ進まない。
- same-OOFでprefix window、sigma floor、scale、temperature、threshold、gateを調整しない。
- exp491型hard replacement、exp226 final差分、GR二重利用、blend、selector、PF救済を行わない。
