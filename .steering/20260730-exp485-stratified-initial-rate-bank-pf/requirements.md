# 要件

## 目的・仮説

5つのprefix rate centerを等数粒子で同時に保持する単一PF priorが、
target-aware選択なしに初期mode lossを減らせるか検証可能な実装にする。

## 依頼

exp268でexact HMMに試した複数initial-rate window bankを、現行
temperature-5 likelihood-PFへ適用可能な単一PF初期化実験として設計する。
当初はdesign-onlyとし、2026-07-30の追加依頼`exp485を実装してください`に
よりStage 0を実装した。同日の`実行してください`でStage 0を実行し、
runtime以外の13 gateをPASSした。その後ユーザーが`そのぐらいの実行時間で
あれば許容範囲内`と明示し、`Stage 1を実行してください`と承認したため、
元のruntime gate FAILを履歴として保持したままStage 1を例外実行する。
inferenceとsubmissionは承認対象外とする。

## 根拠

- exp268のHMM best `w128`はtail30 controlを`0.042706 ft`改善した。
- tail30/w32/w64/w128/w256のwhole-well oracle headroomは`0.097314 ft`だった。
- 423/773 wellsでrate spreadが0となり、候補重複も大きかった。
- 現行PFはtail30の単一rate centerとspread `0.01`だけで初期化しており、
  複数center bankは未検証である。

## 制約

- Routeは`pf_beam`。親exp417、実装親・保存control exp404。
- windowは`[30,32,64,128,256]`、rate式とfallbackはexp268に固定する。
- 500 particlesをparticle index modulo 5で100粒子ずつ5中心へ層化する。
- seed数、PF dynamics、GR likelihood、resampling、roughening、T=5は固定する。
- window別PFを5本実行してtarget-awareに選ばない。単一mixture初期化だけを評価する。
- Stage 0 fixed32、Stage 1はruntime以外の全PASS、runtime例外、別承認時だけ773 wells。
- Stage 1 Kaggle CPU実行まで承認済み。inferenceとsubmissionは未承認。

## 受け入れ基準

- 5 center、100粒子ずつ、interleave規則、rate式、valid-step fallbackが一意である。
- component ancestry/collapseとduplicate centerのtarget-free診断を固定する。
- Stage 0/1実行量、保存control、truth-late、seed/SHA、全AND gateを記載する。
- window、allocation、spread、particle/seed数、selector/oracle救済を禁止する。
- compact self-contained train Notebook、5×100 allocation、fallback、
  duplicate-center、stable seed、exp404 duplicate-center parityのtestが通る。

## 次のアクション

Stage 1 version 3はCOMPLETE。technical gateはPASSしたが、candidate
`11.092618091`は保存control `10.914522073`より悪く、scientific gateを
FAILした。Stage 0の元のruntime gate FAILは再分類せず、candidate PF再実行、
inference、submission、同一OOF救済なしでbranchを閉じる。
