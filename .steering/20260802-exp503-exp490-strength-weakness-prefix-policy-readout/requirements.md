# 要件

## 依頼

正解TVTをreadoutに使用し、exp490が強いwell・弱いwellの条件を可能な限り多面的に
調査する。特に、公開notebookで使われている「既知prefixからsuffixへ進むにつれて
補正をfade-inする」「既知区間で挙動を判定して処理を変える」という発想が、exp490の
保存済みOOFでも有効かを検証する。

## 制約

- Route: `ensemble`。exp490 PF/HMM候補とexp357親予測の後処理比較であり、両方が
  予測生成に本質的に寄与する。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- exp490/exp357の候補生成、HMM/PF/Beam、control再学習、GPU実行は0とする。
- 保存済みexp490 full OOFとexp499で凍結済みtarget-free well特徴だけを使う。
- truthを使う説明的監査と、outer-fold-safeな方策評価を明確に分離する。
- early-suffix truthを使う方策は実運用可能とは扱わず、prefix backtestの楽観的な
  transfer上限としてだけ評価する。
- 初回の全件実行はKaggle CPU notebookで行い、ローカルでは構文・契約・小規模
  synthetic testだけを行う。

## 受け入れ基準

- 773 wells / 3,783,989 suffix rowsをSHA-pinned入力から再現できる。
- fold、well、suffix深度、親の難度、補正量、posterior不確実性、prefix情報量、
  誤差bias/drift、連続悪化区間の各観点で強弱を表と図にする。
- well別best/worst、特徴量とのSpearman/AUC、depth profile、失敗archetypeを保存する。
- 公開notebook由来のfade-in族を固定gridで作り、outer 4 foldsだけでprofileを選び、
  held foldを一度だけ評価する。
- prefix-only特徴による解釈可能なalpha treeをouter-fold-safeに評価し、exp499同様に
  routing可能性とtail riskを分けて判断する。
- early 128/256/512 suffix rowsのtruthで選んだprofileが後半へtransferするかを、
  非deployable optimistic auditとして記録する。
- `metrics.json`、well/depth/feature/policy各CSV、plot、summary JSON、
  `SESSION_NOTES.md`、`result.md`、`experiment_summary.md`、`KAGGLE_DIRECTION.md`を更新する。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
