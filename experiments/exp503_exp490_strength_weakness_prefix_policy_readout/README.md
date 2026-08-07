# exp503_exp490_strength_weakness_prefix_policy_readout

保存済みexp490 OOFを正解TVT付きで分解し、強いwell・弱いwellの誤差形状と条件を調べる
診断実験です。公開notebook由来のprefix fade-inと、prefix特徴から補正量を変える方策も
outer-fold-safeに評価します。

## 状態

- Route: `ensemble`
- 状態: `completed_diagnostic_no_inference`
- 親: `exp490_geometry_centered_mean_reverting_offset_hmm`
- 比較: 保存済み`exp357_parent_prediction`

## 仮説

exp490の平均回帰補正は、parentが大きく外れていて補正方向が正解方向と整合するwellでは
強い一方、補正方向が逆、誤差biasが持続、またはprefix直後の不安定さが後半まで残るwellで
弱い。公開notebookのwarm-up fadeとprefix情報に基づく補正量変更で一部を緩和できる可能性がある。

## 検証方針

- truth-aware説明: well、fold、suffix depth、誤差bias/drift、連続悪化、target-free特徴、
  archetypeを集計する。
- fold-safe方策: 29 fade profileをouter 4 foldsで選びheld foldへ適用する。prefix/context
  alpha treeもouter-train wellだけでfitする。
- 楽観上限: early suffix truthで選んだprofileのlate transferを測り、CVとは分離する。

## 制約

exp490/exp357の予測、fold、候補生成は固定です。HMM/PF/Beam/GPU再実行、inference、
submissionは行いません。early-suffix truthを使う評価は、masked-prefix replayの価値を
測る楽観的上限であり、実運用可能なCVとは扱いません。

## 所見

- exp490はexp357がwhole-well biasで大きく外れた難しいwellに強く、parentが既に良い
  wellへ逆向き補正を入れると数千row続くcatastrophic biasになる。
- tau=500固定fadeは5/5 foldsで`0.033123 ft`の探索的小改善。ただし元のtailは残る。
- prefix alpha treeは平均改善してもfold 3とwell tailを悪化させるため不採用。
- early truthから後半へのtransferが弱く、masked-prefix HMM replayは行わない。

## 実行入口

- compact train source: `exp503_exp490_strength_weakness_prefix_policy_readout_compact_selfcontained_train.py`
- inference guard: `exp503_exp490_strength_weakness_prefix_policy_readout_compact_selfcontained_inference.py`
- 完了output: `kaggle/output/train_v3`
