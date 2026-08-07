# exp367_stratified_signed_curvature_pf

## 状態

- ルート: `pf_beam`
- 状態: `completed_stage0_failed_close_without_rescue`
- CV / LB / Submit: なし
- 作成日: 2026-07-23
- 親実験: `exp072_exp063_full_replay_feature_cache`
- Kaggle train: version 1 / id_no `128528103` / `267.914282461 sec`

## 仮説

GRがsigned-curvature dynamicsを識別でき、resampling後も各符号の粒子を残せるなら、
exp072 PFの単一rate modeよりmode slipに強くなる。

## 変更点

- 粒子状態を `(position, rate, c)`、`c=-1/0/+1`へ拡張する。
- 500粒子の初期層を`100/300/100`、resample後の各符号最低数を50に固定する。
- 128 seeds、GR likelihood、noise、momentum、particle総数はexp072のまま。

## 検証方針

- Stage 0はPFを回さず、独立に凍結した3本のsigned pathをGRが識別するか読む。
- Stage 1は全gateと別承認後だけ500 particles × 128 seeds × 773 wells。
- 保存済みexp072 `likpf_mean`をcontrolとし、control PFは再実行しない。

## 実行入口

- `exp367_stratified_signed_curvature_pf_compact_selfcontained_train.py`
- `exp367_stratified_signed_curvature_pf_train.ipynb`

train notebookはStage 0だけを実装する。`execution.run_stage_0=false`か
`execution.kaggle_push_approved=false`なら実行前に停止する。inference notebookは
Stage 1未実装を明示して必ず停止するfail-closed guardであり、sample submissionや予測を
生成しない。

## 結果

Kaggle private CPU Stage 0はtechnical gateをすべてPASSしたが、scientific gateをFAILした。

- overall top1: `0.469591`（PASS）
- MRR gain vs zero-first: `+0.276771`（PASS）
- real - circular top1: `+0.005576`（FAIL、必要`>=0.03`）
- passing folds: `2/5`（FAIL、必要`>=4/5`）
- 1000+ / hidden-like spatial / hidden-like typewell-purgedのRMSE方向: 3面ともPASS
- decision: `stage_0_failed_close_without_rescue`
- Stage 1 eligible: false

## 所見

### 良かった点

- PF固有のparticle impoverishmentへ、総粒子数を増やさない固定quotaで対処する。
- truth-free path / score freeze、SHA readback、late truth / hidden-like joinを実装した。
- 3固定軌道、circular control、fold / 1000+ / hidden-like gateをnotebook上で追跡できる。

### 悪かった点

- exp242/273では動的state追加が全距離帯とworst-wellを悪化させた。
- real GRのtop1はcircular controlを`0.005576`しか上回らず、fold一貫性も2/5だった。

### リスク / 注意

- quotaが弱い符号を過剰維持して精度を落とす可能性がある。
- quota、transition、curvature、particle/seed数のgrid searchは禁止する。

## 次

exp367は事前契約どおり救済なしで閉じる。Stage 1 PF、inference、submissionへ進まず、
quota / curvature / transition調整など同一結果上のparameter rescueも行わない。
