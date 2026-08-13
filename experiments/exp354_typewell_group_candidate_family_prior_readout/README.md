# exp354_typewell_group_candidate_family_prior_readout

## 状態

- ルート: `ml_model`
- 状態: Kaggle CPU Stage 0完了、real-minus-shuffle gate FAIL、branch closed
- CV: Stage 0 gate 9/10 PASS、総合FAIL
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-23
- 科学的親: `exp293_physics_only_candidate_bank_headroom_contract`
- downstream control: `exp264_exp263_candidate_confidence_dual_selector`
- 履歴参照: `exp316_typewell_group_candidate_family_error_prior`

## 仮説

Type Well群のGR calibration/emissionが失敗しても、固定物理candidate familyごとの得意不得意には
群固有の差が存在する可能性がある。outer-trainだけで作ったgroup×family soft error priorが
held-out wellのfamily順位を再現するかを、selector学習前の0-model readoutで検証した。

## 変更点

- exp311/312/313/315出力への依存を削除した。
- 入力をexp293 v2の固定deployable12 candidate bankと6 familyへ限定した。
- support 10のsoft prior、global/neutral fallback、stable group-label shuffleを固定した。
- prior/fallback/shuffleをouter-valid error結合前にSHA固定した。
- Stage 0はreal prior 1 + shuffle 1 / 5 reporting folds / model・booster各0。

## 検証方針

- Fold: exp293固定outer 5 folds。
- Primary: shrunk family RMSE対held-out family RMSEのwell内Spearmanをwell等重み集約。
- Negative control: fold内stable group-label shuffle。
- Gate: Spearman 0.15、4/5 folds、real-minus-shuffle 0.05、
  coverage 0.90、hidden-like 2面非負。
- Leakage check: candidate/family/fold、group membership、shuffle、prior scheduleを
  held-out truth join前にfreezeし、fit-valid well overlap 0を要求する。

## 実行入口

- 正規train:
  `exp354_typewell_group_candidate_family_prior_readout_train.ipynb`
- Jupytext参照元:
  `exp354_typewell_group_candidate_family_prior_readout_compact_selfcontained_train.py`
- Kaggle kernel:
  `kentookumura/exp354-typewell-family-prior-readout-train` version 1
- inferenceはfail-closed、Stage 1とsubmissionは未実装のまま。

## 結果

| メトリック | 値 | 判定 |
| --- | ---: | --- |
| family rank Spearman | 0.325789 | PASS |
| positive folds | 5/5 | PASS |
| held-out group coverage | 0.980595 | PASS |
| hidden-like spatial Spearman | 0.381736 | PASS |
| hidden-like typewell-purged Spearman | 0.376570 | PASS |
| shuffle Spearman | 0.327079 | control |
| real minus shuffle Spearman | -0.001290 | **FAIL** |
| Stage 1 eligible | false | closed |

## 所見

### 良かった点

- target-free freeze前のtruth rowは0、fit/valid well overlapも0で、leakage guardを通過した。
- real prior単体のfamily rankは5/5 folds正方向で、hidden-like 2面も非負だった。

### 悪かった点

- group labelを安定shuffleしても同じSpearmanが残り、native Type Well群の増分signalを示せなかった。
- observed rank signalは主にglobal family base rateで説明でき、group固有priorのStage 1根拠にならない。

### リスク / 注意

- 同じreadoutでfamily、support、group、rank metricを救済調整しない。
- Stage 1の40 selector models、inference、submissionへ進まない。

## 次

- exp354 branchは救済grid・再実行なしで閉じる。
- 独立仮説のexp353 quality featureは本結果だけで閉鎖しないが、自動昇格の根拠にも使わない。
- 次の実験は既存backlogから別仮説として判断する。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、
実験名や設定名を除いて日本語優先で記録する。
