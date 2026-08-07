# exp353_typewell_group_quality_feature_preflight

## 状態

- ルート: `ml_model`
- 状態: Kaggle CPU version 1完了、固定Stage 0 gate FAIL、branch closed
- CV: Stage 0 5/8 checks PASS、総合FAIL
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-23
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 履歴参照: `exp314_label_derived_typewell_gr_quality_addonly`

## 仮説

Type Well群priorを直接補正へ使うとtailを壊しても、support/noise/reliabilityをsoftなML特徴として
与える用途には独立した価値があり得る。6列のfold-safe featureが保存済みexp148 OOF errorと
fold横断で関連することを、15 boostersを学習する前に0-boosterで確認する。

## 変更点

- 旧exp314をreopenせず、exp311/313のpromotionに依存しないStage 0 preflightへ切り出した。
- 6列schemaを固定し、結果後の列選択を禁止した。
- Stage 1はPASSと別承認時のみ15新規boosters、control再学習0とした。

## 検証方針

- Fold: exp148と同じwell GroupKFold 5 folds。
- Group: native Type Well content group。
- Stage 0: coverage/fallback/finite、well-RMSE Spearman、quartile lift、group shuffle差。
- Leakage Check: outer-valid well IDがprior fit tableへ0件であることをassertし、featureを先にSHA固定する。
- Stage 1予約: exp148へ6列add-only、lgb0/1/2 × 5 folds = 15 boosters、control 0。

## 実行入口

- 学習 notebook: `exp353_typewell_group_quality_feature_preflight_train.ipynb`
- 推論 notebook: `exp353_typewell_group_quality_feature_preflight_inference.ipynb`
- compact self-contained trainを正規train Notebookへ採用した。
- compact inferenceはfail-closedで、正規inference Notebookはplaceholderのままとする。
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| exact group coverage | 0.980595 |
| fallback fraction | 0.019405 |
| real residual sigma vs exp148 well-RMSE Spearman | 0.006134 |
| positive folds | 4/5 |
| exp148 well-RMSE q4-q1 | +0.202701 ft |
| real minus shuffle Spearman | -0.059166 |
| Stage 0 | FAIL |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- GPU学習前にfeature familyのtransfer性を安価に反証できる。
- exp148 GroupKFoldの重みを、true TVTを読まず`TVT_input`欠損3,783,989行から再構成する。
- exp065 membershipとexp148 summary/by-well controlをraw SHAで固定した。
- coverage、fallback、全feature finite、freeze前truth 0、相関の正方向4/5 foldsはPASSした。

### 悪かった点

- Stage 0相関があってもLightGBMのCV改善を保証しない。
- pooled Spearmanは`0.006134`で閾値`0.15`を満たさなかった。
- q4-q1は`+0.202701 ft`で閾値`+0.25 ft`未満だった。
- shuffle Spearman`0.065301`がreal`0.006134`を上回り、group固有signalを支持しなかった。

### リスク / 注意

- label-derived priorのouter-fold混入は強いleakになる。
- feature/prior manifestを凍結するまでexp148 by-well OOF errorを開かない。
- Stage 1の15 boostersは未承認かつStage 0 FAILのため実装・実行しない。

## 次

- 列選択、group/fallback/閾値救済、再実行を行わずbranchを閉じる。
- Stage 1、raw-test、inference、submissionへ進まない。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
