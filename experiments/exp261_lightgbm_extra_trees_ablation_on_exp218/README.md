# exp261_lightgbm_extra_trees_ablation_on_exp218

## 状態

- ルート: `ml_model`
- 状態: Kaggle train v1完了・guard fail・不採用
- CV: `8.755217124`（親`8.475793752`から`+0.279423372`悪化）
- Public LB: 未実行
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-16
- 親実験: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`

## 仮説

exp218の回帰LightGBMで `extra_trees=True` を有効にすると、380-feature surfaceやその他の
ハイパーパラメータを変えずにsplit thresholdの多様性が増え、OOF汎化または親OOFとの
補完性が改善する可能性がある。

## 変更点

- 選択したexp218 LightGBM configへ `extra_trees=True` だけを追加する。
- `full_family`（3 configs × 5 folds = 15 boosters）を単一train notebookで実行した。
- exp218の保存済み15 boostersを同一foldでcontrol推論し、control自体は再学習しない。
- config差分が `extra_trees` だけであることをruntime assertionする。
- 親OOFとの相関、固定blend、distance / hidden-like / fold / by-well / worst-wellを保存する。

## 検証方針

- Fold: exp218と同じwell `GroupKFold` 5 folds。
- Group: `well`。
- Stratification: なし。
- Leakage Check: exp218のknown-prefix/target-free feature契約を固定し、exp115 roleはstress評価にのみ使用する。
- Primary comparison: 選択configの保存済みexp218 booster OOF対 `extra_trees=True` OOF。
- GPU cost guard: `model.selected_plan`、`model.run_approved`、approval scopeが揃うまで学習前に停止する。

## 実行入口

- 学習 notebook: `exp261_lightgbm_extra_trees_ablation_on_exp218_train.ipynb`
- 推論 notebook: `exp261_lightgbm_extra_trees_ablation_on_exp218_inference.ipynb`（train guard通過まで意図的に停止）
- Kaggle準備: `task prepare-kaggle-notebooks EXP=exp261_lightgbm_extra_trees_ablation_on_exp218`
- notebook実行: Kaggle kernel runを正とし、ローカルnotebook実行はしない。

## 結果

| メトリック | 値 |
| --- | --- |
| 親3-config mean CV | 8.475793752 |
| `extra_trees=True` 3-config mean CV | 8.755217124 |
| delta | +0.279423372 |
| 改善fold | 1/5 |
| 1000+ delta | +0.315774718 |
| hidden-like spatial / typewell-purged delta | +0.243318548 / +0.250390215 |
| worst-well regression | +11.324423 |
| adoption guard | 全項目false |
| Public LB | 未実行 |
| Private LB | - |

## 所見

### 良かった点

- 親control再学習なしで保存済みboosterからmatched OOFを復元し、親3-config平均と誤差0で一致した。
- 380 features、15 models、parameter差分`extra_trees`のみ、評価CSV/JSON SHAを確認できた。

### 悪かった点

- lgb0/lgb1/lgb2がすべて悪化し、3-config meanは`+0.279423372`悪化した。
- 1000+、hidden-like 2面、worst-well、fold stabilityを含む全guardが不通過だった。
- 親とのOOF相関が`0.9999955`と非常に高く、fixed blend 0.25でもoverallは`+0.031917897`悪化した。

### リスク / 注意

- LightGBM GPUはrerunなしにbitwise deterministicと見なさないが、全config・主要stress面で一貫して悪化しておりrescue rerunは行わない。
- selector LightGBMは目的と学習surfaceが異なるため、exp262で独立評価する。

## 次

- 回帰variantは不採用。inference / submission / parameter gridへ進めない。
- near bucketの小さなblend改善を確認する場合も保存OOFだけの0-booster readoutに限定する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
