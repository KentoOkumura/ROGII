# exp122_exact_override_negative_control

## 状態

- ルート: pf_beam
- 状態: completed_kaggle_train_v1
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-25
- 親実験: `exp079_public_artifact_replay_integrity_audit`、`exp064_train_test_well_id_assert_probe`

## 仮説

Pilkwang replay の exact-match recovery / guarded overlap override は、hidden-safe な改善根拠として使うべきではない。既存の exp079 branch audit と exp064 well-id overlap probe、および利用可能な guard output を集約し、発火有無と prediction diff を negative control として記録する。

## 変更点

- `exact_override_negative_control.py` を追加し、notebook risk hits、source flags、exp079 summary、exp064 metrics、guard output inventory を集約する。
- train / inference notebook はモデル学習や submission 生成を行わず、監査 summary と metrics を保存する。

## 検証方針

- Fold: なし
- Group: なし
- Stratification: なし
- Leakage Check: optional same-well exact/override layer を改善根拠から除外すること自体を検証対象にする。

## 実行入口

- 学習 notebook: `exp122_exact_override_negative_control_train.ipynb`
- 推論 notebook: `exp122_exact_override_negative_control_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp122_exact_override_negative_control`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 対象外 |
| Public LB | 対象外 |
| Private LB | 対象外 |

## 所見

### 良かった点

- Kaggle train v1 で `negative_control_passed_current_evidence`。Pilkwang final は archived base branch と一致し、exp064 hidden code submission は exposed well-id overlap assertion non-trigger。

### 悪かった点

- Kaggle train v1 output を正とする。

### リスク / 注意

- 同じ物理 well が別 anonymized id で出る可能性はこの監査でも否定しない。
- optional layer が発火した場合も diagnostic only とし、submit 候補にしない。

## 次

- 完了。追加提出や inference run は不要。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
