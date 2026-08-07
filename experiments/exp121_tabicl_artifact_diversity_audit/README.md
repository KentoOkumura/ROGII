# exp121_tabicl_artifact_diversity_audit

## 状態

- ルート: ensemble
- 状態: audit_completed
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-25
- 親実験: `tabicl_artifact_diversity_audit` backlog
- 実行状態: Kaggle train v3 完了

## 仮説

TabICL / 保存済み artifact-stack 予測は、単体本命ではなく後続のアンサンブル候補として、既存 anchor との多様性材料に価値がある可能性がある。まずは exp027 / exp063 / exp073 / exp082 との差分、SHA、source mount 状態を target-free に監査する。

## 変更点

- `tabicl_artifact_diversity_audit.py` を追加し、candidate source root の探索、submission contract validation、SHA 記録、pairwise / by-well distance を CPU-only で保存する。
- TabICL 本体の再推論、GPU 実行、モデル学習、提出候補生成はしない。
- source が mount されていない場合も missing inventory として記録する。

## 検証方針

- Fold: なし。target-free submission diversity audit。
- Group: `well_id` は `id` prefix から by-well distance 集計に使う。
- Stratification: なし。
- Leakage Check: test submission 同士の距離だけを計算する。OOF error correlation は fold-safe OOF がある場合だけに限定し、test 真値は使わない。

## 実行入口

- 学習 notebook: `exp121_tabicl_artifact_diversity_audit_train.ipynb`
- 推論 notebook: `exp121_tabicl_artifact_diversity_audit_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp121_tabicl_artifact_diversity_audit`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Valid submissions | 7 |
| Candidate submissions | 3 |
| Anchor submissions | 4 |
| Pairwise rows | 15 |

## 所見

### 良かった点

- CPU-only で実行でき、候補 source の有無も監査結果として残せる。
- Kaggle v3 で TabICL / artifact-stack candidate 3 件と anchor 4 件を読み込めた。
- `needless` と `kojimar` は exp082 final source-port に近く、RMSE は 1.220332 / 1.447558。
- `thbdh v10 fresh artifact` は exp063 に最も近く、RMSE は 1.809928。

### 悪かった点

- CV や LB を直接改善する実験ではない。OOF 候補がなければ誤差相関は出せない。
- Kaggle v1/v2 では anchor kernel output が expected path に mount されず、v3 で anchor CSV を bootstrap input として同梱した。

### リスク / 注意

- external artifact / public notebook output の version と mount path に依存する。
- この実験単体では submit 判断をしない。位置づけは、単体提出候補ではなく後続のアンサンブル候補の監査である。

## 次

- 直接 submit はしない。OOF がある候補だけ、後続のアンサンブル / selector 診断で error correlation を確認する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
