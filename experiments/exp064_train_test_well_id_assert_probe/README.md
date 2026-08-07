# exp064_train_test_well_id_assert_probe

## 状態

- ルート: pf_beam
- 状態: completed
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: 53627058
- 作成日: 2026-06-13
- 親実験: `KAGGLE_DIRECTION.md` の `train_test_well_id_assert_probe`

## 仮説

hidden / private test に、train と同じ horizontal-well ファイル名 prefix として公開される exact `well_id` が含まれるなら、no-overlap assertion が失敗する。含まれないなら assertion は通り、placeholder `submission.csv` が生成される。この probe は、同じ物理 well が別 filename / anonymized id で公開されるケースまでは検出しない。

## 変更点

- `well_id_assert_probe.py` を追加し、horizontal well filename から `well_id` を抽出する。
- 公開 sample の既知 3 wells (`000d7d20`, `00bbac68`, `00e12e8b`) は overlap を許可する。
- public sample 以外の test set で train/test の exposed filename-prefix `well_id` overlap があれば `AssertionError("HIDDEN_TRAIN_TEST_WELL_ID_OVERLAP_DETECTED")` を出す。
- 成功時は sample submission を `submission.csv` にコピーする。予測品質を測る実験ではない。

## 検証方針

- Fold: なし
- Group: `well_id`
- Stratification: なし
- Leakage Check: hidden test の内部件数や id は取得できない前提。run status だけを診断に使う。

## 実行入口

- 学習 notebook: `exp064_train_test_well_id_assert_probe_train.ipynb`
- 推論 notebook: `exp064_train_test_well_id_assert_probe_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp064_train_test_well_id_assert_probe`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Kaggle inference run | `kentookumura/exp064-train-test-well-id-assert-probe-inference` v1 public sample completed |
| Probe status | `public_sample_overlap_allowed` |
| Code submission ref | `53627058` |
| Code submission status | complete |
| Public LB | 11551.955 |
| Scoring test assertion | not triggered |

## 所見

### 良かった点

- hidden test の overlap 件数、対象行数、id hash、予測差分を記録する前提を置かない。
- 公開 sample の既知 overlap で kernel version 作成が落ちないように分岐する。

### 悪かった点

- failed submission でも提出回数や実行枠を消費する可能性がある。
- competition policy / leakage 解釈リスクがあるため、実行前にルール確認が必要。

### リスク / 注意

- hidden success は「exposed filename-prefix overlap が検出されなかった」、hidden failure は「exposed filename-prefix overlap が検出された可能性が高い」という status 診断に限定する。同じ物理 well が別 id で出る可能性は否定しない。
- LB score の数値最適化には使わない。
- 2026-06-13 の code submission ref `53627058` は complete。placeholder zero submission のため Public LB 11551.955 はモデル性能として扱わない。

## 次

- train/test same exposed-`well_id` 前提の static replay / visible override は優先度を下げる。
- 見えない新規 well 用の hidden branch、public replay integrity audit、PF confidence residual clip を優先する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
