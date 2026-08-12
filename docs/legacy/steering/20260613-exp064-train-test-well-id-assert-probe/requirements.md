# 要件

## 依頼

`train_test_well_id_assert_probe` を実装し、hidden test に train と同じ `well_id` が含まれるかを Kaggle run status だけで確認できるようにする。

## 制約

- Route: `pf_beam`
- hidden test 実行時の overlap well 数、対象行数、id hash、予測差分、詳細ログは取得できない前提にする。
- 公開 sample は train 由来の 3 wells (`000d7d20`, `00bbac68`, `00e12e8b`) と overlap するため、公開 sample 実行は落とさない。
- hidden / private test と判定した場合だけ、train/test `well_id` overlap があれば generic assertion で落とす。
- LB score を使った数値最適化にはしない。

## 受け入れ基準

- `config.yaml` に route、既知 public sample wells、assert policy が明記されている。
- inference notebook が公開 sample の既知 overlap を許可し、その他の test set で overlap があれば `AssertionError` を出す。
- hidden run で取得不能な件数や id を記録する前提がない。
- notebook が成功した場合は `submission.csv` を生成する。
- `task` がない環境では `make validate-exp` / `make prepare-kaggle-notebooks` で検証できる。
