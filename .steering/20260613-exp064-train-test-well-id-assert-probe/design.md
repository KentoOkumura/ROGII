# 設計

## アプローチ

公開 sample の test wells は既知の 3 wells であり、すべて train にも存在する。この状態を許可しないと Kaggle kernel version 作成時点で失敗するため、次の 2 段階にする。

1. `test_wells == expected_public_test_wells` の場合は public sample と見なし、overlap が既知 3 wells と一致することだけを sanity check する。
2. それ以外の test set は hidden / private test と見なし、`train_wells & test_wells` が非空なら `AssertionError("HIDDEN_TRAIN_TEST_WELL_ID_OVERLAP_DETECTED")` で落とす。

hidden test の詳細ログや output は観測できない前提なので、記録できる結果は Kaggle run status / submission ref / 取得可能な error 種別だけに限定する。

## 実験範囲

- 対象実験: `exp064_train_test_well_id_assert_probe`
- Route: `pf_beam`
- 親実験: `KAGGLE_DIRECTION.md` の `train_test_well_id_assert_probe`
- 変更する変数: hidden / private test での train/test `well_id` overlap assertion
- 固定する変数: 予測モデル、PF/Beam 処理、特徴量、後処理は使わない

## リスク

- リークリスク: overlap の有無を status probe で確認する行為自体に competition policy / leakage 解釈リスクがある。実施前にルールを再確認する。
- CV/LB 不一致リスク: CV や LB 改善実験ではない。hidden success は「overlap なし」、hidden failure は「overlap あり」の yes/no にしか使わない。
- ランタイム/メモリリスク: CSV の中身は読まず、ファイル名だけを見るため小さい。
