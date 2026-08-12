# タスクリスト

## TODO

- なし。2026-06-16 のユーザー指示により exp068 は破棄。

## 進行中

- なし

## ブロック中

- なし

## 完了

- `exp068_equivalent_pixiux_inference_port` の steering を作成。
- `exp063` を親に experiment scaffold を作成。
- notebook / config / settings を exp068 名へリネーム。
- `exp063_branch_audit.py` を実装。
- train notebook を exp039 CV surface + exp063 tracker/PF/Beam output features による exp063 LightGBM 再学習評価へ変更。
- inference notebook を exp063 inference prediction artifact による branch inference audit へ変更したが、レビュー後に static artifact 依存として廃止。
- レビュー後、train notebook に full model artifact 保存を追加。
- レビュー後、inference notebook を exp068 full model artifact + hidden-test exp063 replay feature generation に変更。
- 2026-06-16 にユーザー指示で exp068 を破棄。代替 backlog `exp073_exp039_cv_reassessment` を作成する方針へ変更。
