# exp150_formation_physical_imputer_revisit

## 状態

- ルート: MLモデル
- 状態: completed
- CV: 28.233897
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-27
- 親実験: exp138_ancc_surface_predictability_audit

## 仮説

Sunny 系の formation contact physical branch は、train-only formation columns を直接使うと hidden test で成立しない。  
ただし、train-fold wells だけから fold-safe に formation contact surface を推定し、各 well の既知 `TVT_input` prefix だけで offset calibration すれば、直接 TVT 置換ではなく弱い prior / confidence feature として使える可能性がある。

## 検証方針

`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` の surface を `global_median`、`row_knn_xy`、`well_plane_knn` で推定し、`contact_median`、`contact_prefix_weighted`、`contact_best_prefix` の prefix-calibrated physical TVT 候補を評価する。

## 所見

Kaggle train v1 完了。best は `well_plane_knn/contact_best_prefix` で RMSE 28.233897。直接 TVT 置換や target conditioning には使わず、`formation_pred_spread`、`neighbor_dist`、`prefix_mae_best` の confidence diagnostic に限定する。

## 参照ファイル

- 設定: `config.yaml`
- 実装: `formation_physical_imputer_revisit.py`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- 学習 notebook: `exp150_formation_physical_imputer_revisit_train.ipynb`
- 推論 notebook: `exp150_formation_physical_imputer_revisit_inference.ipynb`

## 注意

この実験は提出候補を生成しない。候補 TVT は監査用で、後続に使う場合も add-only confidence feature に限定する。
