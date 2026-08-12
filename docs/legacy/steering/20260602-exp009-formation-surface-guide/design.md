# 設計

## アプローチ

formation columns は train horizontal CSV にだけ存在するため、直接 feature columns へ入れない。
代わりに train fold の horizontal rows から `X,Y -> formation surface Z` の KNN regressor を fit し、対象 well の evaluation zone では推定 surface と現在行の `Z` の距離を特徴化する。

各 evaluation row で作る特徴:

- formation ごとの推定 surface Z
- formation ごとの `Z - surface_z` と絶対値
- 最近傍 formation surface までの signed / absolute distance
- 全 formation surface に対する mean absolute distance と span

KNN guide は `build_drift_feature_frame` の optional input として渡す。`fit_drift_model_from_files` は model training files だけから guide を fit し、その guide で train frame を作る。CV valid frame と inference frame は、fit 済み model に保持された guide を使って作る。

## 実験範囲

- 対象実験: `exp009_formation_surface_guide`
- 親実験: `exp008_gr_ncc_matcher`
- 変更する変数: `model.formation_guide.enabled` と feature set (`*_plus_formation_guide`)
- 固定する変数: residual model、GroupKFold、sampling caps、residual shrink、GR NCC disabled

## リスク

- リークリスク: valid fold の formation columns を guide fitting に使うと CV が過大評価される。fold の train files だけを `fit_formation_guide_from_files` に渡す。
- CV/LB 不一致リスク: formation surfaces が public-like visible wells だけに合う可能性がある。control rows と well-level artifact で悪化 well を確認してから提出する。
- ランタイム/メモリリスク: 全 row KNN は重い。per-well / total sampling cap を設け、KNN は `X,Y` の 2D 入力だけで fit/query する。
