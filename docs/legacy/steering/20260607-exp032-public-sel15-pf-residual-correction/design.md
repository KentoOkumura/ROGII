# 設計

## アプローチ

`exp029` の PF/Beam OOF-like row artifact を読む。この生成物は train well の途中以降を隠し、本番 test 風に予測させたもの。`target_tvt - pf_pred` を回帰し、推論時の大外しを抑えるため、モデル出力の residual は候補ごとに clip と shrink を適用してから `pf_pred` に足す。

評価は 2 系統に分ける。

- `leave_one_original_fold_out`: exp029 の `fold` を holdout とする OOF residual prediction。
- `well_hash_holdout`: `well_id` の stable hash 5-fold を holdout とする stress audit。

どちらも `public_pf_selector` と `pf090_hold010` を required controls とし、両方を上回った residual candidate だけを inference 移植候補にする。

## 実験範囲

- 対象実験: `exp032_public_sel15_pf_residual_correction`
- Route: `pf_beam`
- 親実験: `exp029_public_sel15_pf_oof_feature_generation`
- 変更する変数: PF residual model、residual shrink、residual clip
- 固定する変数: exp029 の PF/Beam feature artifact、distance bucket、required controls

## リスク

- リークリスク: `target_tvt` は residual target と scoring にのみ使い、error diagnostic columns は特徴量から除外する。
- CV/LB 不一致リスク: public sel15 replay は 見えない test well 評価の LB が非常に強いため、train-side train well の途中以降を隠した疑似 test CV 改善がそのまま LB 改善に直結しない可能性がある。
- ランタイム/メモリリスク: 1,782,279 rows を使うため、HGB は train row cap と bucket cap を設定する。
