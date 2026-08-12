# 設計

## アプローチ

`exp034` の train-side meta stack audit を土台に、見えない test well 用処理を train well の途中以降を隠した疑似 test rows に強制適用する代理監査を作る。`exp029` の PF/Beam feature 生成物を読み、audit split ごとに `exp026` pseudo-tail bucket-shrink anchor を fold-safe に再生成する。その上で、固定 PF blend、PF residual、exp026+PF meta residual を見えない test well 用処理候補として比較する。

## 実験範囲

- 対象実験: `exp046_hidden_branch_surrogate_audit`
- Route: `pf_beam`
- 親実験: `exp045_public_pf_meta_strict_parity_audit`
- 入力: `exp029_public_sel15_pf_oof_feature_generation/features/public_sel15_pf_oof_features.csv.gz`
- 変更する変数: visible train branch を使わず、見えない test well 用処理候補を train well の途中以降を隠した疑似 test rows に強制適用して監査する。
- 固定する変数: exp026 pseudo-tail 手順、exp029 PF/Beam feature、exp031/033/035/045 の既知候補設定。

## 監査出力

- overall RMSE と reference delta
- candidate vs reference diff RMSE、変更行数、変更 well 数、予測範囲
- row 距離 bucket、audit split、exp044 由来層化 fold / metadata bucket の segment RMSE
- well-level RMSE と reference delta
- exp026 anchor 再生成 source summary

## リスク

- リークリスク: `target_tvt` は残差学習の supervised target と scoring にだけ使う。visible train oracle surrogate は差分診断のみで、特徴量には入れない。
- CV/LB 不一致リスク: exp031/033/035/045 は代理面で良く見えても Public LB で悪化済み。summary に既知 LB 失敗を併記し、代理監査を submit 許可の十分条件にしない。
- ランタイム/メモリリスク: full 実行は exp026 anchor 再生成を split system ごとに行うため重い。ローカルは `--max-wells` smoke に限定し、正式実行は Kaggle train notebook を正とする。
