# exp046_hidden_branch_surrogate_audit

## 状態

- ルート: pf_beam
- 状態: completed
- CV: 14.313668
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-09
- 親実験: exp045_public_pf_meta_strict_parity_audit

## 仮説

public sample の `changed_rows=0` は visible train well 用処理が発火した結果であり、見えない test well 用処理の安全性を示さない。public sample の 3 well は train 由来なので、本番採点で使う処理が動かないことがある。train well の途中以降を隠した `exp029` の疑似 test 生成物上で見えない test well 用処理を強制適用すれば、code submit 前に変更量、予測範囲、bucket/層化 fold の破壊的悪化を確認できる。

## 検証方針

`exp029` の public sel15 PF/Beam 生成物を読む。この生成物は train well の途中以降を隠し、本番 test 風に予測させたもの。original-fold、well-hash、stratified-group fold の各 split で `exp026` anchor と residual/meta branch を fold-safe に再生成する。結果は overall、distance bucket、split、metadata bucket、well、candidate diff として保存する。提出ファイルは生成しない。

## 所見

Kaggle train version 1 で full audit 完了。`exp035/045` 相当の meta residual が代理面では最良だが、既知 Public LB では失敗済みのため、同系統の見えない test well 用処理の追加チューニングは採用しない。

## 参照ファイル

- 設定: `config.yaml`
- 監査スクリプト: `hidden_branch_surrogate_audit.py`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp046_hidden_branch_surrogate_audit_train.ipynb`
- 推論 notebook: `exp046_hidden_branch_surrogate_audit_inference.ipynb`

## 読み方

この README は実験フォルダの入口です。仮説、変更点、実行コマンド、出力、失敗理由、次のアクションは `SESSION_NOTES.md` と `result.md` を正とします。
