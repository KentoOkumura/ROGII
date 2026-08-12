# タスクリスト

## 完了

- [x] exp504、exp413、exp264、exp502のlineageとnested splitを確認する。
- [x] `docs/06_reproducibility.md`を確認する。
- [x] 45列compact schema、数式、順序、dtypeを固定する。
- [x] outer 5 × inner 4のstrict nested生成と25 partition契約を固定する。
- [x] 実行量を20 CPU rank boosters + 15 GPU TVT boosters、control再学習0に固定する。
- [x] promotion gate、禁止事項、FAIL後の停止条件を固定する。
- [x] steering、実験scaffold、backlog、experiment summaryを設計状態で作成する。
- [x] exp504 version 1のblock / pair / row metadata / target-free配列を取得し、file / logical SHAを固定する。
- [x] Jupytext percent形式のcompact self-contained train候補を別名で実装する。
- [x] 45列schema builderとnested leakage contract testを実装する。
- [x] Jupytext round-trip、構文、Ruff F821、専用test、strict validateを通す。

## 実行と停止判定

- [x] Stage N用Kaggle CPU packageを作り、push前にvariant/config/fold/booster数を再確認する。
- [x] Stage Nの20 CPU boostersを実行し、25 partitionsとSHAを記録する。
- [x] Stage N technical PASS後、Stage D 15 GPU boostersの実行承認を確認する。
- [x] Stage Dを実行し、保存exp413 OOFとのall-AND gateを判定する。
- [x] Stage D scientific FAILのためhidden-dynamic inferenceを設計・実装しない。
- [x] Stage D scientific FAILのため提出へ進まない。

## 終端状態

- Stage N package/push/runと正規train Notebook採用は2026-08-03に完了し、technical PASS。
- Stage D version 1は15 GPU modelsを完走し、technical PASS / scientific FAIL。
- pooled、fold、scopeの性能3条件をすべてFAILしたため、same-OOF rescue、推論、提出は禁止。
