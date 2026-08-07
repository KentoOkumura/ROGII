# タスクリスト

## 完了

- [x] 最新実験番号を確認し、対象を`exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264`に固定する。
- [x] exp264 fixed12 selector、exp496 fixed13前例、exp490 full OOF、exp499 well routerを確認する。
- [x] exp490入力ファイル、row/well count、raw gzip / decompressed content SHAを固定する。
- [x] fixed12→fixed13の単一変更、target-free allowlist、phase分離、40 CPU booster契約を確定する。
- [x] technical / leakage / dual score / integration / tailの全AND gateを事前登録する。
- [x] reranking、same-OOF rescue、exp498/499特徴、current-test、inference、submissionの禁止範囲を固定する。
- [x] 再現性設計を`design.md`に記入する。
- [x] design-only実験scaffold、`KAGGLE_DIRECTION.md` backlog、`experiment_summary.md`を登録する。
- [x] YAML/JSON、実行量算術、実験文書を検証し、`validate-exp` strict PASSを確認する。

## 実装完了

- [x] exp496 compact self-contained trainを構成参照にし、exp501の別名Jupytext percent形式候補を作る。
- [x] exp490 allowlist / SHA / global-key join / truth-late phaseを検証する専用contract testを作る。
- [x] 13候補feature schema、nested outer5 × inner4、compact / score / diagnostic出力を実装する。
- [x] `py_compile`、Ruff、Jupytext test、専用tests、`validate-exp --strict`を実行する。

## 実行完了

- [x] ユーザーの実行承認後、正規train Notebookを採用してstrict Kaggle packageを作成する。
- [x] Kaggle CPU Stage A/Cを1 variant / 2 objectives / 40 selector boostersで完走する。
- [x] technical / leakage / selector score / integration / tail gateを固定契約どおり判定する。
- [x] version、runtime、CV、fold/scope/by-well、SHA、decisionを実験記録へ反映する。

## terminal close（実行しない）

- [ ] current-test candidate/feature生成とselector inference port。
- [ ] downstream TVT学習。
- [ ] competition inference / submission。
- [ ] same-OOF rescue、weight / threshold / domain / feature / candidate subset / gate調整。

## ブロック条件

- 実装は2026-08-01のユーザー明示承認に基づき完了した。
- Kaggle runは2026-08-01のユーザー明示承認に基づきversion 2で完了した。
- 親control再学習、HMM/PF/Beam再生成、GPU学習はこの実験では行わない。
- by-well p95 / worst gateをFAILしたため、pooled / 5 folds / 7 scopes改善で救済しない。

## 次のアクション

`FAIL_CLOSE_EXP490_MEAN_REVERTING_HMM_FIXED13_SELECTOR`でbranchを閉じる。
current-test、downstream TVT、inference、submissionへ進まない。
