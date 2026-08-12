# 要件

## 依頼

`exp001_baseline` を、現状の調査結果を反映した実行可能な初期ベースラインとして作成する。

## 制約

- `well_id` 単位の holdout を崩さない。
- 評価対象は `TVT_input` が NaN の tail rows のみ。
- 予測では各 well の既知 `TVT_input` prefix だけを使い、hidden tail の true `TVT` を参照しない。
- train-only formation columns (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`) は exp001 では使わない。
- まず提出可能な最小基準線を作り、NCC / formation surface / beam / PF は次実験に回す。

## 受け入れ基準

- `experiments/exp001_baseline/` に train / inference の実装がある。
- `GroupKFold` by well の CV が `metrics.json` に記録される。
- `last_known_TVT` anchor baseline の full CV を記録する。
- `sample_submission.csv` と同じ `id,tvt` の `submission/submission.csv` を生成できる。
- `SESSION_NOTES.md`、`result.md`、`experiment_summary.md` に結果と次アクションが残る。
