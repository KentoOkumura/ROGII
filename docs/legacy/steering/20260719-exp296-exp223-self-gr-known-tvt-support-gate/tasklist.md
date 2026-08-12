# タスクリスト

## 目的

設計、compact実装、正規Notebook採用、Kaggle CPU実行を分離する。Kaggle CPU version 3まで完了し、performance guard FAILによりbranchを閉じた。

## 実装レビューで完了

- ベースはsaved exp223 `hmm_selfgr_boost_only_a070_c100`であり、唯一の変更はfull-grid self-GR contributionへinclusive known-TVT support maskを掛けることである。
- exp223のHMM/self-GR必要関数だけを統合した別名Jupytext compact self-contained train候補を実装した。
- exp223 surface/quality/config parityとsupport mask単一差分をcontract testで固定した。
- inclusive boundary、outside zero、inside bitwise parity、no-known-TVT neutral、all-true parent parity、final prediction no-clipをtestした。
- generation loaderを`MD/Z/GR/TVT_input`だけへ限定し、prediction/support/schema/decoder SHA freeze後にだけtruth/controlを読むreadoutを実装した。
- saved exp223 decompressed SHA/row identity、raw input/support/prediction/schema/metrics SHA、hard gateを実装した。
- 実行規模を1 variant / 773 HMM well-runs / LightGBM 0 / fold train 0 / booster 0 / control再実行0へassertした。
- Jupytext round-trip、py_compile、full Ruff、専用14 tests、repository `331 passed, 1 skipped`を通した。

## 実行・結果記録

- 別名compact Notebookの正規train Notebook採用とsource parity検証は完了した。
- canonical CPU/private/internet-off metadataとbootstrap内config/source SHA監査は完了した。
- 新variant 1本だけのKaggle CPU version 3を同一kernel idで完了まで監視した。
- logsからruntime、773-well coverage、technical/performance gate、fold/scope/by-well metricsを記録した。
- metrics/manifest/schemaの小規模13 artifactだけを取得し、summary記録SHAと全件一致を確認した。大容量prediction/OOFは取得していない。
- technical 12/12 PASS、performance 2/10 PASS、pooled delta `+0.809806 ft`のためFAIL-closeを確定した。
- result、metrics、SESSION_NOTES、experiment_summary、KAGGLE_DIRECTIONを完了状態へ更新した。

## 実行承認済み

- canonical Notebook採用: 2026-07-19承認済み。
- Kaggle CPU push: 2026-07-19承認済み。`run_variant=true`、`kaggle_cpu_push_approved=true`。
- inference/submission: performance guard FAILによりbranch closed。実行しない。

## 次のアクション

exp296 hard-gate branchは閉鎖済み。padding、soft/hole-aware gate、alpha/clip/window/top-k/threshold救済、inference、submissionは行わない。独立したfeature-only案にはexp296の証拠だけを引き継ぐ。

## 完了

- `kaggle-review-exp`と`docs/06_reproducibility.md`を確認した。
- exp223の現行self-GR条件とsupport hard gate欠如をコードで確認した。
- exp225のknown-range gateがdescriptor motif単独差分ではないこととnegative resultを確認した。
- `docs/legacy/steering/20260719-exp296-exp223-self-gr-known-tvt-support-gate/`を作成した。
- `experiments/exp296_exp223_self_gr_known_tvt_support_gate/`をtemplateから作成し、親コードをコピーしていない。
- support maskの入力、数式、適用順序、inside parity、outside zero、neutral fallbackを固定した。
- planned countを1 variant / 773 HMM well-runs / LightGBM config・trained fold・booster `0/0/0` / control再実行0に固定した。
- technical/performance/subgroup/worst-well hard gateとFAIL-closeを固定した。
- `KAGGLE_DIRECTION.md`の未着手backlogへ、低-中・0-booster・compact実装済み/実行承認待ちとして反映した。
- JSON/YAML contract、strict experiment validation、project template validationを通した。
- `experiment_summary.md`を更新し、parent lineageと`design_locked_not_implemented`を反映した。
- `kaggle-review-exp` reviewerで全対象文書のcore evidence categoryが揃っていることを確認した。
- 正規scaffold train/inference Notebookと`settings.py`が、生成時の実験名置換以外templateと一致することを確認した。
- 別名compact train source/Notebookと専用testsだけを追加し、package、train、inference実装、submissionを行っていない。
- canonical Kaggle CPU version 3で1 variant / 773 HMM well-runs / 0 boosterを16,667.265秒で完走した。
- technical guard 12/12 PASS、performance guard 2/10 PASSを確認し、strict known-TVT support gateを棄却した。
- inference/submissionと事前禁止したrescue gridを実行せずbranchを閉じた。
