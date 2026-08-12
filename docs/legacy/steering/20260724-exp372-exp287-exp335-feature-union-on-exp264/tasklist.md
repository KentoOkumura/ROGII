# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- `exp372_exp287_exp335_feature_union_on_exp264`を採番した。
- steering requirements/design/tasklistを作成した。
- design-only experiment scaffoldを作成した。
- `config.yaml`にroute、lineage、入力SHA、444特徴、gate、再現性、15-booster costを固定した。
- 実験README、SESSION_NOTES、result、metricsをdesign-only状態で記録した。
- `KAGGLE_DIRECTION.md`へ統合backlogを追加した。
- `experiment_summary.md`へdesign-only実験を再生成・記録した。
- strict experiment validationとdesign contract auditをPASSした。
- 2026-07-24の明示指示により実装承認を記録した。
- compact self-contained Jupytext train候補と変換先Notebookを別名で作成した。
- exp264/exp287/exp335保存surfaceを結合する`src.feature_union_pipeline`を実装した。
- manifest/partition SHA、formation logical SHA、fold/role alignment、444列順序、
  15 model slot、technical/incremental/tail AND gateの専用testを実装した。
- 親compact notebookと章立て・記載量を比較した。
- Jupytext `--test`、py_compile、Ruff F821、専用pytest、strict validationを実行した。
- 2026-07-25の明示指示により、正規Notebook採用、Kaggle package/push、
  1 variant・3 configs・5 folds・15 boostersのT4 train run承認を記録した。
- exp264 control、exp287/exp335 standalone、selector再学習0をpush前に再確認した。
- compact self-contained Jupytext候補を18-cell正規train Notebookへ採用した。
- package/bootstrap/metadata SHAを固定し、canonical IDへT4指定でversion 1をpushした。
- pull-back metadataとembedded config/pipeline SHAを照合し、push後run flagをdisarmした。
- version 1はprefit parent compact loaderの`KeyError: compact_features`で停止し、
  booster開始数0を確認した。
- verifierの`features`をexp264 loaderの`compact_features`へ変換するadapterと
  専用回帰testを追加し、専用9 tests、関連44 tests、py_compile、RuffをPASSした。
- 2026-07-25の明示指示により、同じ15-booster契約のversion 2 technical retry承認を
  記録した。inference、submission、same-OOF rescueは未承認のまま。
- 修正版packageのbootstrap/metadata SHAを固定し、同一canonical IDへT4指定で
  version 2をpushした。pull-backのID/T4/7 sources/embedded SHAを照合した。
- Kaggle T4 version 2を完了まで監視し、15/15 boosterの完走を確認した。
- pooled CV `8.071563864946972`、fold/scope/by-well/importanceを記録した。
- technical gate PASS、incremental utility / tail promotion / promotion gate FAILを確定した。
- OOF、metrics、model/reproducibility manifestを取得し、主要10成果物と15 model fileの
  SHA一致を確認した。
- 事前契約どおり`close_without_same_oof_rescue`としてbranchを閉じた。
- inferenceとsubmissionは未実施のまま閉じ、同一OOF救済も行わない。
- `KAGGLE_DIRECTION.md`の完了済みbacklogを削除し、判断メモへ結果を移した。
- 実験文書、metrics、experiment summaryを最終状態へ更新した。
- 2026-07-25の明示指示を、科学gate FAIL後のsaved-model CPU inference overrideとして
  記録した。外部competition submit、再学習、same-OOF rescueは未承認のまま。
- 40 parent selector / 20 signed selector / 15 union TVT model、0 fitの推論契約と、
  raw-test 444列再生成、fail-closed条件、SHA記録をsteering/configへ固定した。
- inference version 1–3のtechnical errorを、科学gateやmodel契約を変えず修正した。
- Kaggle CPU inference version 4を459.376秒で完了し、14,151 rows / 3 wells、
  12 candidates、40/20/15 saved models、444 features、0 fitを確認した。
- outputを一時取得し、prediction/feature/formation/model slot/SHAを監査した。
- skill checkerとrepository checkerでsubmit-check PASS、WARN/FAIL 0を確認した。
- competition submitは未承認のため実行せず、実験文書と台帳を確定した。
- 2026-07-26のユーザーscoring完了連絡後、最新Code submission
  `ref=54975325`をKaggle CLIとmonitorで照合した。
- status `COMPLETE`、Public LB `7.587`を実験文書、metrics、提出台帳、
  experiment summary、戦略メモへ記録した。
- exp335 `7.517`より悪いためML Public-LB anchorを更新せず、train科学gate FAIL、
  same-OOF rescue禁止、非昇格判断を維持した。
