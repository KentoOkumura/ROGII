# タスクリスト

## TODO

- corrected `downstream_mean4_only` 1 variant × 3 configs × 5 folds = 15 GPU boostersを実行する場合、
  保存済みversion 2 compactを入力とし、control再学習0で別途明示承認を得る。

## 進行中

- なし。

## ブロック中

- corrected mean4のscore/leakage/schema guardはPASS済み。`downstream_mean4_only`、mean8/both、
  `aggregate_compare`はそれぞれ新しい実行承認なしに進めない。

## 完了

- `docs/06_reproducibility.md`確認。
- exp271 candidate SHA、exp263/264 selector/downstream契約、GPU cost guardの設計固定。
- PF candidate join、既存`pf_ancc`差し替え3 variant contract、nested selector、fixed-control downstream、aggregate実装。
- canonical train/inference notebook生成とJupytext test。
- targeted 22 tests、full 147 tests、strict experiment validation、template validation。
- CPU/internet-offのKaggle package準備とversion 1 push。
- 2026-07-18、ユーザーから`nested_selector_mean4_only` 40 CPU boostersの実行承認を取得。
- `nested_selector_mean4_only` version 1をCPU 40 boostersで完走。40 model、25 compact partitions、
  model/partition SHA、12候補の`pf_ancc`→mean4差し替えを技術確認。
- 出力schemaにtraining-only formation raw/delta 12特徴が残ることを確認し、score guardを含むrun出力を
  quarantine。local run gateを閉じ、追加variant/downstreamを中止。
- 修正版exp264 Stage A v4 / Stage C v6の88特徴、raw context `MD/X/Y/Z/GR`、40 model SHA、
  compact manifest SHAを監査し、親再構築の完了を確認。
- exp277をraw-test-only schema gate、clean 273 allowlist、修正版Stage D v3 control SHA/RMSEへport。
- train notebookを更新し、actual train/current-test header auditと273+74=347列guardを追加。
- corrected `design_only` / CPU / internet off / run-on-push false packageを生成し、loose/package SHA一致を確認。
- 2026-07-19、ユーザーからcorrected `nested_selector_mean4_only` 40 CPU boostersの再実行承認を取得。
  exp276先行は今回は挟まず、mean4差し替えselectorを直接実行する。control再学習、downstream、GPU、
  PF再生成、推論、提出は承認範囲外。
- corrected packageをcanonical kernel version 2へpushし、Kaggle status `RUNNING`を確認。local
  `run_approved`は重複push防止のためfalseへ戻した。継続監視は行わない。
- version 2 `COMPLETE`を確認。5,707.598秒、40 model、25 compact partitions、86生成物。
- raw-test-only 88特徴、formation raw/delta hit 0、train 773/773・test 3/3 availabilityを確認。
- 40 model実体約30.8MBを選択取得し、manifest SHA 40/40一致。compact parquet本体は取得せず、
  25 partition manifestのkey/row/role/model-count契約を確認。
- selector score guardは3指標ともprior比pooled + 5/5 folds改善でPASS。corrected exp264 original
  `pf_ancc`との直接比較はexpected-error MAEのみ小幅改善、within10 logloss/Brierは小幅悪化でmixed。
