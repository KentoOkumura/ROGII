# タスクリスト

## 未着手

- なし。

## ブロック中

- なし。inference / submissionはblockではなく、readout-only契約とguard FAILにより無効。

## 完了

- 未使用の実験番号`exp282`を確認した。
- `docs/legacy/steering/20260719-exp282-longtail-prediction-zone-self-gr-loop-closure-readout/`を作成した。
- exp281は待機条件ではなく並行比較参照であることを確定した。
- receiver `md_since >= 1000 ft`、donor prediction-zone `0 <= md_since < 500 ft`を固定した。
- raw-GR matching、truth attachment、negative control、confidence、metric、guard、再現性設計を確定した。
- 初回は補正0・model/HMM/PF生成0・booster 0のreadoutに限定した。
- `experiments/exp282_longtail_prediction_zone_self_gr_loop_closure_readout/`をtemplateから作成した。
- `config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`へplanned contractを記録した。
- `KAGGLE_DIRECTION.md`の共通backlogから個別実験へ移管し、判断メモを追加した。
- `experiment_summary.md`へplanned experimentとして記録した。
- Jupytext percent形式のcompact self-contained train sourceを作成した。
- train sourceを別名`.ipynb`へ変換し、Contents、入力契約、matching、score freeze、truth readout、
  guard、SHA保存を10章のnotebook cellで追えるようにした。
- 別名compact inference source/notebookをfail-closed contractで作成した。
- score-stage forbidden-column guardとtruth attachment順序guardを実装した。
- rolling mean 5、half-window `[8,15,25]`、stride 3、forward/reverse NCCを実装した。
- primary edge、multiscale agreement、segment support、equal-weight confidenceを実装した。
- stable SHA256 per-well shuffled donor controlをlocal RNGで実装した。
- exp263 Stage 0 fixed formulaのpost-freeze donor-transfer readoutを実装した。
- overall / fold / scope / hidden-like / orientation / by-well metricsと固定guardを実装した。
- edge contract、schema、logical content SHA、input/output manifest保存を実装した。
- synthetic unit testsでforward、reverse、tie、segment、shuffle determinism、truth rejectionを確認した。
- Jupytext同期、`py_compile`、ruff、strict `make validate-exp`、`make validate-template`をPASSした。
- exp280/281/282 targeted testsは18 passed、repository全体は178 passed / unrelated exp264 1 failedだった。
- 2026-07-19の実行依頼を正規notebook採用とKaggle CPU v1実行承認として記録した。
- compact train/inference notebookを正規notebookへ採用した。
- Kaggle train packageをcanonical id/title、private CPU、internet off、2 kernel sourcesでprepareした。
- bootstrap 15 filesのmanifest/hash/bytes/loose file一致とconfig/source/package SHAを監査した。
- push前にvariant 1 / config 0 / fold 0 / booster 0 / HMM/PF 0 / control再生成0を
  `SESSION_NOTES.md`へ再記録し、ユーザーの実行承認を確認した。
- Kaggle private CPU version 1（id_no `127838798`）を実行し、773 wellsを248.206秒で完了した。
- logsを第一証拠としてtechnical guard全PASS、scientific guard FAILを確認した。
- fold / by-well / donor-transfer数値と成果物SHA確認が必要なためoutputを`/tmp`へ取得し、
  edge raw/decompressed SHAを含む全artifact SHAがlogと一致することを確認した。
- high-confidence within10 0.554309、lift +0.003257、positive fold 4/5、donor-transfer gain
  -6.894739 ft・改善0/5 foldsを記録した。
- `SESSION_NOTES.md`、`result.md`、`metrics.json`、`experiment_summary.md`、
  `KAGGLE_DIRECTION.md`へnegative resultとbranch close判断を反映した。
- parameter rescue、補正、inference、submissionへ進まず、新規救済backlogを追加しないことを確定した。
