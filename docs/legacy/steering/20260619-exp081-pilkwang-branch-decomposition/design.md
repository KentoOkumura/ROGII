# 設計

## アプローチ

`exp079` v4 の `exp079_public_artifact_replay_integrity_audit_submission_summary.csv` と `exp079_public_artifact_replay_integrity_audit_pairwise_distances.jsonl` を二次解析する。候補名から branch role を分類し、final との差分、anchor との差分、blend weight / modelpkg gate の局所感度、source risk を表に落とす。

実行は target-free audit であり、Kaggle Notebook output の再利用に限定する。候補 CSV 本体がローカルにない場合は、summary と pairwise distance から分かる情報だけを採用し、row-level segment / id-level diff は未実施として記録する。

## 実験範囲

- 対象実験: `exp081_pilkwang_branch_decomposition`
- Route: `pf_beam`
- 親実験: `exp079_public_artifact_replay_integrity_audit`
- 変更する変数: 提出候補として読む Pilkwang branch の分類、candidate decision、後続 submit / followup の優先順位。
- 固定する変数: `exp079` v4 の入力 SHA、candidate SHA、pairwise distance、既存 anchor labels。新規予測は作らない。

## 再現性設計

- seed policy: `no_rng_used`
- stochastic 処理の有無: なし
- PF/Beam / likelihood-PF / seed bagging の有無: 新規実行なし。既存 branch output の監査のみ。
- 並列処理と乱数の関係: 並列処理なし。
- CPU/GPU runtime と deterministic flags: CPU only。GPU なし。
- train cache / test feature regeneration の SHA 記録方針: feature generation なし。入力 summary / pairwise JSONL の SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: `exp079` の candidate SHA を再掲し、branch decision に紐づける。
- Kaggle package bootstrap 確認方針: Kaggle push 前には `prepare_kaggle_notebooks --strict` で source files を固定する。初期実装はローカル二次解析として検証する。

## リスク

- リークリスク: public notebook の risk hits は exp079 から引き継いで表示する。exact-match / guarded overlap override は改善根拠にしない。
- CV/LB 不一致リスク: target-free diff audit なので hidden-safe 改善を保証しない。submit 候補は 1-2 個に絞るための診断として扱う。
- ランタイム/メモリリスク: summary CSV / JSONL の読み込みのみで小さい。
- 再現性リスク: `/tmp/kaggle-output` がない環境では入力不足になる。必要に応じて Kaggle output を再取得する。
