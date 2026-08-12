# 設計

## アプローチ

公開 notebook replay 候補を prediction 改良実験としてではなく、提出前の integrity audit として扱う。Kaggle Notebook 上で追加された kernel / dataset sources を `/kaggle/input` から読み、file inventory、SHA、submission-like CSV、branch output、notebook metadata、risk pattern を保存する。

監査結果は `artifacts/exp079_public_artifact_replay_integrity_audit_summary.json`、submission summary CSV、pairwise distance JSONL、Markdown report に分けて保存する。

## 実験範囲

- 対象実験: `exp079_public_artifact_replay_integrity_audit`
- Route: `pf_beam`
- 親実験: public notebook route backlog
- 変更する変数: 公開 notebook / 外部 input / branch output の監査方法
- 固定する変数: 既存 anchor prediction は再学習しない。Kaggle competition input は sample compatibility のみに使う。

## 監査項目

- required input slug の存在確認
- 外部 input / kernel output の file inventory と SHA
- gzip 生成物の decompressed content SHA
- notebook `.ipynb` がある場合の metadata / input refs / CSV writer / risk pattern
- `submission.csv` と branch CSV の sample ID 互換性
- duplicate / missing / extra ID、null prediction、予測範囲
- exp027 / exp073 / exp063 anchor が取得済みなら pairwise RMSE / MAE / max_abs diff
- branch 間の pairwise distance

## 再現性設計

- seed policy: `no_rng_used`
- stochastic 処理の有無: なし
- PF/Beam / likelihood-PF / seed bagging の有無: なし。公開 notebook の出力を監査するだけで、この実験では PF/Beam を再実行しない。
- 並列処理と乱数の関係: 並列 RNG なし。
- CPU/GPU runtime と deterministic flags: CPU only。GPU なし。
- train cache / test feature regeneration の SHA 記録方針: feature cache は生成しない。input / candidate CSV / gzip decompressed content SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: model manifest は作らない。submission-like CSV の raw SHA と gzip decompressed SHA を保存する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` で config と kernel metadata を確認する。

## リスク

- リークリスク: 公開 notebook output が static visible-test CSV に依存する可能性がある。risk pattern と file inventory で確認し、疑いが残るものは submit 候補にしない。
- CV/LB 不一致リスク: この実験は CV を出さない。公開 LB だけで採用しない。
- ランタイム/メモリリスク: file inventory と CSV 読み込み中心なので低い。ただし外部 input が巨大な場合は inventory 上限を `max_inventory_files` で制限する。
- 再現性リスク: Kaggle input source の version が変わると SHA が変わる。Kaggle kernel version と input SHA を実行ログに記録する。
