# 設計

## アプローチ

新規実験 `exp122_exact_override_negative_control` として、ファイル監査スクリプト `exact_override_negative_control.py` を実装する。Kaggle train notebook から実行し、以下を 1 つの summary JSON にまとめる。

- Pilkwang notebook source の risk pattern count と主要 flag assignment。
- exp079 public artifact replay integrity audit の source spec、notebook inspection、submission summary、pairwise summary。
- exp064 train/test well_id assert probe の hidden code submission 結果。
- optional exact/guard output summary が存在する場合の発火件数、before/after submission diff。

decision はスコア改善ではなく、same-well exact / guarded override を hidden-safe 改善根拠から除外できるかで判定する。

## 実験範囲

- 対象実験: `exp122_exact_override_negative_control`
- Route: `pf_beam`
- 親実験: `exp079_public_artifact_replay_integrity_audit`、`exp064_train_test_well_id_assert_probe`
- 変更する変数: exact-match / guarded overlap override の採用可否を評価する監査 logic。
- 固定する変数: Pilkwang replay の予測値、exp079/exp064 の既存証拠、submission contract。モデル学習、PF/Beam 再生成、提出は行わない。

## 再現性設計

- seed policy: `none_deterministic_file_audit`
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。既存公開 replay の出力とメモだけを読む。
- 並列処理と乱数の関係: なし。
- CPU/GPU runtime と deterministic flags: CPU file audit。GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: feature cache は生成しない。入力 notebook、exp079 summary、exp064 metrics、guard output CSV の SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: model / submission は生成しない。before/after submission が見つかる場合だけ SHA と diff を記録する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --notebook train --run-on-push --strict` で metadata / bootstrap を作る。実行時に存在する `/kaggle/input` と `/tmp/kaggle-output` 候補を notebook 上で表示する。

## リスク

- リークリスク: optional exact/override 自体が same-well shortcut risk なので、発火しても採用しない。
- CV/LB 不一致リスク: CV/LB を評価しない診断実験。スコア改善とは分けて記録する。
- ランタイム/メモリリスク: notebook JSON と小さな CSV/JSONL 読み込み中心で低い。submission before/after が存在する場合のみ 14k rows 程度を比較する。
- 再現性リスク: Kaggle output の mount 状態に依存して evidence completeness が変わる。見つからない証拠は欠落として summary に記録する。
