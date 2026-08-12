# 設計

## アプローチ

保存済み prediction / submission CSV を読み、sample submission と id 互換性を確認したうえで、後続のアンサンブル候補である candidate と anchor の距離を target-free に比較する。TabICL / artifact-stack 系は public notebook や外部 dataset に依存するため、まず source root の存在、CSV inventory、SHA、予測範囲を記録する。候補が揃う場合だけ pairwise / by-well distance を出し、OOF 真値が揃わない場合は error correlation を skipped として明示する。

## 実験範囲

- 対象実験: `exp121_tabicl_artifact_diversity_audit`
- Route: `ensemble`。単体提出ではなく、後続アンサンブル候補の監査として扱う。
- 親実験: `tabicl_artifact_diversity_audit` backlog、比較 anchor は `exp027` / `exp063` / `exp073` / `exp082`
- 変更する変数: 監査対象の TabICL / artifact-stack source root と candidate CSV
- 固定する変数: モデル学習なし、TabICL 再推論なし、GPU なし、提出候補生成なし、sample submission contract

## 再現性設計

- seed policy: `no_rng_used`
- stochastic 処理の有無: なし
- PF/Beam / likelihood-PF / seed bagging の有無: なし。既存 PF/Beam/TabICL stack の保存済み CSV を読むだけ。
- 並列処理と乱数の関係: 並列 RNG なし
- CPU/GPU runtime と deterministic flags: CPU-only。`runtime.kaggle.enable_gpu=false`。TabICL 再推論は実施しない。
- train cache / test feature regeneration の SHA 記録方針: feature regeneration はない。入力 CSV と gzip の raw/decompressed SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: model manifest は対象外。candidate / anchor submission-like CSV の SHA、行数、予測範囲を inventory に記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` で metadata と support bundle を確認する。

## リスク

- リークリスク: test submission 同士の target-free 距離だけを見る。真値や hidden score を使って candidate を選ばない。OOF error correlation は fold-safe OOF がある場合に限定する。
- CV/LB 不一致リスク: CV は出さない。Public LB や replay 採用判断とは分ける。
- ランタイム/メモリリスク: 14,151 row 規模の CSV 集計なので CPU で十分。source root の再帰探索は `max_files_per_source` と `max_candidates_per_source` で制限する。
- 再現性リスク: external artifact の version / mount path 依存がある。存在しない source は missing として記録し、採用判断に使わない。
