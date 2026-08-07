# exp369_stratified_registration_offset_pf セッションノート

## 目的

registration offset層を維持するlikelihood-PFの設計を確定する。

## 現在の状態

- Route: `pf_beam`
- 状態: `design_frozen_not_implemented`
- CV / LB: なし
- Notebook / PF helper: 未実装。Notebookはplaceholder。
- Kaggle package / push / run / inference / submission: 未承認。

## コマンドログ

### 2026-07-23 実行済み

```bash
make new-steering EXP=exp369_stratified_registration_offset_pf
make new-exp EXP=exp369_stratified_registration_offset_pf
```

## 変更点

- delta grid、初期count、adjacent transition、最低quotaを固定した。
- Stage 0をknown-prefix rolling-originに固定した。
- Stage 1は1 variant、500 particles、128 seeds、773 wells、98,944 seed-well runs。
- booster / trained fold / parent PF replayは0。

## 再現性メモ

- seed: `SHA256(experiment|well|family|seed_index)`。
- stochastic: initialization、delta transition、propagation、stratified resampling、jitter。
- global RNGとthread schedule依存は禁止。train/testは別生成。
- delta diagnosticsとpredictionのdecompressed content SHAを記録する。
- kernel / prediction / submission SHA: 未実行・未生成。

## 次のアクション

1. 現時点では停止する。
2. 別承認時はStage 0だけを実装する。
