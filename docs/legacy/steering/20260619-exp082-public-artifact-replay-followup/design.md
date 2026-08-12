# 設計

## アプローチ

`exp079_public_artifact_replay_integrity_audit` をコピーし、audit 対象を Pilkwang primary から SP45 / fle3n / Koolbox 系へ移す。source slug は保存済み公開 notebook metadata から固定する。

対象:

- `fleongg/fle3n-rogii-v4`
- `rauffauzanrambe/rogii-sp45-wellbore-for-blend-prediction`
- `jaemin3404/rogii-sp45-fleongg-blend-v2`
- `debatreyabiswas/wellboregeology-prediction-with-koolbox-best-8-188`
- dependency source: `packagemanager/pm-121774751-at-06-05-2026-09-29-28`
- datasets: `phongnguyn23021656/koolbox-offline`, `fleongg/rogii-claude-models-pub`, `ravaghi/wellbore-geology-prediction-artifacts`

## 実験範囲

- 対象実験: `exp082_public_artifact_replay_followup`
- Route: `pf_beam`
- 親実験: `exp079_public_artifact_replay_integrity_audit`
- 変更する変数: public artifact audit の source specs、Kaggle mounted sources、source-code inspection 対象
- 固定する変数: no supervised training、no submit、sample compatibility / SHA / pairwise distance audit 方針

## 再現性設計

- seed policy: `no_rng_used`
- stochastic 処理の有無: なし
- PF/Beam / likelihood-PF / seed bagging の有無: audit 対象 notebook の内部には存在するが、この実験自体は再計算せず生成物を監査する。
- 並列処理と乱数の関係: audit はファイル列挙と SHA 計算のみ。
- CPU/GPU runtime と deterministic flags: CPU only。GPU は不要。
- train cache / test feature regeneration の SHA 記録方針: 対象外。
- model manifest / prediction / submission SHA 記録方針: input inventory と candidate CSV SHA を `artifacts/exp082_public_artifact_replay_followup_*` に保存する。
- Kaggle package bootstrap 確認方針: prepared notebook の support bundle SHA と `kernel-metadata.json` の source list を確認する。

## リスク

- リークリスク: 公開 notebook が static visible CSV、public sample branch、hardcoded `/kaggle/input/*submission.csv` を使う可能性があるため source risk pattern と output inventory で確認する。
- CV/LB 不一致リスク: target-free audit なので CV は出さない。公開 LB title だけでは採用しない。
- ランタイム/メモリリスク: SHA / CSV audit のみで軽いが、mounted source が多く inventory が増えすぎる可能性があるため `max_inventory_files` と `max_source_files` を制限する。
- 再現性リスク: Kaggle の kernel source mount path が `/kaggle/input/notebooks/...` / `/kaggle/input/datasets/...` になりうるため、leaf slug fallback で探索する。
