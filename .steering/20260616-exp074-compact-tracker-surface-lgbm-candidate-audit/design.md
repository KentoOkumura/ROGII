# 設計

## アプローチ

exp070 の実装を元に、目的を reproducibility guard から compact tracker surface の LB candidate audit に変更する。LightGBM 学習コードは保守的に再利用し、実験名、出力 prefix、Kaggle kernel id、記録を exp074 として分離する。

## 実験範囲

- 対象実験: `exp074_compact_tracker_surface_lgbm_candidate_audit`
- Route: `ml_model`
- 親実験: `exp070_gpu_reproducibility_guard_for_exp063`
- feature parent: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 変更する変数: 実験目的、出力 prefix、Kaggle kernel、記録、candidate audit としての解釈
- 固定する変数: exp063 compact tracker train artifact、Pixiux LightGBM 3 configs、GroupKFold by well、raw-test feature regeneration policy

## 再現性設計

- seed policy: LightGBM seed は exp070/exp063 由来の config を維持し、PF/Beam test regeneration は既存 public replay runner の設定を使う。
- stochastic 処理の有無: inference で PF/Beam/likelihood-PF test feature regeneration がある。
- PF/Beam / likelihood-PF / seed bagging の有無: inference は compact PF/Beam/likelihood-PF features を raw test から再生成する。
- 並列処理と乱数の関係: exp070 と同じ public replay implementation に依存するため、正式 deterministic anchor ではなく LB candidate audit と明記する。
- CPU/GPU runtime と deterministic flags: default は GPU train 1 回、mode は `gpu_repro_guard_dp_threads8`。
- train cache / test feature regeneration の SHA 記録方針: train feature source SHA、test feature frame SHA、prediction SHA、submission SHA を `SESSION_NOTES.md` / `metrics.json` に記録する。
- model manifest / prediction / submission SHA 記録方針: train output の `compact_tracker_surface_audit_lgb_models/manifest.json` と inference summary の SHA を保存する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` 後に metadata と bootstrap 内 config を確認し、push 後は pull を存在確認の主根拠にする。

## リスク

- リークリスク: train は fixed exp063 train artifact のみを読む。inference は current raw test から再生成し、公開 replay test CSV を使わない。
- CV/LB 不一致リスク: exp070 では CV が exp063 より悪く Public LB が良い可能性が示されたため、CV 単独で anchor 昇格しない。
- ランタイム/メモリリスク: train は約 2 時間の Kaggle GPU/CPU 実績がある。追加 GPU train は 1 回まで。
- 再現性リスク: compact surface は exp073 の stable per-well full replay guard とは別物であり、deterministic anchor ではなく candidate として扱う。
