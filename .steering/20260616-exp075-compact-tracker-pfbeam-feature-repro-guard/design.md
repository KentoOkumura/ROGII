# 設計

## アプローチ

exp074 の compact tracker LightGBM 実装を親にし、feature generation と model training の責務を分ける。train feature 生成は raw train から PF/Beam/likelihood-PF compact tracker surface を作る専用 notebook とし、LightGBM train notebook はその生成物を固定入力として読む。

## 実験範囲

- 対象実験: `exp075_compact_tracker_pfbeam_feature_repro_guard`
- Route: `ml_model`
- 親実験: `exp074_compact_tracker_surface_lgbm_candidate_audit`
- feature parent: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 主な変更:
  - `exp075_compact_tracker_pfbeam_feature_repro_guard_pfbeam_features.ipynb` を追加。
  - `compact_tracker_pfbeam_repro_guard.py` に train feature generation runner を追加。
  - LightGBM train は generated tracker feature CSV を読む。
  - feature importance を fold/model 単位で CSV 化し、平均重要度トップを matplotlib PNG で保存する。
  - inference は raw test feature regeneration と saved booster inference を維持する。

## 再現性設計

- seed policy: PF/Beam 生成は exp063/exp074 由来の public replay implementation と config を使う。
- stochastic 処理の有無: train feature generation と inference test feature generation に PF/Beam/likelihood-PF がある。
- PF/Beam / likelihood-PF / seed bagging の有無: compact tracker PF/Beam/likelihood-PF features を生成する。
- train feature の再現性確認: 2 回再生成はしない。生成物の row/well/feature count、schema、raw file SHA、decompressed CSV content SHA を記録する。
- LightGBM GPU: exp074 と同じ `gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、固定 `num_threads=8` mode を主モードにする。
- model / prediction 証拠: model manifest、OOF prediction SHA、feature importance CSV/PNG、test prediction SHA、submission SHA を記録する。

## リスク

- train feature generation notebook は標準の `prepare_kaggle_notebooks.py --notebook train/inference` の対象外なので、Kaggle push 時は別 package 化の手順が必要。
- compact surface は 65 feature surface であり、exp073 の full replay deterministic anchor とは別物として扱う。
- Public LB が良くても、feature SHA / submission SHA の証拠が揃うまでは deterministic submission anchor に昇格しない。
