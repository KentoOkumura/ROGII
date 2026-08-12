# 設計

## アプローチ

`exp094_projection_only_on_exp073` の projection-only 実装を親にし、projection fit は変えずに correction の適用係数だけを row-wise にする。

`effective_beta = beta * clip((md_since - fade_start) / (fade_end - fade_start), 0, 1)` とし、`pred_tvt + effective_beta * correction` を評価する。これにより prefix 直後の row は exp073 raw prediction のまま保持し、long-tail 側だけ projection を効かせる。

## 実験範囲

- 対象実験: `exp096_projection_fadein_after_prefix`
- Route: `ml_model`
- 親実験: `exp094_projection_only_on_exp073`
- 変更する変数: projection correction の適用 beta schedule
- 固定する変数: exp073 OOF prediction、projection space、robust polynomial fit、scoring rows、fold reporting、inference disabled policy

## Grid

- projection variants:
  - degree 4 / robust C 2.0
  - degree 5 / robust C 1.5
- beta: 0.50, 0.75
- fade windows:
  - start 250 / end 750
  - start 250 / end 1000

## 再現性設計

- seed policy: `no_new_rng_projection_postprocess`
- stochastic 処理の有無: exp096 自体には RNG、学習、sampling はない。upstream exp073 GPU LightGBM OOF prediction のみ stochastic anchor として扱う。
- PF/Beam / likelihood-PF / seed bagging の有無: なし
- 並列処理と乱数の関係: projection audit に RNG はなく、well ごとの deterministic polynomial fit のみ。
- CPU/GPU runtime と deterministic flags: CPU で十分。GPU 学習なし。
- train cache / test feature regeneration の SHA 記録方針: exp073 prediction gzip は raw SHA と decompressed content SHA を分け、decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: model は対象外。best prediction SHA と、inference port した場合のみ submission SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks` 後、metadata と bootstrap 内 config が exp096 の support files を含むことを確認する。

## リスク

- リークリスク: projection fit が target tail を使わないことを維持する。`target_tvt` は scoring のみに使う。
- CV/LB 不一致リスク: OOF 上の postprocess grid なので overfit しやすい。global RMSE 改善だけで採用しない。
- ランタイム/メモリリスク: exp094 と同程度。8 variants に絞るため exp094 より軽い。
- 再現性リスク: upstream exp073 OOF prediction が GPU 学習由来。exp096 は deterministic transform だが deterministic submission anchor とは呼ばない。
