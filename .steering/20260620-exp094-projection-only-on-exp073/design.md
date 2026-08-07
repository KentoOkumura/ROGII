# 設計

## アプローチ

exp073 の `lgb_mean` OOF `pred_tvt` を固定し、well ごとに次の target-free projection を適用する。

- prefix anchor: raw horizontal well の最後の finite `TVT_input` 行から `anchor_t0`, `anchor_z0`, `anchor_md` を取る。
- projection space: `U = pred_tvt + Z - (anchor_t0 + anchor_z0)`。
- x 軸: `md_since = MD - anchor_md`。
- well 内で `U ~ poly(md_since)` を robust weighted polyfit し、`projected_tvt = projected_U - Z + (anchor_t0 + anchor_z0)` に戻す。
- 最終予測は `pred_tvt + beta * (projected_tvt - pred_tvt)`。

`target_tvt` は variant scoring のみに使う。fit 自体は予測と raw geometry だけで完結するため、同一 OOF 上での target leakage は入れない。

## 実験範囲

- 対象実験: `exp094_projection_only_on_exp073`
- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- 変更する変数: projection degree、blend beta、robust C
- 固定する変数: exp073 OOF prediction、raw train/test files、LightGBM model、PF/Beam feature generation、target definition

## 評価設計

- primary: pooled OOF RMSE vs exp073 baseline
- fold guard: exp073 と同じ `GroupKFold(n_splits=5, group=well)` 相当の original fold と、well hash fold の delta
- bucket guard: `md_since` distance bucket、tail row rank bucket、tail length bucket
- range guard: correction abs mean/p95/max、prediction min/max/std
- branch decision: 全体改善し、fold/bucket guard と correction p95 threshold を満たす場合だけ inference port candidate とする

## 再現性設計

- seed policy: `no_new_rng_projection_postprocess`
- stochastic 処理の有無: なし。upstream exp073 LightGBM OOF は既存生成物として扱う。
- PF/Beam / likelihood-PF / seed bagging の有無: この実験では再生成しない。
- 並列処理と乱数の関係: 並列処理なし、global RNG なし。
- CPU/GPU runtime と deterministic flags: CPU 実行で十分。LightGBM / GPU 学習は行わない。
- train cache / test feature regeneration の SHA 記録方針: exp073 prediction gzip は raw file SHA と decompressed content SHA を分けて記録する。
- model manifest / prediction / submission SHA 記録方針: model manifest は対象外。base prediction SHA、best prediction SHA、selected inference submission SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks` 後、kernel metadata の source と bootstrap config が exp073 train/inference source を参照することを確認する。

## リスク

- リークリスク: fit に target は使わないが、same-OOF 上の後処理探索なので過適合リスクは残る。fold/bucket guard を採用条件にする。
- CV/LB 不一致リスク: projection が真の急変や near rows を平滑化して hidden で悪化する可能性がある。near-row/short-tail guard を見る。
- ランタイム/メモリリスク: exp073 OOF は 4 model 分で大きいため chunk read で selected model のみ読み込む。
- 再現性リスク: gzip raw SHA は変わり得るため decompressed content SHA を主証拠にする。
