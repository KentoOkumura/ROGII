# 設計

## アプローチ

exp073 を base、exp075 compact surface を optional branch として id align する。各 row の予測は次で作る。

```text
pred = exp073 + w * (exp075_compact - exp073)
```

`w` は 0 / 0.05 / 0.10 / 0.20 に限定し、long-tail 条件を満たす row だけ非ゼロにする。

## 実験範囲

- 対象実験: `exp078_compact_surface_longtail_gate`
- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- branch: `exp075_compact_tracker_pfbeam_feature_repro_guard`
- 変更する変数: compact surface branch の gate 条件と blend weight
- 固定する変数: exp073/exp075 の学習済み予測、fold、target、評価対象 row

## Gate 候補

- `tail_rank_ge1000`: evaluation tail 内の 1000 row 以降。
- `tail_or_len_long`: tail rank 1000 以降、または evaluation length 5000+ の well。
- `tail_rank_ge1000_diff_p50`: long-tail かつ exp075 と exp073 の差が p50 以上。
- `tail_rank_ge1000_diff_p75`: long-tail かつ exp075 と exp073 の差が p75 以上。

feature cache が有効なら `md_since` ベースの gate も追加できる。feature cache がない、または壊れている場合は id-derived tail rank / tail length と prediction diff だけで動作する。

## Metric Discussion の反映

- RMSE は `sqrt(mean(squared_error))` なので、policy 比較では RMSE だけでなく `delta_sse_vs_base` を保存する。
- long-tail bucket の SSE delta を別に保存し、全体改善が短い区間の偶然でないかを見る。
- global OOF RMSE が改善しても、well-level の最大 RMSE 悪化が大きい policy は submit candidate にしない。
- `passes_discussion_metric_guard` は、overall SSE 非悪化、long-tail SSE 改善、最大 well RMSE 悪化上限を満たす場合だけ true にする。

## 再現性設計

- seed policy: 新しい stochastic 処理は追加しない。
- stochastic 処理の有無: なし。保存済み exp073/exp075 predictions を deterministic input として扱う。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成はしない。exp075 側の生成物 SHA と kernel source を証拠にする。
- 並列処理と乱数の関係: なし。
- CPU/GPU runtime と deterministic flags: train/inference とも saved predictions の align/blend だけなので GPU 非依存。
- train cache / test feature regeneration の SHA 記録方針: prediction input の raw SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: best prediction SHA、inference prediction SHA、submission SHA を summary と SESSION_NOTES に記録する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --strict` で kernel sources と notebook name を確認する。

## リスク

- リークリスク: OOF の同一データ上 policy selection なので、OOF 改善は診断扱い。inference port と LB で確認する。
- CV/LB 不一致リスク: visible long-tail に合わせすぎると hidden short/mid well を壊す。well-level regression guard を必須にする。
- ランタイム/メモリリスク: predictions の align と集計だけなので低いが、row-level all-policy prediction は保存せず best policy のみ保存する。
- 再現性リスク: exp075 の stable-seed v2/v3 以降の prediction を使う。入力 prediction SHA を必ず記録する。
