# 設計

## アプローチ

exp173 の Beam top-K retained paths は、global posterior / oracle ともに `likpf_mean` より大きく悪化した。一方、バックログの論点は「二峰性が強い行だけ posterior を使うとどうか」なので、exp173 の保存済み diagnostics を再利用し、Beam search は再実行しない。

実装は `beam_topk_bimodal_gate_posthoc_audit.py` に閉じる。`candidate_wide` から `likpf_mean`、`beam_mean`、`top2_commit`、`topk_weighted_mean`、`posterior_mean_t1/t2/t4/t8/t16` を読む。`topk_diagnostics` から `top1_top2_sep`、`top2_cost_gap_per_row`、`topk_entropy`、`topk_spread` を読み、各 Beam variant ごとに configured quantile threshold を作る。Gate 成立 row だけ replacement candidate に置換し、それ以外は `likpf_mean` を維持する。

## 実験範囲

- 対象実験: `exp177_beam_topk_bimodal_gate_posthoc_audit`
- Route: `pf_beam`
- 親実験: `exp173_beam_topk_path_posterior_audit`
- 変更する変数: gate 条件、replacement candidate 種類
- 固定する変数: exp173 top-K path/cost/posterior outputs、baseline `likpf_mean`、score rows、Beam variants

## Gate

- 単独 gate:
  - `top1_top2_sep >= q75/q90`
  - `top2_cost_gap_per_row <= q10/q25`
  - `topk_entropy >= q75/q90`
  - `topk_spread >= q75/q90`
- AND gate:
  - separation と low cost gap
  - separation と entropy
  - spread と low cost gap
  - separation、spread、entropy、low cost gap の joint conservative gate

Replacement は `top2_commit`、`topk_weighted_mean`、`posterior_mean_t1/t2/t4/t8/t16` に限定する。`top1_commit` は exp173 の Beam commit として既に弱く、今回の backlog 目的から外す。

## 評価

- global RMSE / MAE / within10
- changed rows / changed wells / changed subset RMSE
- near `000_050`
- longtail `1000_plus`
- `beam_mean` vs `likpf_mean` gap top quartile
- mode-separation bucket
- by-well max regression vs `likpf_mean`

Positive 判定は global RMSE 改善、changed subset 改善、max well regression `<= 0.25 RMSE` を同時に満たす場合だけ。満たしても inference port / submit には進めず、raw-test parity と feature 化判断を別途確認する。

## 再現性設計

- seed policy: `deterministic_posthoc_grid_no_rng`
- stochastic 処理の有無: exp177 内ではなし
- PF/Beam / likelihood-PF / seed bagging の有無: exp177 では再生成なし。上流 exp072 / exp173 の固定生成物のみ参照
- 並列処理と乱数の関係: 乱数なし、thread scheduling による乱数消費なし
- CPU/GPU runtime と deterministic flags: CPU only、GPU 不使用
- train cache / test feature regeneration の SHA 記録方針: exp173 input gzip は raw SHA と decompressed SHA を summary に記録
- model manifest / prediction / submission SHA 記録方針: model / prediction / submission なし。policy metrics、gate thresholds、group metrics、by-well、summary の SHA を記録
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後、metadata の kernel source と GPU off を確認

## リスク

- リークリスク: threshold を true TVT error で選ぶと漏れるため、configured quantile のみ使う。
- CV/LB 不一致リスク: train-side diagnostic only であり、positive でも direct inference しない。
- ランタイム/メモリリスク: `candidate_wide` と `topk_diagnostics` の 3.78M rows を読む。Beam 再生成や LightGBM はないため CPU で許容範囲の想定。
- 再現性リスク: 上流 exp173 output が Kaggle source として mount されない場合は実行不能。config の path candidates と rglob fallback で探索する。
