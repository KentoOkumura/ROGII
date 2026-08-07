# 設計

## アプローチ

`exp073` / `exp092` の OOF prediction と `exp072` の PF/Beam/likPF train pseudo-tail feature cache を `id, well` で join し、同じ行集合で候補ごとの誤差を比較する。

主な候補:

- `last_anchor_tvt`
- `pf_ancc`
- `beam_mean`
- `likpf_mean`
- `sc_ens`
- `hyb`
- `exp073_lgb_mean`
- `exp092_lgb1`
- `exp092_lgb_mean`

行ごとに oracle best candidate を計算し、bucket / well / path continuity proxy 別に、`exp092_lgb1` がどの条件で `exp073_lgb_mean` より改善または悪化するか、PF/Beam/likPF 候補の headroom がどこにあるかを読む。

## 実験範囲

- 対象実験: `exp126_exp073_exp092_pf_beam_pseudotail_failure_map`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- 参照実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`、`exp072_exp063_full_replay_feature_cache`、`exp093_pf_candidate_coverage_then_ranker_audit`、`exp101_pf_candidate_ranker_or_nway_classifier`
- 変更する変数: なし。診断集計のみ。
- 固定する変数: exp073/exp092 OOF prediction、exp072 PF/Beam/likPF feature cache、raw train prefix context。

## 生成物

- `exp126_exp073_exp092_pf_beam_pseudotail_failure_map_row_failure_map.csv.gz`
- `exp126_exp073_exp092_pf_beam_pseudotail_failure_map_candidate_metrics.csv`
- `exp126_exp073_exp092_pf_beam_pseudotail_failure_map_bucket_metrics.csv`
- `exp126_exp073_exp092_pf_beam_pseudotail_failure_map_well_metrics.csv`
- `exp126_exp073_exp092_pf_beam_pseudotail_failure_map_summary.json`
- `README.md`

## 再現性設計

- seed policy: no new RNG diagnostic。
- stochastic 処理の有無: exp126 内ではなし。
- PF/Beam / likelihood-PF / seed bagging の有無: exp126 では再生成しない。upstream exp072 cache を読む。
- 並列処理と乱数の関係: exp126 内では joblib/thread RNG なし。
- CPU/GPU runtime: CPU train-side audit。GPU 不要。
- train cache SHA: gzip は decompressed content SHA を主証拠として summary JSON に記録する。
- model manifest / prediction / submission SHA: 新規 model / prediction / submission なし。入力 prediction/cache SHA だけ記録する。
- Kaggle package bootstrap: `prepare-kaggle-notebooks --notebook train --strict` 後、kernel sources に exp072/exp073/exp092 train が入ることを確認する。

## リスク

- リークリスク: target は誤差・oracle 診断だけに使う。後続 gate 条件はこの実験から直接 inference へ移植しない。
- CV/LB 不一致リスク: train-side pseudo-tail 診断であり、LB を主張しない。
- ランタイム/メモリリスク: exp072 feature cache と OOF prediction は大きい。読み込み列を必要列に限定する。
- 再現性リスク: upstream cache/prediction は既存実験由来。exp126 は deterministic anchor ではない。
