# 要件

## 依頼

`typewell_late_range_hard_window_pct50_full_cache_replacement` backlog を `exp192_typewell_late_range_hard_window_pct50_full_cache_replacement` として実装する。

typewell の TVT range 前半を生成前に切り、`typewell_pct >= 0.50` の typewell rows だけを使って exp072-style full replay train feature cache を raw train horizontal/typewell files から作り直す。

## 制約

- Route: `pf_beam`
- 既存 exp072 full replay cache は生成 input として読まない。比較対象としてのみ扱う。
- threshold は `typewell_pct >= 0.50` の 1 本だけにする。`0.60/0.70` grid は同時実行しない。
- soft prior、posthoc clip、hard prediction replacement、LightGBM 学習、inference、submit は今回の初期実装範囲外。
- 再現性: `docs/06_reproducibility.md` に従い、PF/Beam / likelihood-PF の stable seed、gzip decompressed SHA、Kaggle kernel version を記録する。

## 受け入れ基準

- exp192 の `docs/legacy/steering/` と `experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/` が存在する。
- config に `experiment.route: pf_beam` と hard-window `min_typewell_pct: 0.50` が明記されている。
- train notebook 起点 `.py` が Jupytext percent 形式で、raw input check、hard-window contract、feature cache generation、generated artifacts をセルで追える。
- Kaggle train push 前に CPU-only、feature cache variant 1、LightGBM config 0、fold 0、booster 0、control 再学習なしを `SESSION_NOTES.md` に記録している。
- Kaggle 実行後は rows/wells/feature_count、runtime、raw gzip SHA、decompressed content SHA、schema SHA、exp072 direct PF/Beam RMSE/MAE/within10 比較を記録する。
- direct RMSE で `likpf_mean` が exp072 から大きく悪化する場合は downstream 学習へ進めない。
