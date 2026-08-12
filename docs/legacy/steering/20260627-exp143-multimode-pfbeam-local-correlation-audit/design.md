# 設計

## アプローチ

exp142 の PF-Z audit 実装を親にし、候補 TVT の生成自体は大きく変えない。新規差分は seed path 分布の読出しで、各 well / variant / row について以下を計算する。

- seed TVT を 5ft bin に切った `mode_count`
- mode bin 分布の `mode_entropy`
- seed TVT の `seed_spread_p90_p10`
- eval GR window と typewell GR window の local correlation
- local correlation 上位 seed の `local_corr_topk_spread`
- `local_corr_mean` / `local_corr_max` / `local_corr_topk_mean`
- local correlation 最大 seed の `best_local_corr_tvt`

候補 TVT の RMSE は文脈として保存するが、初回の主目的は `likpf_mean` を直接超える候補探しではなく、topK に複数 mode が残っているか、また collapse がどの well / bucket で起きているかの診断とする。

## 実験範囲

- 対象実験: `exp143_multimode_pfbeam_local_correlation_audit`
- Route: `pf_beam`
- 親実験: `exp142_trajectory_aware_pf_transition_prior`
- 変更する変数: mode/correlation 診断列、quality CSV、summary JSON
- 固定する変数: exp072 train pseudo-tail rows、strict PF-Z parity、既存 exp072 candidate surface、提出なし

## 再現性設計

- seed policy: `OUTPUT_PREFIX`, `well`, `variant`, `seed_index` から SHA256 stable seed を生成する。
- stochastic 処理の有無: あり。PF particle initialization、process noise、resampling。
- PF/Beam / likelihood-PF / seed bagging の有無: PF-Z multiseed あり。likelihood-PF は exp072 cache から既存候補として読む。
- 並列処理と乱数の関係: joblib thread parallel でも各 well / variant が事前生成された seed vector を使うため、thread scheduling による乱数消費順依存を避ける。
- CPU/GPU runtime と deterministic flags: CPU-only、GPU 不使用。
- train cache / test feature regeneration の SHA 記録方針: exp072 cache SHA、decompressed SHA、schema SHA、raw train horizontal/typewell SHA、出力 CSV/GZIP SHA を summary JSON に記録する。
- model manifest / prediction / submission SHA 記録方針: モデル、推論、提出なし。candidate wide と metrics 生成物の SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --strict` 後に notebook JSON、support `.py` py_compile、metadata を確認する。

## リスク

- リークリスク: true TVT を local correlation や mode selection に使うと漏洩になる。true TVT は scoring と後段の readout に限定する。
- CV/LB 不一致リスク: train-side diagnostic のため LB 改善根拠にはならない。submit 候補化には raw-test-compatible inference port と別の guard が必要。
- ランタイム/メモリリスク: seed path x rows の local correlation は重い。window は小さくし、row-level long 出力は既定で保存しない。
- 再現性リスク: Numba 内の global RNG は seed ごとに `np.random.seed(seed)` を設定する。並列 worker 内で seed vector を固定し、実行順に依存しないようにする。
