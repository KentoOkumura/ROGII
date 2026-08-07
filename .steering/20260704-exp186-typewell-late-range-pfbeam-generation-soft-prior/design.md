# 設計

## アプローチ

exp072-style の full replay train feature cache を raw competition train well/typewell から作り直す。
既存 full replay cache は generation input として読まず、比較対象 / downstream baseline としてのみ扱う。

PF/Beam 生成時には、typewell TVT 前半 range へ戻る候補に soft penalty を加える。
hard invalidation や clipping ではなく、GR likelihood と path cost の中で候補を弱く抑制する。

selected soft prior は `pct50_strong2_pct70_weak0p5` とし、config から差し替え可能にする。

## 実験範囲

- 対象実験: `exp186_typewell_late_range_pfbeam_generation_soft_prior`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache` / `exp176_typewell_late_range_pfbeam_candidate_prior`
- 変更する変数: PF_ANCC、PF_Z、Beam path cost、128-seed likelihood-PF に入れる typewell late-range soft penalty。
- 固定する変数: raw train input、exp072-style feature surface、PF seeds/particles、Beam config、stable per-well seed policy。
- 範囲外: LightGBM 学習、inference port、submission、hard invalidation、clip、既存 cache を入力にした後処理。

## 出力

- `exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_pixiux_likpf_late_soft_prior_public_replay_train_features.csv.gz`
- `exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_feature_schema.csv`
- `exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_summary.json`

## 再現性設計

- seed policy: exp072 と同じく stable SHA256 key を per-well / split / PF 種別に使う。
- stochastic 処理: PF particle propagation / resampling / likelihood-PF seed ensemble。
- Beam: fixed input and config で deterministic。
- 並列処理: joblib threads。seed は worker order に依存させない。
- CPU/GPU runtime: Kaggle CPU (`enable_gpu=false`)。
- SHA 記録: generated train feature cache、schema、summary を完了後に記録する。
- Kaggle package bootstrap 確認: `prepare-kaggle-notebooks` 後、metadata と bootstrap manifest に `feature_cache.py` / `late_soft_prior_public_replay.py` が含まれることを確認する。

## リスク

- ランタイム: full replay cache rebuild のため multi-hour になり得る。前回 v2 の 30 分 audit とは比較しない。
- 実装リスク: local environment に `numba` が無いため、JIT import は Kaggle runtime で初めて検証される。
- 評価リスク: この実験は train feature cache 生成まで。downstream CV/LB 判断には、同じ generation code で raw test feature を再生成する inference 側が必要。
- 互換性リスク: feature count は exp072 互換の 196 を期待し、variant 名変更で likelihood-PF 絶対値列が増えないように feature selection を固定する。
