# 要件

## 依頼

`exact_override_negative_control` backlog を実装する。Pilkwang replay に含まれる exact-match recovery / guarded overlap override が改善根拠として採用されないことを、既存 exp079 / exp064 の証拠と任意の guard output から機械的に確認できるようにする。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- submission は作らない。optional exact / override layer を有効化して差分が出ても diagnostic only とし、submit 候補にしない。
- 同じ物理 well が別 anonymized id で出る可能性までは否定しない。
- exp064 の exposed filename-prefix overlap probe と exp079 の public artifact replay integrity audit を親証拠として扱う。

## 受け入れ基準

- Pilkwang notebook の risk hits、主要 assignment、guard output write path が `artifacts/notebook_risk_summary.csv` と summary JSON に保存される。
- exp079 summary / submission summary / pairwise JSONL が見つかる場合、Pilkwang final と base branch の一致、source spec の disabled expected checks、risk hits が summary JSON に保存される。
- exp064 hidden code submission が assertion non-trigger だったことを summary JSON に取り込む。
- `guarded_overlap_override_summary.csv` / `exact_match_recovery_summary.csv` または before/after submission が存在する場合、発火有無と prediction diff を記録する。
- 最終 decision は `exclude_same_well_exact_or_guarded_override` を明示し、改善根拠にしない。
- deterministic anchor としては扱わない。モデル、feature、submission は生成しない。
