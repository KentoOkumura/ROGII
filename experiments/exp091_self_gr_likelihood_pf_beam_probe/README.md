# exp091_self_gr_likelihood_pf_beam_probe

## Status

Kaggle train v1 completed.

## Hypothesis

PF/Beam/likelihood-PF は全体の直接置換では弱いが、well や bucket によっては真値近傍候補を含んでいる。`exp091` では、同一 horizontal well 全体の GR self-similarity から作る候補を、既存 `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb` と横並びにし、候補集合が真値近傍を含むかを先に測る。

## Validation Strategy

exp072 deterministic full replay train cache を固定し、raw train horizontal GR と finite prefix `TVT_input` だけで self-GR 候補を作る。評価用 true TVT は `last_known_tvt + target` として materialize するが、候補生成、score、ranking には使わない。

## Scope

- Route: `pf_beam`
- Parent: `exp090_lateral_self_gr_match_pseudotail_probe`
- Cache parent: `exp072_exp063_full_replay_feature_cache`
- First variant: `all_horizontal_self_similarity_candidate_rank_audit`
- No LightGBM training
- No PF sampling / beam pruning change
- No submission / inference

## Candidates

- `last_anchor_tvt`
- `pf_ancc`
- `beam_mean`
- `likpf_mean`
- `sc_ens`
- `hyb`
- `self_gr_ens`
- `self_gr_best`
- `self_gr_sc8`
- `self_gr_sc15`
- `self_gr_sc25`

## Expected Outputs

- `exp091_self_gr_likelihood_pf_beam_probe_candidate_metrics.csv`
- `exp091_self_gr_likelihood_pf_beam_probe_rank_metrics.csv`
- `exp091_self_gr_likelihood_pf_beam_probe_bucket_metrics.csv`
- `exp091_self_gr_likelihood_pf_beam_probe_by_well.csv`
- `exp091_self_gr_likelihood_pf_beam_probe_self_gr_well_summary.csv`
- `exp091_self_gr_likelihood_pf_beam_probe_candidate_long.csv.gz`
- `exp091_self_gr_likelihood_pf_beam_probe_feature_schema.csv`
- `exp091_self_gr_likelihood_pf_beam_probe_summary.json`

## Findings

- `likpf_mean` is the best single candidate: RMSE 11.594897, within 10ft 0.772807.
- Self-GR standalone candidates are weak: `self_gr_ens` RMSE 191.215912 and `self_gr_best` RMSE 250.161697.
- Oracle best candidate has strong headroom: RMSE 6.873199, within 10ft 0.925153, selected self-GR rate 0.135212.
- Current target-free `candidate_rank_score` is not enough: top1 RMSE 29.985529, worse than `likpf_mean`.
- Do not use self-GR as a direct replacement or hard switch. Keep it only as possible ranker feature material.

## Decision Rule

If topK/oracle coverage is low in the target buckets, close this as candidate-generation failure. If coverage exists but `candidate_rank_score` cannot recover it, the next experiment should be a supervised candidate ranker. If self-GR candidates are useful only in narrow buckets, move to a scale selector or soft switch instead of a hard replacement.
