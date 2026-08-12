# 要件

## 依頼

`self_gr_likelihood_pf_beam_probe` を実装する。最初の variant は `all_horizontal_self_similarity_candidate_rank_audit` とし、horizontal 全体 GR self-similarity 由来 candidate を既存 PF/Beam/likelihood-PF candidate と横並びに監査する。

## 制約

- Route: `pf_beam`
- 親実験: `exp090_lateral_self_gr_match_pseudotail_probe`
- 入力 cache: `exp072_exp063_full_replay_feature_cache`
- 評価区間 true TVT は candidate 生成、candidate score、rank score に使わない。
- GR self-match だけで TVT を直接置換しない。
- 初回は PF likelihood / beam pruning へ入れず、candidate coverage / rank headroom の監査に限定する。
- typewell/PF candidate を消さず、`pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`、self-GR candidate を横並びにする。
- coverage が低い場合は scorer 改善ではなく candidate 生成失敗として閉じる。
- 再現性: `docs/06_reproducibility.md` に従い、既存 PF/Beam stochastic 生成物は exp072 deterministic cache として扱い、今回の追加処理は deterministic self-GR candidate generation に限定する。

## 受け入れ基準

- `experiments/exp091_self_gr_likelihood_pf_beam_probe/` に config、train/inference notebook、補助 `.py`、記録ファイルが揃っている。
- `config.yaml` の `experiment.route` が `pf_beam` で、candidate 変換と expected artifacts が明示されている。
- train notebook は setup、cache/schema 確認、candidate audit、metrics/artifacts 確認のセル構成になっている。
- 補助 `.py` は `true_tvt = last_known_tvt + target` を評価専用にし、self-GR candidate 生成には raw GR と finite prefix `TVT_input` だけを使う。
- 出力に candidate metrics、oracle/rank topK metrics、bucket metrics、by-well metrics、candidate long、summary JSON が含まれる。
- 静的検証と synthetic smoke test が通る。
