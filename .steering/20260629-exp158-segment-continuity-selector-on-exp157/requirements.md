# 要件

## 依頼

exp157 の `candidate_ranker_feature_enrichment` は train-side OOF で改善したが、row-wise selector の path switch が大きいため、そのまま inference / submit に進めない。exp157 の score surface を使って well-local continuity constraint を評価する。

## 制約

- Route: `pf_beam`
- 親実験: `exp157_candidate_ranker_feature_enrichment`
- 新規 LightGBM booster は学習しない。exp157 の保存済み 15 booster を fold-held-out OOF score 復元に使う。
- Kaggle runtime は CPU、GPU なし、internet なし。
- 候補集合は exp157 と同じ 8 候補: `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`、`tvt_dense`、`tvt_densew`、`tvt_dense50`。
- dense 候補と enrichment feature は exp072 train cache から target-free に復元し、exp157 feature schema と一致させる。
- Viterbi / continuity selector は true TVT、oracle label、true error を gate や local cost に使わない。
- 再現性は `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- exp158 の `config.yaml`、notebook、補助 `.py`、README、SESSION_NOTES、result、metrics が exp157 continuity audit として整合している。
- Kaggle train package が CPU / no internet / kernel source 3件で作成される。
- push 前に実行対象が 1 posthoc audit、Viterbi variants 180、新規 booster 0、control / parent 再学習なしであることを `SESSION_NOTES.md` に記録する。
- Kaggle train 完了後、`likpf_mean_single`、`exp157_error_ranker_rowwise`、best Viterbi、oracle の RMSE / within10 / switch rate / by-well / bucket metrics を記録する。
- deterministic anchor として扱わないが、input feature SHA、exp157 model manifest SHA、prediction SHA、Kaggle kernel version は記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
