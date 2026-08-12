# 要件

## 依頼

`pf_candidate_generation_likelihood_upgrade` の初手として、既存 PF/Beam/likPF 候補を直接 selector で選ぶのではなく、候補ごとの observation likelihood を学習・校正する train-side smoke audit を実装する。

## 制約

- Route: `pf_beam`
- 親実験: `exp099_pf_multi_observation_likelihood_probe`
- 比較文脈: `exp093_pf_candidate_coverage_then_ranker_audit`、`exp101_pf_candidate_ranker_or_nway_classifier`
- 入力は exp099 v2 の train-side pseudo-tail wide cache に固定する。
- 候補は `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb` に固定する。
- 直接 `multiobs_top1` や learned top1 を submission candidate として採用しない。
- 学習 label は train pseudo-tail の true TVT から作るが、GroupKFold by well の validation に限定する。
- 再現性: `docs/06_reproducibility.md` に従い、upstream cache SHA、model manifest SHA、OOF likelihood gzip の decompressed SHA を記録する。

## 受け入れ基準

- `experiments/exp111_learned_pf_observation_likelihood_probe/` に config、補助コード、train/inference notebook、記録ファイルがある。
- train notebook から `learned_pf_observation_likelihood_probe.py` を実行し、candidate-long likelihood model を 1 fold smoke として学習・評価できる。
- 出力には metrics、top-K coverage、calibration table、bucket metrics、OOF likelihood long cache、feature importance、model manifest、summary JSON が含まれる。
- 判定は candidate likelihood AUC / brier / calibration / topK coverage を主にし、diagnostic top1 RMSE は補助扱いにする。
- deterministic anchor として扱わない理由が config と notes に明記されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
