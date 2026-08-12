# 設計

## アプローチ

監査を二つの問いに分ける。

1. **bank-selectability**: 各row/block/wellでbest K8 medoidがbest base8を上回るかを
   `union_best_source_is_k8` labelとする。well/PF全体のtarget-free診断がこのlabelを
   rankingできるかをAUCと固定top-decile coverageで読む。
2. **candidate-selectability**: 各K8 medoidについて、base8+K8 unionのbest candidateに
   なったかをlabelとする。cluster単位のtarget-free scoreで8 medoidをrankし、AUC、
   top1 oracle match、useful-medoid coverage、best-K8へのregretを読む。

row lossはabsolute error、block/well lossはRMSEとする。blockはwell内row orderを
128/256/512行で先頭から非重複分割し、末尾short blockも保持する。tie toleranceは
`1e-6 ft`、候補/score tieは固定slot順で解く。

## 実験範囲

- 対象実験: `exp252_pf_seed_medoid_selectability_audit`
- Route: `pf_beam`
- 親実験: `exp243_pf_seed_medoids`
- 変更する変数: target-free scoreのreadoutだけ。
- 固定する変数: exp243 v3 row candidates、base8、K8、cluster assignment/manifest、
  PF diagnostics、K=8、scope、loss、label、shuffle、coverage fraction。

## Score contract

### Bank scores

- `cluster_normalized_entropy`: 高いほどmode massが分散。
- `negative_cluster_hhi`: 高いほどmode massが分散。
- `effective_cluster_count`: 高いほど有効mode数が多い。
- `mean_assignment_distance`: 高いほどseed trajectoryが単一modeから離れる。
- `max_pairwise_seed_distance`: 高いほどseed bankのtrajectory spreadが大きい。
- `negative_ess_mean`: 高いほどPF degeneracyが強い。
- `resampling_rate`: 高いほどPF resamplingが多い。
- `log_likelihood_std`: 高いほどseed likelihoodの分散が大きい。
- `seed_prediction_std`: scope内のrow-wise seed std平均。
- `k8_nearest_base_disagreement`: bestではなく、K8 bankとbase8 bankの最小path距離。

### Candidate scores

- `cluster_seed_mass`
- `cluster_likelihood_mass`
- `medoid_likelihood_rank_score`（K8内で高likelihoodほど高い）
- `medoid_likelihood_gap_from_best`（bestとの差、0が最大）
- `negative_mean_within_distance`
- `separation_to_within_ratio`
- `nearest_base8_disagreement`（scope内RMS distance）

scoreの符号、K、scope、coverage fractionはKaggle実行結果を見て変更しない。各scoreは
単独で監査し、学習・重み付き合成・best score選択をしない。

## Labelとreadout

- bank primary label: `best_k8_loss + tol < best_base8_loss`。
- candidate primary label: 当該medoidがunion minimum lossと`tol`内で一致し、かつ
  best K8がbest base8をstrictに上回る。
- secondary label: 当該medoid単体がbest base8をstrictに上回る。
- AUCはprimary/secondary labelごとに保存する。
- coverageは固定top 10% scoreのprecision、positive recall、prevalence比と、candidate top1が
  useful K8 medoidを回収したunit比率を保存する。
- top1 regretはscore top1 medoid lossからbest K8 lossを引く。rowではabsolute-error差、
  block/wellではRMSE差で、mean/p90/maxを記録する。
- negative controlは`SHA256(experiment, scope, score, seed=42)`から作るlocal RNGでscope unitを
  permutationし、labelは固定する。

## 再現性設計

- seed policy: score/scope名を含むSHA256由来seedでlocal `np.random.default_rng`を使う。
- stochastic 処理: shuffled-score controlだけ。実score計算は決定的。
- PF/Beam / likelihood-PF / seed bagging: 再実行しない。exp243の保存済み生成物だけを読む。
- 並列処理: single process。global RNG、Python `hash()`、thread scheduling依存なし。
- runtime: Kaggle CPU、GPU/internet disabled。
- input SHA: exp243 gzipはdecompressed content SHA、通常CSVはfile SHAをhard guardする。
- output SHA: summary/CSV/metricsのSHAを記録する。model/prediction/submissionは生成しない。
- Kaggle bootstrap: source / loose package / bootstrap内configをbyte比較する。
- deterministic anchor: prediction/submission anchorではなく、固定exp243生成物に対する
  deterministic diagnosticだけを主張する。

## リスク

- リークリスク: targetからscoreを作らず、score tableを先に構築してからtarget loss/labelを
  joinする。best scoreや符号を結果で選ばない。
- multiplicity: 複数scoreは独立仮説readoutであり、最良AUCを採用値として扱わない。
- CV/LB不一致: train-side pseudo-tail診断のみ。raw-test inferenceとsubmitは禁止。
- ランタイム/メモリ: 3.78M row × 16候補をfloat32 matrixで読み、scoreごとに逐次評価して
  30M-rowの長形式常設を避ける。
- 再現性: exp243 canonical SHAが違えばscore前に停止する。
