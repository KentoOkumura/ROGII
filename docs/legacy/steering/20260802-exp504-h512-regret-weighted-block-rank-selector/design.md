# 設計

## 1. 変更する枠組み

行ごとに候補のabsolute errorやwithin-10確率を推定するexp264系とは異なり、
H512 blockを1 query、12候補をdocumentsとみなすpairwise rank学習にする。
学習対象はTVT値そのものではなく、同じblock内で「候補aとbのどちらのblock MSEが
小さいか」である。誤差差が大きいpairを強く学習し、blockごとに1候補を選ぶ。

LightGBMのbinary classifierをordered-pair差分へ適用するRankNet型のpairwise logistic
rankingを採用する。これはpointwise回帰の後にsortする方式ではない。LambdaRankとの
loss比較は行わず、continuous regretを直接weightへ入れられるこの1方式だけを評価する。

## 2. 入力契約

### Candidate bank

exp293のtarget-free fixed12 bankを順序まで固定する。

1. `exp226_k16`
2. `selfgr_hmm_a070`
3. `likpf_mean`
4. `exact_hmm`
5. `pf_ancc`
6. `beam_mean`
7. `exp226_k16__selfgr_hmm_a070`
8. `exp226_k16__exact_hmm`
9. `exp226_k16__likpf_mean`
10. `selfgr_hmm_a070__likpf_mean`
11. `likpf_mean__exact_hmm`
12. `exp226_w500_50_50`

固定証拠:

- rows / wells / candidates: `3,783,989 / 773 / 12`
- candidate content SHA256:
  `29477141685662bae7417e788ec5dbe914c2220b7f4cc45ab01befa4e5e3b474`
- candidate key content SHA256:
  `42ede2f53e28dc2ccb28f847e0bc23680d1121bb29e980016dee989dbdddfef7`
- truth content SHA256:
  `e9067327058431278a0fd994e8e6005b76ab99acbd3942118974599afb69a8d0`

### Block/query

exp293の保存block assignmentから`horizon_rows == 512`だけを使う。各wellのsuffix先頭から
non-overlapで割り当て、well末尾の短いblockを含む。blockをまたぐ特徴集約や候補選択は
行わない。

- H512 queries: `7,787`
- candidate-block objects: `7,787 × 12 = 93,444`
- unordered pairs before tie removal: `7,787 × C(12,2) = 513,942`
- block assignment decompressed SHA256:
  `b0755c22aa8d791012d3f605e2f1b66063ce9bb6ba46ddd4b48dca77cce032d7`
- block assignment logical SHA256:
  `63f9a26a243ce3b1dd0cbec85c9674fd69a0768246220728ee9d54defba046e5`

### Row feature schema

corrected exp264の88列だけを使用する。schema logical SHA256は
`aaef4ffdd90667893b099b76a52f1957b22197aea9cee5e5b57bc81048ddd3a4`。
`candidate_index`は使わずcandidate ID one-hotを維持する。

各candidate-blockについて88列それぞれを、row orderを固定して次の9演算で集約する。

1. finite fraction
2. mean
3. population std (`ddof=0`)
4. q10 (`numpy linear`)
5. median / q50 (`numpy linear`)
6. q90 (`numpy linear`)
7. first finite
8. last finite
9. last finite - first finite

finite値が0件ならfinite fraction以外をNaNのまま残し、LightGBM native missingへ渡す。
`ctx__`22列はcandidate-independent block contextとして先頭candidateから同じ9演算で
別途集約し、12候補でrow-wise同値であることをtechnical gateにする。追加block contextは
`row_count`、`is_partial_block`、`block_start_md_since`、`block_end_md_since`、
`block_start_evaluation_progress`、`block_end_evaluation_progress`の6列とする。

pair入力は、candidate-block vectorを`g_a`, `g_b`として
`[g_a-g_b, abs(g_a-g_b), (g_a+g_b)/2, shared_ctx, block_context]`の固定順で作る。
feature名、dtype、順序、schema SHA、logical content SHAをtruth読込前に凍結する。

## 3. Pair labelとregret weight

outer-train blockだけで候補jの
`MSE_bj = sum_row((pred_j - truth)^2) / row_count_b`を計算する。
canonical unordered pairは上記candidate順の`a < b`とする。

- label: `y_ab = 1[MSE_ba < MSE_bb]`
- tie: `abs(MSE_ba - MSE_bb) <= 1e-12`ならpairを除外
- raw weight: `row_count_b * log1p(abs(MSE_ba - MSE_bb))`
- fold normalization: outer-trainで残ったunordered pairのraw weight平均が1になるよう除算

各unordered pairから`a,b,y,w/2`と`b,a,1-y,w/2`の2 ordered examplesを作る。
これにより向きによるcandidate順の偏りを抑え、unordered pairの総weightを保存する。

## 4. Rank model

各outer foldに1つ、残り4 foldsのordered pairsでCPU LightGBMを学習する。

- estimator: `LGBMClassifier`
- objective / metric: `binary / binary_logloss`
- boosting: `gbdt`
- learning_rate: `0.03`
- n_estimators: `800`固定、early stoppingなし
- num_leaves: `31`
- min_child_samples: `100`
- max_depth: `-1`
- subsample / subsample_freq: `1.0 / 0`
- colsample_bytree: `0.8`
- reg_alpha / reg_lambda: `0.1 / 1.0`
- max_bin: `255`
- random_state: `42`
- deterministic / force_col_wise: `true / true`
- device / n_jobs: `cpu / 4`

class balancing、pair subsampling、hyperparameter grid、calibration、inner CV、early stoppingは
行わない。5 outer foldsで合計5 models / boostersとする。

## 5. Block内順位とanchor guard

held blockの全66 pairについて両方向を予測する。raw確率を`q(a,b)`, `q(b,a)`として、
反対称化した勝率を

`p(a beats b) = 0.5 * (q(a,b) + 1 - q(b,a))`

とする。候補jのBorda scoreは他11候補に対する勝率平均とする。最大候補をprovisional
winnerにする。同点差が`1e-12`以内ならanchorが同点集合にあればanchor、それ以外は固定
candidate順で先の候補を選ぶ。

固定anchorは`exp226_w500_50_50`。provisional winnerがanchor以外の場合、
`p(winner beats anchor) > 0.5`のときだけ採用し、それ以外はanchorへ戻す。
確率閾値は0.5固定で調整しない。選択候補のTVTをblock全行へそのまま割り当て、block境界の
blend、smooth、transition penaltyは加えない。

## 6. CVと評価

exp293のwell-grouped outer 5 foldsを再利用する。各foldでouter-valid wellsのtruthは、
feature/pair probability/Borda/selected candidate/predictionを保存しSHA固定した後にだけ読む。
全foldを結合してrow-level OOF RMSEを計算する。

比較:

- primary control: fixed anchor `exp226_w500_50_50`, RMSE `8.238331546`
- attainability bound: exp293 H512 oracle RMSE `3.683762664`（直接比較ではなく上限）
- historical diagnostic: exp264 row-hard selector RMSE `8.652531956`
- negative framework reference: exp348 H512 neural path-bank（別candidate bank/modelのため参考のみ）

必須readout:

- pooled/fold row RMSEとanchor delta
- `0--250`, `250--1000`, `1000+`, hidden-like spatial,
  hidden-like typewell-purgedのscope RMSE
- by-well improved/worsened数、delta p50/p90/p95/max、worst wells
- H512 top-1 exact accuracy、weighted/unweighted pair accuracy、NDCG@1、top-3 oracle coverage
- anchor fallback率、candidate choice count、block間switch count

科学的PASSはrequirements記載のRMSE/fold/scope/well-tail/technical全AND。rank accuracyは
機序readoutであり、それ単独でpromotionしない。

## 7. Truth-lateと再現性

順序は次で固定する。

1. candidate bank、fold、H512 block、88列schemaを読みSHA検証する。
2. target-free row featureとcandidate-block/pair featureを生成する。
3. schema/content SHAを凍結し、禁止列とcoverageを監査する。
4. outer-train truthだけを読み、label/weightを生成してmodelをfitする。
5. outer-valid featureからrankとselected predictionを生成しSHA固定する。
6. outer-valid truthを初めて読み、scoreとreadoutを作る。

seedは42。candidate/block/pair/fold/feature順はstable sortする。model manifest、各model SHA、
pair-table logical SHA、feature schema/content SHA、OOF prediction SHAを保存する。gzipはraw file
SHAとdecompressed content SHAを分け、後者を主証拠にする。Kaggle packageを将来作る場合は
bootstrap内configも検証する。

## 8. 禁止事項と停止条件

- 同じexp504内でH128/H256/whole-well/overlap/可変長を試さない。
- LambdaRank、pointwise regression、listwise lossとのwinner選択を行わない。
- pair weight、0.5 guard、LightGBM configをOOF結果で変更しない。
- candidate追加・削除・formula変更、PF/HMM/Beam再生成、control再学習を行わない。
- block選択後のsmooth、blend、boundary correction、well/row gateを行わない。
- downstream ML、inference、submissionへ自動的に進まない。
- FAIL時は救済gridをせずterminal closeする。別仮説は別expとしてユーザー確認を取る。

## 9. 現在の状態

2026-08-02の追加実装承認により、別名compact self-contained train source、候補Notebook、
contract testを実装した。同日の追加実行承認により、候補を正規train notebookへ採用し、
Kaggle CPU package/run、完了監視、train-side OOF記録へ進む。正規inference notebookは
placeholderのままで、inference、submission、downstream昇格は承認されていない。

実装では曖昧だった非primary readoutだけを次のように確定した。

- 88列のうち`ctx__` 22列はcandidate間同値guard後にshared contextとして9統計へ集約する。
- 残り66列をcandidate-specific `g_j`として9統計へ集約する。
- pair feature幅は`1,986`列。
- NDCG@1はMSE rankのlinear relevance `12-rank`を使う。promotion gateには使わない。

## 10. 実行結果と終端判断

Kaggle private CPU version 1（`id_no=129488458`）は5 CPU modelsを完走し、technical
gateを全PASSした。pooled RMSEは`8.114276980`、anchor `8.238331546`比
`-0.124054566 ft`で平均改善条件を満たした。一方、非劣化foldは`3/5`、hidden-like
spatial / typewell-purgedは`+0.285759 / +0.269833 ft`、by-well delta p95 / worstは
`+2.963656 / +16.799044 ft`で固定gateをFAILした。

全AND契約に従い、decisionは
`FAIL_TERMINAL_CLOSE_WITHOUT_HORIZON_LOSS_WEIGHT_OR_THRESHOLD_RESCUE`とする。exp504内で
horizon、loss、weight、model、threshold、guard、smooth/blend/gateを変更せず、再実行、
inference、submissionへ進まない。
