# exp270 exact HMM posterior mode candidate audit

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU aggregate version 4完了。direct mode候補はnegative、branch closed
- CV / Public LB / Private LB: posterior mean direct RMSE 11.938287 / 対象外 / 対象外
- 作成日: 2026-07-17
- 科学的な親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

exp209 exact HMM の posterior mean は多峰 posterior の mode 間を平均し、どの高確率 path にも対応しない中間 TVT path を作る場合がある。同じ posterior と transition grammar から marginal MAP、global Viterbi、top-K joint path を取り出せば、HMM 自体を変更せずに実在 mode path の候補 headroom を観測できる。

## 親と変更点

- exp209 の grid、41 rate states、transition、Gaussian GR emission、既知 prefix calibration、初期位置/rate prior、GR 補間を固定する。
- posterior mean を exp209 control として同じ forward-backward から再生成し、保存済み exp209 cache と `1e-5 ft` 以内で照合する。
- 行ごとの `marginal_map` と、joint `(TVT position, rate)` の exact global top-5 path を追加する。
- joint rank 1 が global Viterbi である。
- top-5 復元後、TVT grid-index sequence が完全一致する path は rate sequence が違っても重複排除する。5 本未満になっても backfill しない。
- exp223 の self-GR emission、exp236 の LightGBM unary、exp243/252 の PF seed medoid は入れない。

## leakage 境界

候補 generator は真の `TVT` を引数に取らない。raw horizontal frame から `TVT` を削除して forward-backward / MAP / top-K / dedup / path 診断を完了し、その後に未知 suffix の真値を id で結合する。真値を使う row / block / well oracle は診断専用で、oracle prediction は保存しない。

## 検証方針

- direct: posterior mean、marginal MAP、global Viterbi、重複排除済み top-K path
- posterior: std、周辺 mode mass、top1-top2 mode gap、path log posterior、top1 score gap
- path: pairwise distance、grid edge、rate switch、step、curvature、TVT path SHA
- metric: overall、MD distance、hidden-like、by-well、worst-well、focus well `11d0f5ac`
- oracle: row、block 128/256/512、whole-well、unique-best rate

## 実行コスト契約

- active exact-HMM variant: 1
- HMM well-runs: 773
- LightGBM config / fold / booster: 0 / 0 / 0
- GPU / inference / submission: false / false / false
- outer workers / Numba threads: 1 / 4
- top-K backpointer: 1 byte per time-position-rate-rank、well 単位処理、上限 6 GB guard

posterior mean は必要な posterior を得る同じ CPU HMM pass 内で再生成する。保存済み親を GPU で再学習するものではなく、親 cache は parity control としてのみ読む。

## 実行入口

- aggregate正本: `exp270_exact_hmm_posterior_mode_candidate_audit_train.py`
- shard正本: `exp270_exact_hmm_posterior_mode_candidate_audit_train_variant0.py` / `train_variant1.py`
- Kaggle notebook: canonical aggregateと2つのself-contained shard notebook
- inference notebook: disabled fail-closed guard
- Kaggle package: `kaggle/train/`（aggregate）、`kaggle/train_variant0/1/`（shards）

Kaggle Notebook 実行を正とする。ローカル full 実行は行っていない。

## 結果

- posterior mean RMSE 11.938287が最良。marginal MAPは12.592479（+0.654192 ft）、global Viterbiは15.551665（+3.613377 ft）で悪化した。
- hidden-like spatial / typewell-purgedでもposterior mean 12.564491 / 12.367244が最良だった。
- 教師値を使うall-mode oracleはrow 7.516850、block-128 7.567530、whole-well 8.536362。ただしmean / MAP / Viterbiの3候補だけでも7.517189 / 7.567871 / 8.536605で、top-2からtop-5の追加価値は最大0.000342 ftだった。
- 3,783,989 rows / 773 wells、exp209 parity max 0.0 ft、ID/order/finite/禁止列、全artifact SHAとprediction SHAの再計算はすべてPASSした。
- oracle prediction、selector、inference、submissionは生成していない。

## 次

posterior mode pathの直接採用、top-K bank拡張、selector、raw-test inference、submissionには進めず、exp270 branchを閉じる。oracleだけを根拠とする救済backlogは追加しない。

## 所見

version 3の保存dtype parity failureをversion 4で修正し、決定論的363 / 410 well shardとSHA固定aggregateを完了した。technical contractは成立したが、すべてのdirect mode候補がposterior meanを悪化させたため、実装不良ではなく科学仮説のnegative resultとして閉じた。大きなoracle headroomは診断専用で、deployable selector evidenceとは扱わない。
