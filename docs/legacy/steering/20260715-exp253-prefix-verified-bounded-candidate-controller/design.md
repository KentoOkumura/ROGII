# 設計

## 仮説

legalなmasked-prefix pseudo-holdoutはwellごとのcandidate相対順位を推定でき、balanced gateと
bounded moveを併用すればhard selectorより安全にその信号を利用できる。

## アプローチ

各 train well の公式 `TVT_input` prefix を `0.50 / 0.65 / 0.75` で短縮し、既存 exp072
candidate generator を同じ seed contract で再実行する。各 cut から公式 prefix 末尾までを legal
pseudo-holdout とし、candidate ごとに RMSE を計算する。

candidate score は公開 notebook と同じく母標準偏差を使う
`median(cut RMSE) + 0.10 * std(cut RMSE)` とする。consistency は各 cut の局所 best が
default candidate を 0.25 ft 以上上回った cut の割合とする。
default candidate は raw-test 再生成可能な既存 `likpf_mean` とし、best candidate の gain、second
margin、cut consistency から balanced commitment を計算する。official tail では保存済み exp238
OOF を base、exp072 official-start candidate path を correction direction とし、fade-in ramp、
`alpha <= 0.40`、`abs(move) <= 30 ft` を適用する。

Stage 0 は sorted 32 wells で runtime、row alignment、mask、score、move の contract を監査する。
Stage 1 は同じ固定設定を 773 wells へ広げるだけで、parameter grid は行わない。Stage 0実測から
単一CPU kernelは約22.2時間と見積もられるため、stable SHA256 well moduloで4 shardへ分割する。
各shardは全773 raw wellsをimputerへ渡し、評価wellだけを分割するため、wellごとの候補・seed・予測は
単一Stage 1と同じに保つ。shard metricsは採用判定に使わず、row-level OOF結合後にglobal RMSEを再計算する。

## 実験範囲

- 対象実験: `exp253_prefix_verified_bounded_candidate_controller`
- Route: `ensemble`
- 親実験: base `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`、candidate/replay `exp072_exp063_full_replay_feature_cache`
- 変更する変数: prefix candidate score に基づく well-level commitment と bounded correction の追加だけ
- 固定する変数: base OOF、candidate family、PF/Beam/likelihood-PF実装、seed、cut fractions、balanced profile、fold manifest、distance buckets、hidden-like assignment
- 実行だけの分割: 4 CPU shards。scientific variantは1、model config/fold/boosterは0のまま。
- 採用guard変更: worst-well回帰と補正well勝率は監視するが拒否条件にしない。overall、near、long-tail、hidden-like、fold stabilityは維持する。

## 再現性設計

- seed policy: exp072 の `stable_seed(feature_family, split/request, well)` を継承し、request id は source well と cut fraction の SHA256 から作る。
- stochastic 処理の有無: PF / likelihood-PF は stochastic。controller、score、ramp、clip は deterministic。
- PF/Beam / likelihood-PF / seed bagging の有無: 既存 PF/Beam/likelihood-PF を replay する。新規 seed bagging は追加しない。
- 並列処理と乱数の関係: 各shard内は`n_jobs=1`を維持する。shard分割はimmutable well idのSHA256で固定し、request idとper-request seedを変えない。
- CPU/GPU runtime と deterministic flags: CPU only、GPU 0、LightGBM 0 booster。Kaggle internet disabled。
- train cache / test feature regeneration の SHA 記録方針: raw input inventory、official candidate cache、exp238 OOF、masked-prefix score table、controller OOF の content SHA を保存する。gzip は decompressed SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: 新規 model なし。OOF prediction SHA を保存し、inference/submit は Stage 1 guard 通過後まで作らない。
- Kaggle package bootstrap 確認方針: exp072 replay source、exp115 hidden-like assignment、config の bootstrap 内 SHA と loose file SHA を比較する。

## リスク

- リークリスク: cut 後の `TVT` が candidate generation に入ると致命的。masked `TVT_input` と source-well exclusionをassertし、truthはscore/evaluation後にだけjoinする。
- CV/LB 不一致リスク: exp160でselector confidenceのCV改善がLB悪化した。overallだけでなくhidden-like、long-tail、fold、worst-well guardを必須とし、Stage 1通過前はsubmitしない。
- ランタイム/メモリリスク: Stage 0は32 wells / 3,313.476秒で、単一Stage 1は約22.2時間。4 shardを各約5.6時間見込みとし、aggregateはOOFを結合してrow-level RMSEを計算する。
- 再現性リスク: exp072 PFのNumba global RNGをthread並列すると順序依存になり得る。`n_jobs=1`とstable request well idで固定し、rerun SHA確認前はdeterministic anchorと呼ばない。
