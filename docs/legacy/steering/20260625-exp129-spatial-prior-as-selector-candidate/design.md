# 設計

## アプローチ

exp099 v2 の PF/Beam/likelihood-PF candidate surface を固定入力にし、exp114 v1 の fold-safe spatial prior OOF artifact から `xy_plus_trajectory_shape_k8_prior_tvt` と `xy_only_k8_prior_tvt` を追加候補として結合する。

まず候補集合だけを監査する。base 5候補 (`pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`) と spatial 2候補を比較し、expanded oracle が base oracle をどれだけ改善するか、spatial candidate の oracle selection rate、true-error top2/top3/top5 への spatial 露出、distance / tail bucket / worst well / path continuity を見る。

次に exp101 の row-wise selector が切替過多だった反省を踏まえ、candidate-long predicted-error LightGBM を 1 系統だけ学習し、row-wise argmin と Viterbi switch penalty grid を比較する。Viterbi は soft average ではなく、候補 path の離散選択だけを滑らかにする。

## 実験範囲

- 対象実験: `exp129_spatial_prior_as_selector_candidate`
- Route: `ensemble`
- 親実験: `exp101_pf_candidate_ranker_or_nway_classifier`
- 診断親: `exp099_pf_multi_observation_likelihood_probe`, `exp114_spatial_neighbor_prior_signal_audit`, `exp118_spatial_neighbor_prior_confidence_gate_on_exp092`
- 変更する変数: selector candidate set に spatial prior TVT path を 2本追加し、candidate-specific spatial confidence feature と Viterbi smoothing を追加する。
- 固定する変数: exp099 v2 candidate cache、exp114 v1 fold-safe spatial OOF prior、GroupKFold by well、seed 42、base PF/Beam candidate values。

## 再現性設計

- seed policy: GroupKFold は project seed 42。candidate-long subsample は `np.random.default_rng(seed + fold)` の局所 RNG を使う。
- stochastic 処理の有無: 新規 candidate 生成の stochastic 処理なし。LightGBM histogram training と row subsample は seed 固定だが bitwise deterministic anchor とは扱わない。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規実行しない。上流 exp099 の保存済み cache を読む。
- 並列処理と乱数の関係: 乱数は fold ごとの row subsample のみ。thread scheduling で global RNG を消費しない。
- CPU/GPU runtime と deterministic flags: CPU notebook、GPU なし、internet off。LightGBM は CPU `n_jobs=-1`。
- train cache / test feature regeneration の SHA 記録方針: exp099 gzip raw SHA と decompressed SHA、schema SHA、exp114 gzip raw SHA と decompressed SHA、summary SHA を summary JSON に記録する。
- model manifest / prediction / submission SHA 記録方針: fold model SHA、model manifest SHA、OOF selected prediction raw/decompressed SHA、variant 別 prediction SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に metadata の `enable_gpu=false`、kernel sources、bootstrap support files を確認する。

## リスク

- リークリスク: exp114 OOF spatial prior は fold-safe だが、selector label と scoring には true TVT を使う。true TVT は学習 label と評価だけに限定し、candidate generation / selector feature source には入れない。
- CV/LB 不一致リスク: train-side OOF だけでは raw-test/full-train regeneration の一致を保証しない。positive result は parity follow-up が必要。
- ランタイム/メモリリスク: 3.78M rows x 7 candidates の long validation frame が大きい。train rows per fold を cap し、モデルは predicted-error ranker 1 系統に絞る。
- 再現性リスク: LightGBM CPU training は完全な deterministic submission anchor ではない。結果は selector audit として扱う。
