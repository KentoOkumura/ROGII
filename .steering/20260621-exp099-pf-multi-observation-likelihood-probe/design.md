# 設計

## アプローチ

exp093 の候補集合を固定し、`pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb` を target-free に再採点する。各候補 TVT を同一 well の finite prefix TVT_input 上の最近傍位置へ写し、評価 row 周辺の GR ベクトルと prefix 側 GR ベクトルを複数 offset で比較する。

score は GR MAE、正規化相関、prefix TVT range 外への penalty から作る。候補生成は次を保存する。

- 既存 5 候補それぞれの `multiobs_score_*`
- `multiobs_top1`
- score softmax weighted TVT candidates
- `likpf_mean` と `multiobs_top1` の軽い blend candidates

結果は candidate RMSE、oracle topK、target-free rank score topK、bucket metrics、by-well metrics で読む。採用判断は提出候補ではなく、後続 scorer / ranker feature として使う価値があるかに限定する。

## 実験範囲

- 対象実験: `exp099_pf_multi_observation_likelihood_probe`
- Route: `pf_beam`
- 親実験: `exp093_pf_candidate_coverage_then_ranker_audit`
- 変更する変数: target-free multi-observation likelihood scorer と、その top1 / softmax / likPF blend 診断候補。
- 固定する変数: exp072 feature cache、既存 PF/Beam/likelihood-PF 候補生成、LightGBM/ML prediction、submission policy、supervised ranker の有無。

## 再現性設計

- seed policy: exp099 内では新規乱数なし。
- stochastic 処理の有無: exp099 内はなし。upstream exp072 PF/Beam cache は stochastic 由来だが、この実験では再生成しない。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規実行なし。候補列を読み込んで target-free likelihood を計算するだけ。
- 並列処理と乱数の関係: 並列 RNG なし。well ごとの deterministic loop。
- CPU/GPU runtime と deterministic flags: CPU audit。GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: source cache file SHA と gzip decompressed content SHA、schema SHA を summary JSON に記録する。
- model manifest / prediction / submission SHA 記録方針: model / prediction / submission は作らないため対象外。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後、metadata と bootstrap 内 config を正とする。

## リスク

- リークリスク: 評価 row の GR は target-free 入力として使うが、true TVT は scorer に使わない。true TVT は scoring 専用。
- CV/LB 不一致リスク: train-side diagnostic のため LB はない。改善しても inference port 前に raw-test parity と hidden-compatible feature availability を別途確認する。
- ランタイム/メモリリスク: full cache 3.8M rows x 既存 5 候補の scoring。candidate long 保存は大きくなるため、debug では `max_rows` と `save_candidate_long=false` を使う。
- 再現性リスク: deterministic submission anchor ではない。upstream cache content SHA を根拠として記録する。
