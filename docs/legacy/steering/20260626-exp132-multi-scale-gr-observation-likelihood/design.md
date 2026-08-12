# 設計

## アプローチ

exp099 の構造を引き継ぎ、exp072 の固定 candidate surface を target-free に再採点する。各候補 TVT を同一 well の finite prefix `TVT_input` 上の最近傍位置へ写し、評価 row 周辺の GR と prefix 側 GR を multi-scale に比較する。

score は以下から作る。

- smoothed GR MAE
- offset vector NCC
- local z-score MAE
- derivative MAE
- rolling derivative energy MAE
- prefix TVT range 外 penalty
- shifted decoy score gap

保存する診断候補は `msgr_top1`、`msgr_top2`、softmax weighted TVT、`likpf_mean` との軽い blend、低頻度 `msgr_gate_*`。直接置換の良し悪しだけでは採用せず、topK coverage、low-switch gate、後続 ML confidence feature の材料として使えるかを見る。

## 実験範囲

- 対象実験: `exp132_multi_scale_gr_observation_likelihood`
- Route: `pf_beam`
- 親実験: `exp099_pf_multi_observation_likelihood_probe`
- 変更する変数: target-free multi-scale GR observation likelihood scorer と、その feature / gate diagnostics。
- 固定する変数: exp072 feature cache、既存 PF/Beam/likelihood-PF 候補生成、LightGBM/ML prediction、submission policy、supervised ranker の有無。

## 再現性設計

- seed policy: exp132 内では新規乱数なし。
- stochastic 処理の有無: exp132 内はなし。upstream exp072 PF/Beam cache は stochastic 由来だが、この実験では再生成しない。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規実行なし。候補列を読み込んで target-free likelihood を計算するだけ。
- 並列処理と乱数の関係: 並列 RNG なし。well ごとの deterministic loop。
- CPU/GPU runtime と deterministic flags: CPU audit。GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: source cache file SHA と gzip decompressed content SHA、schema SHA、生成 feature cache SHA を summary JSON に記録する。
- model manifest / prediction / submission SHA 記録方針: model / prediction / submission は作らないため対象外。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後、metadata と bootstrap 内 config を正とする。

## リスク

- リークリスク: 評価 row の GR は target-free 入力として使うが、true TVT は scorer に使わない。true TVT は scoring / oracle / diagnostics 専用。
- CV/LB 不一致リスク: train-side diagnostic のため LB はない。改善しても inference port 前に raw-test parity と hidden-compatible feature availability を別途確認する。
- ランタイム/メモリリスク: full cache 3.8M rows x 既存 5 候補 x offsets x windows の scoring。candidate long 保存は大きくなるため、debug では `max_rows` と `save_candidate_long=false` を使う。
- 再現性リスク: deterministic submission anchor ではない。upstream cache content SHA と生成 feature cache SHA を根拠として記録する。
