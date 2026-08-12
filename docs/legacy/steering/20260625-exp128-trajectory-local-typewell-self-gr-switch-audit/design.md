# 設計

## アプローチ

exp099 の train-side OOF/PF candidate cache を読み、各 well の評価区間 window ごとに次を作る。

- `typewell_cost`: `likpf_mean` trajectory の局所 TVT 値を typewell GR へ写像し、horizontal evaluation GR window と正規化 RMSE で比較する。
- `self_cost`: 同じ horizontal well の visible prefix GR window だけを検索し、evaluation GR window と正規化 RMSE で比較する。
- `self_gr_prefix_prior_tvt`: best prefix window の finite `TVT_input` と local prefix slope から、query MD 位置の TVT prior を外挿する。
- switch / blend candidate: `self_cost` が `typewell_cost` より十分低く、prefix coverage 条件を満たす row だけ、既存 PF/Beam / likPF 候補から self prior へ hard switch または soft blend する。

## 実験範囲

- 対象実験: `exp128_trajectory_local_typewell_self_gr_switch_audit`
- Route: `ensemble`
- 親実験: `exp099_pf_multi_observation_likelihood_probe`
- 診断親: `exp091`、`exp093`、`exp119`、`exp120`、`exp125`
- 変更する変数: local GR window radius/stride、self-vs-typewell cost gap、hard switch threshold、soft blend weight。
- 固定する変数: exp099 candidate cache、well-level OOF rows、raw train horizontal/typewell files、baseline candidates。

## 再現性設計

- seed policy: 新規処理は RNG 不使用。window scan と prefix match は deterministic。
- stochastic 処理の有無: exp128 自体はなし。上流 exp099 PF/Beam / likelihood-PF cache は stochastic 由来の可能性がある。
- PF/Beam / likelihood-PF / seed bagging の有無: 直接再生成せず exp099 cache を読む。
- 並列処理と乱数の関係: `num_workers=1`、global RNG 不使用。
- CPU/GPU runtime と deterministic flags: CPU-only、GPU 不使用。
- train cache / test feature regeneration の SHA 記録方針: exp099 gzip cache の raw SHA と decompressed SHA、schema SHA、row/well count を summary JSON に記録する。raw-test regeneration はこの実験では行わない。
- model manifest / prediction / submission SHA 記録方針: 学習モデルなし。OOF gzip raw/decompressed SHA と feature schema SHA を記録する。submission SHA は対象外。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train --strict` 後、kernel metadata と bootstrap 内 config を確認して push する。

## リスク

- リークリスク: self-GR prefix だけを使う制約を破ると、同じ well の evaluation-zone true TVT を間接利用する危険がある。実装では finite `TVT_input` prefix row のみを source とする。
- CV/LB 不一致リスク: train-side self-GR motif は同じ well 内で過度に合いやすく、hidden test へ転移しない可能性が高い。改善時も raw-test parity audit を別途作る。
- ランタイム/メモリリスク: full 3.78M rows に window scan を行うため Kaggle CPU runtime が長くなる。stride と radius を config で固定し、single-process deterministic にする。
- 再現性リスク: 上流 exp099 cache の生成は stochastic 由来。exp128 は input SHA を記録するが deterministic submission anchor とは扱わない。
