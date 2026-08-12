# 設計

## アプローチ

exp072 deterministic full replay train cache を読み、既存 PF/Beam/likelihood-PF candidate を TVT 空間に戻す。同じ row に対し、raw train horizontal well の GR と finite prefix `TVT_input` だけから multi-scale self-GR candidate を作る。

候補集合に対して、候補別 RMSE / within threshold、oracle topK coverage、target-free rank score topK coverage、distance/tail bucket miss rate、by-well worst case を保存する。これは候補生成と ranking headroom の診断であり、LightGBM 学習や提出は行わない。

## 実験範囲

- 対象実験: `exp091_self_gr_likelihood_pf_beam_probe`
- Route: `pf_beam`
- 親実験: `exp090_lateral_self_gr_match_pseudotail_probe`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: horizontal self-GR candidate の candidate set への追加と target-free rank score
- 固定する変数: exp072 PF/Beam/likelihood-PF candidate、raw train cache、評価 row、submission policy

## 再現性設計

- seed policy: 新規乱数なし。exp072 cache 内 PF/Beam/likelihood-PF は exp072 の stable per-well seed 方針を継承する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。既存 exp072 deterministic cache を読むだけ。
- 並列処理と乱数の関係: 並列処理なし、self-GR candidate は deterministic な rolling mean / NCC / argmax。
- CPU/GPU runtime と deterministic flags: GPU 不要。Kaggle CPU で audit を実行する。
- train cache / test feature regeneration の SHA 記録方針: train cache file SHA と gzip decompressed SHA を summary JSON に記録する。test regeneration はこの実験では行わない。
- model manifest / prediction / submission SHA 記録方針: モデル・submission は作らない。candidate long gzip は診断生成物として保存する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --notebook train --strict` 後、generated metadata の GPU disabled / internet disabled / exp072 kernel source と bootstrap manifest SHA を確認する。

## リスク

- リークリスク: evaluation true TVT を candidate 生成や rank score に使うと leakage になる。補助実装では `target_tvt` を candidate long の `abs_error` 計算だけに使う。
- CV/LB 不一致リスク: train-side pseudo-tail coverage audit なので LB を直接予測しない。提出判断は後続 selector 実験で別途行う。
- ランタイム/メモリリスク: full candidate long は rows x candidates の gzip になる。exp072 cache 自体も大きいため Kaggle train package 上で実行する。
- 再現性リスク: exp072 cache が mount されない場合は失敗する。local cache が無い場合はローカル full run ではなく synthetic smoke test に留める。
