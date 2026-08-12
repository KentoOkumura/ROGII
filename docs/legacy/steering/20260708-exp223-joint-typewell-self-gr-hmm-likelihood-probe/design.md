# 設計

## アプローチ

`exp209` exact HMM をベースにする。HMM grid、transition、typewell GR emission、band 設定は固定し、追加するのは self-GR motif likelihood のみ。

各 well で raw horizontal GR から local window descriptor を作る。finite `TVT_input` prefix rows を anchor とし、評価 row の descriptor と prefix anchor descriptor を照合する。top-k prefix anchor の `TVT_input` 周辺に Gaussian mixture を置き、HMM state TVT grid 上の `centered_logL_self_GR` を作る。

HMM emission は次の形にする。

```text
logL_total = logL_typewell_GR + alpha * quality_self * clip(centered_logL_self_GR, -c, c)
```

初回 run は `boost_only` だけを使い、正の clipped boost だけを足す。typewell と self-GR が矛盾する row を強く罰しないため、runtime-limited な初回探索として扱いやすい。`symmetric` は正負両側を clipped に足し、self-GR が明確に否定する state を弱く落とす後続候補に回す。

## 実験範囲

- 対象実験: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`
- Route: `ensemble`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: self-GR emission surface、`alpha`、clip、boost mode
- 固定する変数: HMM grid / transition / typewell GR emission、exp072 comparison baseline、raw train data、no model training

## 再現性設計

- seed policy: HMM と self-GR matching は RNG 不使用。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp072 saved cache を比較基準として読むだけ。
- 並列処理と乱数の関係: `joblib` outer well parallel は RNG を消費しない。浮動小数の微小差はあり得るため deterministic submission anchor にはしない。
- CPU/GPU runtime と deterministic flags: CPU-only、GPU 無効。
- train cache / test feature regeneration の SHA 記録方針: Kaggle train 後に self-GR HMM train feature gzip の raw SHA と decompressed SHA を記録する。raw-test regeneration は未実装。
- model manifest / prediction / submission SHA 記録方針: 学習モデル、prediction、submission は生成しない。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --strict` 後、metadata と bootstrap 内 config を含む generated package を検証する。

## リスク

- リークリスク: self-GR anchor に unknown-suffix true TVT を使うと leakage になる。実装では finite `TVT_input` prefix だけを center source にする。
- CV/LB 不一致リスク: train-side diagnostic であり、raw-test port / submit は行わない。positive でも hidden-like、worst-well、raw-test parity が必要。
- ランタイム/メモリリスク: 初回 active 2 variants でも HMM run は exp209 の約2倍になる。初回実行は CPU-only で、追加 alpha / symmetric は結果と runtime を見て別 run に分ける。
- 再現性リスク: HMM outer parallel の微小浮動小数差。採用時は decompressed content SHA と metrics を記録し、deterministic anchor とは呼ばない。
