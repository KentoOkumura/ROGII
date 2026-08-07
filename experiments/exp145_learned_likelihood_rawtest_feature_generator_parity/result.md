# exp145_learned_likelihood_rawtest_feature_generator_parity 結果

## 仮説

exp111 保存済み learned likelihood model を target-free transform として使えば、exp112 learned likelihood `ml_features` を full train と raw test に schema parity を保って再生成できる。

## 設定

- 親: exp144_learned_likelihood_hidden_stress_and_rawtest_parity
- model parent: exp111_learned_pf_observation_likelihood_probe
- feature parent: exp112_learned_pf_likelihood_weight_or_feature_followup
- cache parent: exp099_pf_multi_observation_likelihood_probe
- 検証: schema / coverage / SHA parity
- メトリック: schema_parity_and_coverage
- シード: 42

## 結果

Kaggle train v2 / inference v3 完了。提出なし。

- Train kernel: `kentookumura/exp145-train` v2
- Inference kernel: `kentookumura/exp145-inference` v3
- Train output: `kaggle/output/train_v2`
- Inference output: `kaggle/output/inference_v3`

| メトリック | 値 |
| --- | --- |
| full-train rows | 3,783,989 |
| full-train wells | 773 |
| raw-test rows | 14,151 |
| raw-test wells | 3 |
| schema columns | 51 |
| schema parity | pass |
| schema mismatch rows | 0 |
| Public LB | - |

### 生成物 SHA

- full-train `ml_features` decompressed SHA: `e1c276d69e9355f6c03c18ac51a0883ee99ec6d80d040a5c62e5d55048bb7456`
- raw-test `ml_features` decompressed SHA: `61a21bb1b52eb8ae2d242c758732fe3cb10682d9d8b147ebe4a40f75419704c8`
- raw-test likelihood long decompressed SHA: `4b50d801be8d3e0977b6699eea5110321d55df15b9ccfa46998a02d1f8b3fdf6`
- feature schema SHA: `b1285777136304d65c927d28a1d0f57d68c0e45a9c4d8a0cbaaff054e4315cf8`
- schema parity SHA: `737455382dad20f6e94a6c196be2a2ed45028ec22eacebadc9641dca5249f2b0`

## 再現性

- deterministic anchor: false
- seed policy: exp145 自体は新規 RNG なし。raw-test PF/Beam replay は exp072 stable per-well seed を継承。
- kernel version: train v2 / inference v3
- feature content SHA: 上記 decompressed SHA を主証拠とする
- model SHA / manifest SHA: classifier `c4c65558ae07fc74735d7c41f7cdc605350112409273aa314cfb0122ed1e9f29`、expected-error `308242bf901c3db167e97b4750d389aa5b69cab492fe61cff2eeff82133725f3`、manifest `178e8b3124b817a2b230080fc041aaaee1b06941e5a4223a68cc31bf26e68010`
- prediction SHA: 対象外
- submission SHA: 対象外
- rerun result: 未実行

## 解釈

raw-test feature regeneration missing と full-train coverage blocker は解消した。exp112 learned likelihood `ml_features` は full train 773 wells と raw test 3 wells に target-free に再生成でき、exp112 schema 51列と完全一致した。

ただし、これは feature generator / parity audit であり、提出候補ではない。exp111 saved model は fold0 model で、学習時 imputation medians が保存されていないため、batch median imputation 制約は残る。次に使う場合は exp092 系 add-only feature として再学習し、worst-well / exp115 hidden-like stress / raw-test submission flow を別途確認する。

## 次

1. exp145 full-train/raw-test cache を使って、exp127 の learned likelihood feature family を exp092 full-row add-only 実験として再評価する。
2. 改善しても direct submit せず、worst-well regression、exp115 hidden-like stress、raw-test inference flow を確認する。
