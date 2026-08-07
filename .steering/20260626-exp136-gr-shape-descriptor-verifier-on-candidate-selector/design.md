# 設計

## アプローチ

exp102 を親に、exp101 の OOF candidate error ranker score surface を復元する。exp131 の GR shape descriptor 計算をこの実験内に移植し、raw train GR と visible `TVT_input` prefix から候補別 descriptor score を再計算する。

selector 本体は置き換えない。`likpf_mean` を default とし、exp101 が `pf_ancc` / `beam_mean` を選ぶ高信頼 row だけを候補にする。その上で descriptor score floor、descriptor-vs-`likpf_mean` margin、switch-rate cap、PF seed std cap、`likpf_mean` からの TVT 差分 cap、minimum segment length を満たす場合だけ切り替える。

## 実験範囲

- 対象実験: `exp136_gr_shape_descriptor_verifier_on_candidate_selector`
- Route: `pf_beam`
- 親実験: `exp102_confidence_gated_likpf_fallback_on_exp101`
- 参照実験: `exp101_pf_candidate_ranker_or_nway_classifier`、`exp131_gr_shape_descriptor_matching_ablation`
- 変更する変数: descriptor verifier score、descriptor threshold、minimum segment length、switch-rate cap
- 固定する変数: exp099 v2 candidate cache、exp101 model manifest/schema、候補集合、GroupKFold by well、`likpf_mean` default

## 再現性設計

- seed policy: exp101 と同じ GroupKFold seed 42 と sampled long-frame seed を再利用。descriptor 計算自体は deterministic。
- stochastic 処理の有無: 新規学習や新規 RNG はなし。exp101 booster と upstream PF/Beam cache は既存 stochastic artifact。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規 PF/Beam 実行なし。exp099/101 の保存済み artifact を読む。
- 並列処理と乱数の関係: descriptor は single-process deterministic loop。
- CPU/GPU runtime と deterministic flags: GPU なし、CPU notebook。
- train cache / test feature regeneration の SHA 記録方針: exp099 cache は raw/decompressed SHA、exp101 schema/manifest/model SHA、descriptor well summary SHA、prediction SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: exp101 resolved manifest と保存対象 OOF predictions の SHA を記録する。submission は生成しない。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --notebook train --strict` で kernel sources と metadata を確認する。

## リスク

- リークリスク: descriptor は raw GR と visible prefix のみを使い、true TVT / oracle / error label を gate 条件に使わない。
- CV/LB 不一致リスク: train-side posthoc audit のため、改善しても raw-test parity と worst-well guard なしでは inference port しない。
- ランタイム/メモリリスク: raw GR descriptor を全 773 wells x 5 candidates に再計算する。exp131 と同系統で数千秒級の Kaggle train runtime を想定する。
- 再現性リスク: upstream artifact は保存済み notebook output に依存するため、SHA を記録する。
