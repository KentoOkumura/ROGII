# 設計

## アプローチ

exp250 Stage 1 の保存済み candidate-segment を `(well, segment_id, candidate)` で
real / shuffled に one-to-one pair し、bad/good weight、segment 範囲、paired risk 差が
一致することを確認する。distance readout は exp250 group artifact の distance contribution
を使い、real / shuffled の base weight が同一であることを再確認する。

near pooled AUC の交絡を分けるため、各 distance x candidate family の AUC を
`bad_weight * good_weight` で集約した within-family conditional AUC を算出する。
family x well についても同じ pair mass を使い、各 stratum の AUC 差を
`pair_mass / total_pair_mass` で加法分解する。これにより寄与の総和が全 family x well
conditional AUC 差と一致する。

## 実験範囲

- 対象実験: `exp256_segment_local_corridor_near_bucket_signal_attribution_readout`
- Route: `pf_beam`
- 親実験: `exp250_segment_local_negative_space_gr_corridor_audit`
- 変更する変数: なし。保存済み診断の集計面だけを追加する。
- 固定する変数: exp250 candidate / segment / corridor / risk / bad threshold / real-shuffled control。

## 生成物

- distance paired AUC と distance conditional summary
- 0--100 / 100+ / all-distance scope summary
- family x well attribution と family / well summary
- risk=1.0 saturation summary
- distance AUC、family 寄与、risk saturation plot
- input manifest、summary JSON、`metrics.json`

## 再現性設計

- seed policy: 新規乱数なし。入力順を stable sort し deterministic 集計する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 再実行なし。exp250 保存生成物だけを読む。
- 並列処理と乱数の関係: single process、global RNG 不使用。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU/internet off。
- input SHA: candidate-segment gzip は decompressed SHA、他は raw SHA を fail-closed 検証する。
- model manifest / prediction / submission SHA: model・prediction・submission を生成しないため対象外。
- Kaggle package bootstrap: source / loose package / bootstrap 内 config の一致を push 前に確認する。
- deterministic anchor: prediction anchor ではない。固定入力に対する diagnostic determinism だけを主張する。

## リスク

- リークリスク: candidate error は exp250 で固定済みの評価 label。新しい予測・selector・rule を作らない。
- 統計リスク: pooled AUC は family / distance / well の base rate に交絡するため conditional readout と分離する。
- estimability: bad または good が 0 の stratum は AUC を算出せず、欠落数を明示する。
- overlap weight: segment overlap により row weight は unique row 数ではない。exp250 contribution contract 内の相対 weight としてのみ解釈する。
- 再現性リスク: upstream exp250 / exp072 の stochastic provenance を継承し、submission anchor とは扱わない。

